import datetime
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import requests

DATA_DIR = "Codes/data" if os.path.exists("Codes/data") else "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(
    DATA_DIR, "live_fno_ultimate_breakouts.json"
)
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")
EOD_RESULTS_FILE = "fno_champion_scan_results.json"
MIN_TURNOVER_CR = 10.0

# ---------------------------------------------------------
# TELEGRAM CREDENTIALS
# ---------------------------------------------------------
DEFAULT_BOT_TOKEN = "9160227761:AAE2zmSpvqttJl3_KKl3jjyKkMg-CopgtSY"
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN
).strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID_ULTIMATE", "").strip()

if not TELEGRAM_CHAT_ID:
  # Check local_secrets.json fallback
  for sec_path in ["local_secrets.json", "../local_secrets.json"]:
    if os.path.exists(sec_path):
      try:
        with open(sec_path, "r", encoding="utf-8") as f:
          cfg = json.load(f)
          TELEGRAM_CHAT_ID = cfg.get(
              "TELEGRAM_CHAT_ID_ULTIMATE",
              cfg.get("TELEGRAM_CHAT_ID", cfg.get("7860290981", "")),
          )
          if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == DEFAULT_BOT_TOKEN:
            TELEGRAM_BOT_TOKEN = cfg.get(
                "TELEGRAM_BOT_TOKEN_ULTIMATE", DEFAULT_BOT_TOKEN
            )
          if TELEGRAM_CHAT_ID:
            break
      except Exception:
        pass

# Fallback default chat id if not set in secrets
if not TELEGRAM_CHAT_ID:
  TELEGRAM_CHAT_ID = "7860290981"

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EXCLUDED_SYMBOLS = {
    "LIQUIDCASE",
    "LIQGRWBEES",
    "LIQUIDBEES",
    "GOLDBEES",
    "NIFTYBEES",
    "BANKBEES",
    "SILVERBEES",
}


def get_ist_now():
  return datetime.datetime.now(IST)


def send_telegram_alert(message: str):
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print(f"[SIMULATION ALERT]:\n{message}\n")
    return
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TELEGRAM_CHAT_ID,
      "text": message,
      "parse_mode": "HTML",
      "disable_web_page_preview": True,
  }
  try:
    r = requests.post(url, json=payload, timeout=10)
    if not r.json().get("ok"):
      print(f"❌ Telegram API Error: {r.text}")
  except Exception as e:
    print(f"❌ Telegram Dispatch Error: {e}")


def get_market_session_elapsed():
  now = get_ist_now()
  market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
  market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
  if now < market_open:
    return 1
  elif now > market_close:
    return 375
  else:
    return max(1, min(375, int((now - market_open).total_seconds() / 60)))


def get_u_curve_expected_volume_fraction(elapsed_mins):
  if elapsed_mins <= 0:
    return 0.02
  elif elapsed_mins <= 75:
    return 0.05 + (elapsed_mins / 75.0) * 0.27
  elif elapsed_mins <= 225:
    return 0.32 + ((elapsed_mins - 75.0) / 150.0) * 0.28
  elif elapsed_mins <= 330:
    return 0.60 + ((elapsed_mins - 225.0) / 105.0) * 0.20
  else:
    return 0.80 + ((elapsed_mins - 330.0) / 45.0) * 0.20


class LivePriceClient:

  def __init__(self):
    self.session = requests.Session()
    self.session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })

  def fetch_live_quote(self, symbol):
    # 1. Google Finance Primary
    try:
      url = f"https://www.google.com/finance/quote/{symbol}:NSE"
      r = self.session.get(url, timeout=4)
      if r.status_code == 200:
        html = r.text
        p_m = re.search(r'data-last-price="([\d\.]+)"', html)
        if p_m:
          ltp = float(p_m.group(1))
          d_high, d_low = ltp, ltp
          rng_m = re.search(
              r"Day range.*?₹([\d\.,]+)\s*-\s*₹([\d\.,]+)", html, re.DOTALL
          )
          if rng_m:
            d_low = float(rng_m.group(1).replace(",", ""))
            d_high = float(rng_m.group(2).replace(",", ""))

          d_vol = 0.0
          vol_m = re.search(
              r'Volume</div><div class="P6K39c">([\d\.,MKmk]+)</div>', html
          )
          if vol_m:
            v_str = vol_m.group(1).replace(",", "").upper()
            if "M" in v_str:
              d_vol = float(v_str.replace("M", "")) * 1_000_000
            elif "K" in v_str:
              d_vol = float(v_str.replace("K", "")) * 1_000
            else:
              d_vol = float(v_str)

          if ltp > 0:
            return {
                "ltp": ltp,
                "day_high": max(d_high, ltp),
                "day_low": min(d_low, ltp),
                "day_vol": d_vol,
            }
    except Exception:
      pass

    # 2. Yahoo Finance Fallback
    try:
      url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS?interval=1d&range=1d"
      r = self.session.get(url, timeout=4)
      if r.status_code == 200:
        meta = (
            r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
        )
        ltp = float(meta.get("regularMarketPrice", 0) or 0)
        d_high = float(meta.get("regularMarketDayHigh", ltp) or ltp)
        d_low = float(meta.get("regularMarketDayLow", ltp) or ltp)
        d_vol = float(meta.get("regularMarketVolume", 0) or 0)
        if ltp > 0:
          return {
              "ltp": ltp,
              "day_high": max(d_high, ltp),
              "day_low": min(d_low, ltp),
              "day_vol": d_vol,
          }
    except Exception:
      pass
    return None


def clean_and_prepare(raw_data):
  if not raw_data or not isinstance(raw_data, list):
    return []
  date_map = {}
  for r in raw_data:
    if not isinstance(r, dict):
      continue
    t = str(r.get("time", ""))[:10]
    c = float(r.get("close", 0) or 0)
    if not t or c <= 0:
      continue
    v = float(r.get("volume", 0) or 0)
    dv = float(r.get("delivery_vol", 0) or 0)
    dp = float(r.get("deliv_pct", 0) or 0)
    if dp > 0 and dv == 0:
      dv = v * (dp / 100.0)
    if v > 0 and dv > v:
      dv = v
    date_map[t] = {
        "time": t,
        "open": float(r.get("open", c) or c),
        "high": float(r.get("high", c) or c),
        "low": float(r.get("low", c) or c),
        "close": c,
        "volume": v,
        "delivery_vol": dv,
        "deliv_pct": (dv / v * 100.0) if v > 0 else 0.0,
    }
  clean = [date_map[k] for k in sorted(date_map.keys())]

  known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
  for i in range(len(clean) - 1, 0, -1):
    prev_c = clean[i - 1]["close"]
    curr_o = clean[i]["open"]
    if prev_c > 0 and curr_o > 0:
      ratio = prev_c / curr_o
      adj = None
      if ratio >= 1.35:
        for k in known_multipliers:
          if abs(ratio - k) / k < 0.15:
            adj = k
            break
        if not adj and 1.70 <= ratio <= 2.30:
          adj = 2.0
        elif not adj and 4.30 <= ratio <= 5.50:
          adj = 5.0
      if adj:
        for j in range(0, i):
          clean[j]["open"] = round(clean[j]["open"] / adj, 2)
          clean[j]["high"] = round(clean[j]["high"] / adj, 2)
          clean[j]["low"] = round(clean[j]["low"] / adj, 2)
          clean[j]["close"] = round(clean[j]["close"] / adj, 2)
          clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj
          clean[j]["volume"] = clean[j]["volume"] * adj
  return clean


def load_nifty_benchmark():
  all_json = glob.glob("**/*.json", recursive=True)
  for p in all_json:
    bname = os.path.basename(p).lower()
    if bname in ["nifty.json", "nifty50.json"]:
      try:
        with open(p, "r", encoding="utf-8") as fp:
          clean = clean_and_prepare(json.load(fp))
        if len(clean) > 40:
          df_bm = pd.DataFrame(clean)
          df_bm["p60"] = df_bm["close"].pct_change(60).fillna(0)
          return {
              str(r["time"]): float(r["p60"]) for _, r in df_bm.iterrows()
          }
      except Exception:
        pass
  return {}


# ---------------------------------------------------------
# ACTIVE POSITIONS MONITOR (EXIT 2.2 ENGINE)
# ---------------------------------------------------------
def monitor_exit2_2_active_positions(
    live_client: LivePriceClient, elapsed_mins: int
):
  if not os.path.exists(MY_TRADES_FILE):
    return
  try:
    with open(MY_TRADES_FILE, "r", encoding="utf-8") as f:
      trades = json.load(f)
  except Exception:
    return

  if not isinstance(trades, list) or not trades:
    return

  expected_fraction = get_u_curve_expected_volume_fraction(elapsed_mins)
  updated_trades = []

  for t in trades:
    if str(t.get("status", "OPEN")).upper() != "OPEN":
      updated_trades.append(t)
      continue

    sym = str(t.get("symbol", "")).upper().strip()
    entry_price = float(t.get("entry_price", 0.0))
    catastrophe_sl = float(t.get("hard_sl", entry_price * 0.92))
    peak_gain_pct = float(t.get("peak_gain_%", 0.0))
    peak_close = float(t.get("peak_close", entry_price))
    vdu_under_count = int(t.get("vdu_under_count", 0))

    ema20 = entry_price * 0.95
    sma20_vol = 50000.0
    atr_14 = entry_price * 0.02
    dist_count_10d = 0

    sym_matches = glob.glob(f"**/{sym}.json", recursive=True)
    if sym_matches:
      try:
        with open(sym_matches[0], "r", encoding="utf-8") as fp:
          clean = clean_and_prepare(json.load(fp))
        if len(clean) >= 20:
          df_h = pd.DataFrame(clean)
          ema20 = float(
              df_h["close"].ewm(span=20, adjust=False).mean().iloc[-1]
          )
          vols = df_h["volume"].iloc[-20:].tolist()
          sma20_vol = float(sum(vols) / len(vols)) if vols else 50000.0
          prev_c = df_h["close"].shift(1)
          tr = pd.concat([
              df_h["high"] - df_h["low"],
              (df_h["high"] - prev_c).abs(),
              (df_h["low"] - prev_c).abs(),
          ],
                         axis=1).max(axis=1)
          atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
          dv_sma = (
              df_h["delivery_vol"].rolling(20, min_periods=1).mean().iloc[-1]
          )
          dp_avg = df_h["deliv_pct"].rolling(50, min_periods=5).mean().iloc[-1]
          c_rng = df_h["high"] - df_h["low"]
          rc = np.where(
              c_rng > 0, (df_h["close"] - df_h["low"]) / c_rng, 0.5
          )
          is_dist = (
              (df_h["close"] < df_h["close"].shift(1))
              & (
                  (df_h["delivery_vol"] >= 1.30 * dv_sma)
                  | (df_h["deliv_pct"] >= 1.15 * dp_avg)
              )
              & (rc <= 0.45)
          ).astype(int)
          dist_count_10d = int(
              is_dist.rolling(10, min_periods=1).sum().iloc[-1]
          )
      except Exception:
        pass

    quote = live_client.fetch_live_quote(sym)
    time.sleep(0.04)

    if not quote:
      updated_trades.append(t)
      continue

    ltp = quote["ltp"]
    day_high = quote["day_high"]
    day_low = quote["day_low"]
    day_vol = quote["day_vol"]

    curr_gain_pct = (
        ((ltp - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
    )
    peak_gain_pct = max(peak_gain_pct, curr_gain_pct)
    peak_close = max(peak_close, ltp)

    vol_pace = (day_vol / expected_fraction) / max(1.0, sma20_vol)
    bar_range = day_high - day_low
    range_closeness = (
        (ltp - day_low) / bar_range if bar_range > 0 else 0.50
    )

    exit_triggered = False
    exit_rule = ""

    # Rule 1: Catastrophe Stop
    if ltp < catastrophe_sl:
      exit_triggered = True
      exit_rule = (
          f"1. Catastrophe SL Breach (LTP ₹{ltp:,.2f} < Hard SL"
          f" ₹{catastrophe_sl:,.2f})"
      )
    # Rule 2: Climactic Churn Harvest
    elif peak_gain_pct >= 20.0 and vol_pace >= 2.0 and range_closeness <= 0.35:
      exit_triggered = True
      exit_rule = (
          f"2. Climactic Churn Harvest (Peak +{peak_gain_pct:.1f}%, Vol Pace"
          f" {vol_pace:.1f}x)"
      )
    # Rule 3: Distribution Cluster Breakdown
    elif dist_count_10d >= 3 and ltp < ema20:
      exit_triggered = True
      exit_rule = (
          f"3. Distribution Cluster ({dist_count_10d} Dist Days & LTP < 20"
          f" EMA ₹{ema20:,.2f})"
      )
    # Rule 4: Mega-Runner 15% Cushion Lock
    elif peak_gain_pct >= 50.0:
      cushion_stop = max(ema20, peak_close * 0.85)
      if ltp < cushion_stop:
        exit_triggered = True
        exit_rule = (
            f"4. Mega-Runner 15% Cushion Lock (LTP ₹{ltp:,.2f} < Cushion"
            f" ₹{cushion_stop:,.2f})"
        )
    # Rule 5: 20 EMA Heavy Vol Breakdown Engine
    elif ltp < ema20:
      if vol_pace >= 1.25:
        exit_triggered = True
        exit_rule = (
            f"5A. Heavy Volume 20 EMA Breach (LTP ₹{ltp:,.2f} < ₹{ema20:,.2f} |"
            f" Pace {vol_pace:.1f}x)"
        )
      else:
        vdu_under_count += 1
        atr_floor = ema20 - (1.5 * atr_14)
        pct_floor = ema20 * 0.95
        dynamic_leeway = min(pct_floor, atr_floor)
        guaranteed_be = entry_price * 1.015
        active_vdu_stop = max(guaranteed_be, dynamic_leeway)

        if ltp < active_vdu_stop:
          exit_triggered = True
          exit_rule = (
              f"5B. Dynamic VDU Floor Breach (LTP ₹{ltp:,.2f} < Floor"
              f" ₹{active_vdu_stop:,.2f})"
          )
        elif vdu_under_count > 5:
          exit_triggered = True
          exit_rule = "5B. 5-Session VDU Reclaim Failure"
    else:
      vdu_under_count = 0

    if exit_triggered and not t.get("exit_alert_sent", False):
      pnl_pct = round(curr_gain_pct, 2)
      exit_msg = (
          f"🚨 <b>EXIT 2.2 TRIGGER: CLOSE POSITION</b>\n"
          f"• <b>Symbol</b>: <code>#{sym}</code>\n"
          f"• <b>Rule Triggered</b>: <b>{exit_rule}</b>\n"
          f"• <b>Live NSE LTP</b>: ₹{ltp:,.2f} (<b>{pnl_pct:+.2f}%</b>)\n"
          f"• <b>Entry Price</b>: ₹{entry_price:,.2f}\n"
          f"• <b>20 EMA Floor</b>: ₹{ema20:,.2f}\n"
          f"• <b>Peak Unrealized</b>: +{peak_gain_pct:.2f}%\n"
          f"━━━━━━━━━━━━━━━━━━━━\n"
          f"🛑 <i>Action: Execute square-off order immediately.</i>"
      )
      send_telegram_alert(exit_msg)
      t["exit_alert_sent"] = True
      t["exit_price"] = ltp
      t["exit_reason"] = exit_rule

    t["peak_gain_%"] = round(peak_gain_pct, 2)
    t["peak_close"] = round(peak_close, 2)
    t["vdu_under_count"] = vdu_under_count
    updated_trades.append(t)

  try:
    with open(MY_TRADES_FILE, "w", encoding="utf-8") as f:
      json.dump(updated_trades, f, indent=2)
  except Exception:
    pass


# ---------------------------------------------------------
# TARGET CANDIDATE GENERATION (ULTIMATE FNO RULES)
# ---------------------------------------------------------
def build_live_targets():
  """Scans historical data to isolate stocks meeting the Accumulation Ratio ≥ 1.50,

  Shelf Spread ≤ 8.5%, Stage 2, RS+, and Turnover ≥ ₹10 Cr gates.
  """
  all_json = glob.glob("**/*.json", recursive=True)
  nifty_map = load_nifty_benchmark()

  stock_map = {}
  for p in all_json:
    bname = os.path.basename(p)
    sym = os.path.splitext(bname)[0].upper()
    if any(
        k in bname.lower()
        for k in [
            "nifty",
            "audit",
            "result",
            "trade",
            "secret",
            "package",
            "checkpoint",
        ]
    ):
      continue
    if sym not in stock_map or os.path.getsize(p) > os.path.getsize(
        stock_map[sym]
    ):
      stock_map[sym] = p

  targets = {}
  for sym, filepath in stock_map.items():
    if sym in EXCLUDED_SYMBOLS:
      continue
    try:
      with open(filepath, "r", encoding="utf-8") as fp:
        clean = clean_and_prepare(json.load(fp))
    except Exception:
      continue

    if len(clean) <
