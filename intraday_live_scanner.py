import os
import json
import time
import datetime
import requests
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. PATHS & TIME CONFIGURATION
# ==============================================================================
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

DATA_DIR = "data"
OUTPUT_JSON = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
S3_RESULTS = os.path.join(DATA_DIR, "swing3_scanner_results.json")
R2_RESULTS = os.path.join(DATA_DIR, "swing3_run2_non_nifty750_scanner_results.json")
MY_TRADES_FILE = os.path.join(DATA_DIR, "my_trades.json")

os.makedirs(DATA_DIR, exist_ok=True)

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip().replace('"', '').replace("'", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip().replace('"', '').replace("'", "")
if TG_TOKEN.lower().startswith("bot"):
    TG_TOKEN = TG_TOKEN[3:]

# ==============================================================================
# 2. TELEGRAM BROADCASTER (SAFE HTML ESCAPING)
# ==============================================================================
def send_telegram_msg(msg_text):
    if not TG_TOKEN or not TG_CHAT_ID or not msg_text:
        print("⚠️ Telegram skipped: Missing credentials or empty message.")
        return
    tg_url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": msg_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(tg_url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("📡 Telegram alert dispatched successfully (HTTP 200)")
        else:
            print(f"❌ Telegram API Error ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"❌ Telegram delivery network exception: {e}")

# ==============================================================================
# 3. REAL-TIME DATA QUOTE ENGINE (YFINANCE CLOUD-SAFE)
# ==============================================================================
def fetch_live_quotes_batch(symbol_list):
    if not symbol_list:
        return {}

    ticker_map = {f"{sym}.NS": sym for sym in symbol_list}
    tickers_str = " ".join(ticker_map.keys())
    print(f"📥 Fetching live cloud market data for {len(symbol_list)} symbols...")

    quotes = {}
    try:
        data = yf.download(
            tickers=tickers_str,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        for yf_sym, clean_sym in ticker_map.items():
            try:
                df = data if len(symbol_list) == 1 else data[yf_sym]
                df = df.dropna(how="all")
                if not df.empty:
                    last_row = df.iloc[-1]
                    ltp = float(last_row["Close"])
                    high = float(df["High"].max())
                    vol = float(df["Volume"].sum())
                    quotes[clean_sym] = {"ltp": ltp, "high": high, "vol": vol}
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Batch quote fetch error: {e}")

    # Fallback for individual tickers if missing
    for sym in symbol_list:
        if sym not in quotes:
            try:
                t = yf.Ticker(f"{sym}.NS")
                hist = t.history(period="1d", interval="1m")
                if not hist.empty:
                    quotes[sym] = {
                        "ltp": float(hist["Close"].iloc[-1]),
                        "high": float(hist["High"].max()),
                        "vol": float(hist["Volume"].sum()),
                    }
            except Exception:
                pass

    print(f"✅ Successfully received live quotes for {len(quotes)}/{len(symbol_list)} tickers.")
    return quotes

# ==============================================================================
# 4. WATCHLIST & CUMULATIVE STATE LOADERS
# ==============================================================================
def load_verified_watchlists():
    targets = {}
    for univ, path in [("NIFTY 750", S3_RESULTS), ("NON-N750", R2_RESULTS)]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                for item in d.get("Watchlist", []):
                    sym = item.get("symbol", "").strip().upper()
                    pivot = float(item.get("pivot_target", 0.0) or item.get("shelf_high", 0.0))
                    sl = float(item.get("shelf_low", 0.0) or (pivot * 0.92))
                    vsma = float(item.get("vol_sma20", 50000.0) or 50000.0)
                    if sym and pivot > 0:
                        targets[sym] = {
                            "universe": univ,
                            "setup": item.get("setup_class", "LAUNCHPAD COIL"),
                            "pivot": pivot,
                            "sl": sl,
                            "v_sma": vsma,
                        }
            except Exception as e:
                print(f"⚠️ Could not parse watchlist from {path}: {e}")
    print(f"🎯 Total Watchlist Targets Loaded: {len(targets)}")
    return targets

def load_previous_day_state(today_date_str):
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if d.get("Scan_Date") == today_date_str:
                return {item["symbol"]: item for item in d.get("Breakouts_All_Day", [])}
        except Exception:
            pass
    return {}

# ==============================================================================
# 5. ACTIVE TRADE TRAILING STOP EVALUATION
# ==============================================================================
def monitor_active_trades(quotes_map, scan_ts, mins):
    print("────────────────────────────────────────────────────────────")
    print(f"🔍 Checking active trade positions from: {MY_TRADES_FILE}")

    if not os.path.exists(MY_TRADES_FILE):
        print(f"❌ Active trade file '{MY_TRADES_FILE}' not found.")
        return

    try:
        with open(MY_TRADES_FILE, "r", encoding="utf-8") as f:
            trades = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return

    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    print(f"📋 Found {len(open_trades)} active trade(s) with status 'OPEN'.")
    if not open_trades:
        return

    # Closing Window: After 3:00 PM IST (345 mins elapsed out of 375 total mins)
    is_closing_window = (mins >= 345.0)
    warnings = []
    hard_stops = []

    for t in open_trades:
        sym = t.get("symbol", "").upper().strip()
        sl = float(t.get("hard_sl", 0.0))
        entry = float(t.get("entry_price", 0.0))
        if not sym or sl <= 0:
            continue

        q = quotes_map.get(sym)
        if not q or q["ltp"] <= 0:
            print(f"⚠️ Quote unavailable for active trade: {sym}")
            continue

        ltp = q["ltp"]
        pnl_pct = round(((ltp - entry) / entry) * 100, 2) if entry > 0 else 0.0
        dist_sl_pct = round(((ltp - sl) / sl) * 100, 2)
        print(f"🔎 [{sym}] LTP: ₹{ltp:.2f} | Stop Loss: ₹{sl:.2f} | PnL: {pnl_pct:+.2f}% | Breach Margin: {dist_sl_pct:+.2f}%")

        if ltp <= sl:
            row = {"symbol": sym, "ltp": ltp, "sl": sl, "pnl_%": pnl_pct, "dist_sl_%": dist_sl_pct}
            if is_closing_window:
                hard_stops.append(row)
            else:
                warnings.append(row)

    # 1. Closing breakdown alert (action required)
    if hard_stops:
        lines = [
            "🛑 <b>CONFIRMED CLOSING STOP BREACH</b>",
            f"🕒 <i>Time: {scan_ts} (Closing Window)</i>",
            "────────────────────────",
        ]
        for h in hard_stops:
            lines.append(
                f"• <b>{h['symbol']}</b>: LTP ₹{h['ltp']:.2f} (SL: ₹{h['sl']:.2f})\n"
                f"  PnL: <b>{h['pnl_%']:+.2f}%</b> (Breach: {h['dist_sl_%']:.2f}%)\n"
                f"  👉 <i>Closing below stop. Action recommended per rules.</i>\n"
            )
        send_telegram_msg("\n".join(lines))

    # 2. Midday wick test (advisory warning)
    elif warnings:
        lines = [
            "⚠️ <b>INTRADAY SUPPORT / SL WICK TEST</b>",
            f"🕒 <i>Time: {scan_ts}</i>",
            "────────────────────────",
        ]
        for w in warnings:
            lines.append(
                f"• <b>{w['symbol']}</b>: LTP ₹{w['ltp']:.2f} tested SL ₹{w['sl']:.2f}\n"
                f"  PnL: {w['pnl_%']:+.2f}% (Breach: {w['dist_sl_%']:.2f}%)\n"
                f"  ℹ️ <i>Candle unconfirmed. Review support bounce vs breakdown.</i>\n"
            )
        send_telegram_msg("\n".join(lines))
    else:
        print("🛡️ All active trades are holding safely above their stop losses.")
    print("────────────────────────────────────────────────────────────")

# ==============================================================================
# 6. MAIN ENGINE EXECUTION
# ==============================================================================
def run():
    now_ist = datetime.datetime.now(IST)
    today_date = now_ist.strftime("%Y-%m-%d")
    m_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    m_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if now_ist < m_open:
        mins = 1.0
    elif now_ist > m_close:
        mins = 375.0
    else:
        mins = max(1.0, (now_ist - m_open).total_seconds() / 60.0)

    pace_factor = 375.0 / mins
    scan_ts = now_ist.strftime("%Y-%m-%d %I:%M:%S %p IST")
    print(f"\n🚀 Running Intraday Live Scanner @ {scan_ts} (Elapsed: {int(mins)}m/375m)")

    targets = load_verified_watchlists()
    existing_breakouts = load_previous_day_state(today_date)

    # Collect all symbols (Watchlists + My Trades)
    all_symbols = list(targets.keys())
    if os.path.exists(MY_TRADES_FILE):
        try:
            with open(MY_TRADES_FILE, "r", encoding="utf-8") as f:
                tr = json.load(f)
            for t in tr:
                if t.get("status") == "OPEN":
                    all_symbols.append(t["symbol"].upper().strip())
        except Exception:
            pass

    unique_symbols = list(set(all_symbols))
    quotes_map = fetch_live_quotes_batch(unique_symbols)

    all_breakouts_map = dict(existing_breakouts)
    new_triggers = []
    coiling = []

    for sym, info in targets.items():
        q = quotes_map.get(sym)
        if not q or q["ltp"] <= 0:
            continue

        ltp, high, vol = q["ltp"], q["high"], q["vol"]
        pivot, sl, vsma = info["pivot"], info["sl"], info["v_sma"]

        surge = round((vol * pace_factor) / vsma, 2) if vsma > 0 else 1.0
        diff = round(((ltp - pivot) / pivot) * 100.0, 2)
        has_broken_out = (ltp >= pivot) or (high >= pivot) or (sym in existing_breakouts)

        row = {
            "symbol": sym,
            "universe": info["universe"],
            "setup": info["setup"],
            "pivot_ceiling": pivot,
            "ltp": ltp,
            "day_high": max(high, existing_breakouts.get(sym, {}).get("day_high", high)),
            "stop_loss": sl,
            "risk_%": round(((pivot - sl) / pivot) * 100, 2) if pivot > 0 else 0.0,
            "dist_pivot_%": diff,
            "proj_vol_pace": surge,
            "action": "BREAKOUT" if has_broken_out else "COILING",
            "first_trigger_time": existing_breakouts.get(sym, {}).get("first_trigger_time", now_ist.strftime("%I:%M %p")),
        }

        if has_broken_out:
            if sym not in existing_breakouts:
                new_triggers.append(row)
            all_breakouts_map[sym] = row
        else:
            coiling.append(row)

    breakout_list = sorted(list(all_breakouts_map.values()), key=lambda x: x["dist_pivot_%"], reverse=True)
    coiling_list = sorted(coiling, key=lambda x: x["dist_pivot_%"], reverse=True)

    # Persist JSON State
    output_payload = {
        "Scan_Date": today_date,
        "Scan_Timestamp": scan_ts,
        "Elapsed_Minutes": int(mins),
        "Total_Breakouts_Today": len(breakout_list),
        "Breakouts_All_Day": breakout_list,
        "Coiling": coiling_list[:25],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)

    # 1. Send Fresh Breakout Triggers
    if new_triggers:
        lines = [
            "🚨 <b>FRESH INTRADAY BREAKOUT TRIGGERED!</b>",
            f"🕒 <i>Time: {scan_ts} (Elapsed: {int(mins)}m/375m)</i>",
            "────────────────────────",
        ]
        for b in new_triggers:
            lines.append(
                f"• <b>{b['symbol']}</b> ({b['universe']} | {b['setup']})\n"
                f"  LTP: <b>₹{b['ltp']:.2f}</b> (Pivot: ₹{b['pivot_ceiling']:.2f}, {b['dist_pivot_%']:+.2f}%)\n"
                f"  SL: ₹{b['stop_loss']:.2f} (Risk: {b['risk_%']:.1f}%)\n"
                f"  Vol Pace: <b>{b['proj_vol_pace']:.2f}x SMA20</b>\n"
            )
        send_telegram_msg("\n".join(lines))

    # 2. Send Cumulative Progress Monitor (only if breakouts exist)
    if breakout_list:
        summary_lines = [
            "📊 <b>ALL-DAY BREAKOUT PROGRESS MONITOR</b>",
            f"🕒 <i>Time: {scan_ts} | Active Breakouts: {len(breakout_list)}</i>",
            "────────────────────────",
        ]
        for b in breakout_list:
            status_icon = "🟢" if b["dist_pivot_%"] >= 0 else "🔴"
            summary_lines.append(
                f"{status_icon} <b>{b['symbol']}</b>: ₹{b['ltp']:.2f} ({b['dist_pivot_%']:+.2f}% vs Pivot) "
                f"| High: ₹{b['day_high']:.2f} | Pace: {b['proj_vol_pace']:.1f}x"
            )
        send_telegram_msg("\n".join(summary_lines))

    # 3. Check Active Trailing Stop-Losses
    monitor_active_trades(quotes_map, scan_ts, mins)

    print(f"✅ [{scan_ts}] Breakouts: {len(breakout_list)} ({len(new_triggers)} new) | Coils: {len(coiling_list)}")

if __name__ == "__main__":
    run()
