import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_scanner_results.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0


# -------------------------------------------------------------
# 1. DATA PREPARATION & SPLIT MULTIPLIER CORRECTION
# -------------------------------------------------------------
def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []

    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()
        if not raw_t:
            continue
        d_str = raw_t[:10]
        try:
            c = float(r.get("close", 0) or 0)
            if c <= 0:
                continue
            entry = {
                "time": d_str,
                "open": float(r.get("open", c) or c),
                "high": float(r.get("high", c) or c),
                "low": float(r.get("low", c) or c),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0),
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj_factor = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj_factor = k
                        break
                if not adj_factor and 1.70 <= ratio <= 2.30:
                    adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50:
                    adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50:
                    adj_factor = 10.0
            if adj_factor:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj_factor, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj_factor, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj_factor, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj_factor, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj_factor
                    clean[j]["volume"] = clean[j]["volume"] * adj_factor

    running_vol = 50000.0
    for i in range(len(clean)):
        v = clean[i]["volume"]
        dv = clean[i]["delivery_vol"]
        pct = clean[i]["deliv_pct"]
        if v > 0:
            running_vol = 0.9 * running_vol + 0.1 * v
        else:
            clean[i]["volume"] = running_vol
            v = running_vol
        if dv <= 0:
            clean[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            clean[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            clean[i]["delivery_vol"] = v
            clean[i]["deliv_pct"] = 100.0

    return clean


def load_nifty_benchmark():
    nifty_candidates = [
        os.path.join(DATA_DIR, "nifty.json"),
        os.path.join(DATA_DIR, "nifty50.json"),
        os.path.join(DATA_DIR, "NIFTY.json"),
        os.path.join(DATA_DIR, "NIFTY50.json"),
    ]
    raw = None
    for path in nifty_candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list) and len(data) > 50:
                        raw = data
                        break
            except Exception:
                continue

    if not raw:
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=5y"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                yf_data = json.loads(resp.read().decode("utf-8"))
                res = yf_data["chart"]["result"][0]
                timestamps = res["timestamp"]
                quotes = res["indicators"]["quote"][0]
                raw = []
                for ts, o, h, l, c, v in zip(timestamps, quotes["open"], quotes["high"], quotes["low"], quotes["close"], quotes["volume"]):
                    if c is not None and c > 0:
                        d_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        raw.append({"time": d_str, "open": o, "high": h, "low": l, "close": c, "volume": v or 100000})
        except Exception:
            pass

    clean = clean_and_prepare(raw)
    if not clean:
        return {}

    df = pd.DataFrame(clean)
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    nifty_map = {}
    for _, row in df.iterrows():
        nifty_map[row["time"]] = float(row["perf_60d"]) if pd.notnull(row["perf_60d"]) else 0.001
    return nifty_map


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            elif isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


# -------------------------------------------------------------
# 2. MULTI-PHASE SCANNER ENGINE
# -------------------------------------------------------------
def scan_stock(symbol, clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return None

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(0)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    direction = np.where(df["close"] >= df["close"].shift(1), 1.0, -1.0)
    df["demat_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["demat_obv_20max"] = df["demat_obv"].rolling(20, min_periods=5).max()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.0)

    i = len(df) - 1
    c = df["close"].iloc[i]
    h = df["high"].iloc[i]
    l = df["low"].iloc[i]
    v = df["volume"].iloc[i]
    v_avg = df["gross_vol_sma20"].iloc[i]
    dv = df["delivery_vol"].iloc[i]
    dv_avg = df["deliv_sma"].iloc[i]
    turnover = df["turnover_50d"].iloc[i]
    dp = df["deliv_pct"].iloc[i]
    dp_avg = df["deliv_pct_50d"].iloc[i]
    rc = df["range_closeness"].iloc[i]
    ema20 = df["ema20"].iloc[i]
    sma200 = df["sma200"].iloc[i]
    stock_p60 = df["perf_60d"].iloc[i]
    t = df["time"].iloc[i]

    if turnover < MIN_TURNOVER_CR:
        return None

    # Trend & Relative Strength Baseline
    is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
    n_perf = nifty_perf_map.get(t)
    if n_perf is None:
        prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
        n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
    is_rs = bool(stock_p60 > n_perf)

    if not (is_stage2 and is_rs):
        return None

    deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
    rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
    conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

    # -------------------------------------------------------------
    # 1. CHECK PHASE 1: ACTIVE BUY TRIGGER ON LATEST CLOSE
    # -------------------------------------------------------------
    # Setup A: V-Reversal
    recent_20_high = df["high"].iloc[i - 20:i].max()
    if (recent_20_high - l) / recent_20_high >= 0.18:
        recent_trough = df["low"].iloc[i - 5:i].min()
        prior_3d_high = df["high"].iloc[i - 4:i].max()
        c_reclaim = (c >= prior_3d_high) and (c > df["close"].iloc[i - 1])
        c_vol_rev = v >= (1.3 * v_avg)
        c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg if dp_avg > 0 else False)
        c_candle_rev = rc >= 0.55

        if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
            sl = round(recent_trough * 0.995, 2)
            eff_sl = max(sl, round(float(ema20), 2))
            risk_pct = round(((c - eff_sl) / c) * 100.0, 2)
            if risk_pct <= 12.0 and conviction_score >= 2.5:
                return {
                    "symbol": symbol,
                    "phase": "TRIGGERED",
                    "setup": "V-REVERSAL",
                    "close": round(c, 2),
                    "pivot_ceiling": round(prior_3d_high, 2),
                    "stop_loss": eff_sl,
                    "risk_pct": risk_pct,
                    "conviction_score": conviction_score,
                    "turnover_cr": round(turnover, 1),
                }

    # Setup B: Micro-Launchpad Breakout
    valid_launchpad = False
    launchpad_high = 0.0
    launchpad_low = 0.0
    for shelf_len in range(10, 19):
        s_high = df["high"].iloc[i - shelf_len:i].max()
        s_low = df["low"].iloc[i - shelf_len:i].min()
        if s_low > 0 and ((s_high - s_low) / s_low) <= 0.08:
            valid_launchpad = True
            launchpad_high = round(float(s_high), 2)
            launchpad_low = round(float(s_low), 2)
            break

    if valid_launchpad:
        base_up_deliv = sum(df["delivery_vol"].iloc[k] for k in range(i - 15, i) if df["close"].iloc[k] >= df["close"].iloc[k - 1])
        base_down_deliv = sum(df["delivery_vol"].iloc[k] for k in range(i - 15, i) if df["close"].iloc[k] < df["close"].iloc[k - 1])
        base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5

        c_bo = (c >= launchpad_high) and (h >= launchpad_high)
        c_vol_bo = v >= (1.4 * v_avg)
        c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
        c_candle_bo = rc >= 0.65
        obv_high = bool(df["demat_obv"].iloc[i] >= df["demat_obv_20max"].iloc[i - 1])

        if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.50) and obv_high:
            sl = round(min(l, launchpad_low), 2)
            eff_sl = max(sl, round(float(ema20), 2))
            risk_pct = round(((c - eff_sl) / c) * 100.0, 2)
            if risk_pct <= 12.0 and conviction_score >= 2.5:
                return {
                    "symbol": symbol,
                    "phase": "TRIGGERED",
                    "setup": "LAUNCHPAD-BO",
                    "close": round(c, 2),
                    "pivot_ceiling": launchpad_high,
                    "stop_loss": eff_sl,
                    "risk_pct": risk_pct,
                    "conviction_score": conviction_score,
                    "turnover_cr": round(turnover, 1),
                }

        # -------------------------------------------------------------
        # 2. CHECK PHASE 2: PRE-BREAKOUT COILING WATCHLIST (<= 2.5% of ceiling)
        # -------------------------------------------------------------
        dist_to_ceiling = round(((launchpad_high - c) / launchpad_high) * 100.0, 2)
        if 0.0 < dist_to_ceiling <= 2.50:
            eff_sl = max(launchpad_low, round(float(ema20), 2))
            risk_pct = round(((c - eff_sl) / c) * 100.0, 2)
            return {
                "symbol": symbol,
                "phase": "COILING",
                "setup": "PRE-BREAKOUT",
                "close": round(c, 2),
                "pivot_ceiling": launchpad_high,
                "dist_to_ceiling_pct": dist_to_ceiling,
                "stop_loss": eff_sl,
                "risk_pct": risk_pct,
                "conviction_score": conviction_score,
                "turnover_cr": round(turnover, 1),
            }

    # -------------------------------------------------------------
    # 3. CHECK PHASE 3: ACTIVE TRAILING POSITION MONITOR
    # -------------------------------------------------------------
    # If currently trading above 20 EMA in Stage 2 with sustained momentum
    if c >= ema20 and (c > df["close"].iloc[i - 10]):
        return {
            "symbol": symbol,
            "phase": "ACTIVE_HOLD",
            "setup": "TREND_RUNNER",
            "close": round(c, 2),
            "trailing_stop_ema20": round(float(ema20), 2),
            "cushion_pct": round(((c - ema20) / c) * 100.0, 2),
            "turnover_cr": round(turnover, 1),
        }

    return None


# -------------------------------------------------------------
# 3. SCANNER DISPATCHER
# -------------------------------------------------------------
def run_daily_scanner():
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "gap_margin_candidates.json", "swing3_results.json",
        "backtest_results.json", "swing3_compare.json", "swing3_run2_results.json",
        "swing3_run2_compare_results.json", "swing3_run3_results.json",
        "swing3_target400_results.json", "swing3_exit_study_results.json",
        "swing3_scanner_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"📡 Scanning {len(target_files)} symbols in Nifty 750 for Swing 3.0 Real-Time Phases...")

    triggered = []
    coiling = []
    active_holds = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym not in nifty750_set:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                res = scan_stock(sym, clean, nifty_perf_map)
                if res:
                    if res["phase"] == "TRIGGERED":
                        triggered.append(res)
                    elif res["phase"] == "COILING":
                        coiling.append(res)
                    elif res["phase"] == "ACTIVE_HOLD":
                        active_holds.append(res)
        except Exception:
            continue

    # Rank Phase 1 by Conviction Score (Max 1 primary recommendation)
    triggered.sort(key=lambda x: x["conviction_score"], reverse=True)
    coiling.sort(key=lambda x: (x["dist_to_ceiling_pct"], -x["conviction_score"]))
    active_holds.sort(key=lambda x: x["cushion_pct"], reverse=True)

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Triggered Today": triggered,
        "Coiling Watchlist": coiling[:15],
        "Active Swings (20 EMA Trail)": active_holds[:20],
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Scan Complete | Triggers: {len(triggered)} | Coiling Setups: {len(coiling)} | Active Swings: {len(active_holds)}")


if __name__ == "__main__":
    run_daily_scanner()
