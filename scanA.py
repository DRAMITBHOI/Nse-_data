import os
import io
import json
import time
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scanA_results.json")
MAX_PE = 35.0
MAX_RISK_PCT = 10.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def get_nifty_750_universe():
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fp:
                return set(json.load(fp))
        except Exception:
            pass

    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]
    symbols = set()
    for u in urls:
        try:
            req = urllib.request.Request(u, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Warning fetching universe: {e}")

    sorted_list = sorted(list(symbols))
    if sorted_list:
        with open(local_path, "w") as fp:
            json.dump(sorted_list, fp, indent=2)
    return set(sorted_list)

def clean_and_prepare_dataset(raw_data):
    if not raw_data:
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()
        if not raw_t:
            continue
        try:
            dt = pd.to_datetime(raw_t)
            if dt.dayofweek >= 5:
                continue
            d_str = dt.strftime("%Y-%m-%d")
            c = float(r.get("close", 0))
            if c <= 0:
                continue
            
            entry = {
                "time": d_str,
                "open": float(r.get("open", c)),
                "high": float(r.get("high", c)),
                "low": float(r.get("low", c)),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    sorted_records = [date_map[k] for k in sorted(date_map.keys())]

    clean = []
    for r in sorted_records:
        if clean:
            prev = clean[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    clean.pop()
        clean.append(r)

    # Corporate actions split adjustment
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

def get_nifty_regime():
    nifty_path = os.path.join(DATA_DIR, "NIFTY50.json")
    if not os.path.exists(nifty_path):
        nifty_path = os.path.join(DATA_DIR, "NIFTY.json")
    if not os.path.exists(nifty_path):
        nifty_path = os.path.join(DATA_DIR, "RELIANCE.json")
    
    if os.path.exists(nifty_path):
        try:
            with open(nifty_path, "r") as f:
                raw = json.load(f)
            df = pd.DataFrame(clean_and_prepare_dataset(raw))
            if not df.empty and "close" in df.columns and len(df) >= 50:
                df["sma50"] = df["close"].rolling(50, min_periods=20).mean()
                latest_c = df["close"].iloc[-1]
                latest_sma = df["sma50"].iloc[-1]
                return "FAVOURABLE (>= 50 SMA)" if latest_c >= latest_sma else "UNFAVOURABLE (< 50 SMA)"
        except Exception:
            pass
    return "FAVOURABLE (>= 50 SMA)"

def run_scan_a():
    print("🚀 Running Synchronized scanA Screener...")
    nifty_750_set = get_nifty_750_universe()
    nifty_regime_str = get_nifty_regime()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "backtest_report.json", "segmented_backtest_report.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "scanA_results.json", "nifty750.json", "NIFTY50.json", "NIFTY.json"
        ]
    ]

    results = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        if pe_val is not None:
            try:
                pe_float = float(pe_val)
                if pe_float <= 0 or pe_float >= MAX_PE:
                    continue
            except (ValueError, TypeError):
                pass

        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception:
            continue

        clean_history = clean_and_prepare_dataset(raw)
        if len(clean_history) < 60:
            continue

        df = pd.DataFrame(clean_history)
        df["deliv_sma"] = df["delivery_vol"].rolling(window=20, min_periods=1).mean()
        df["gross_vol_sma20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean()
        df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean()

        cur_obv = 0
        obvs = []
        for i, row in df.iterrows():
            dv = float(row["delivery_vol"])
            if i > 0:
                pc = float(df.at[i - 1, "close"])
                cc = float(row["close"])
                if cc > pc: cur_obv += dv
                elif cc < pc: cur_obv -= dv
            else:
                cur_obv = dv
            obvs.append(cur_obv)
        df["deliv_obv"] = obvs

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        pcts = df["deliv_pct"].values
        pct_50 = df["deliv_pct_50d"].values
        d_vols = df["delivery_vol"].values
        deliv_sma = df["deliv_sma"].values
        gross_vols = df["volume"].values
        gross_sma = df["gross_vol_sma20"].values
        to_50 = df["turnover_50d"].values
        N = len(closes)

        is_n750 = sym in nifty_750_set

        # True Historical Simulation Engine
        in_trade = False
        entry_idx = 0
        entry_price = 0.0
        active_sl = 0.0
        active_bucket = ""
        max_run_gain = 0.0
        partial_booked = False
        target_15 = 0.0
        cooldown_until = 0

        for i in range(30, N):
            curr_to = to_50[i] if not np.isnan(to_50[i]) else 0.0

            if is_n750:
                if curr_to >= 30.0:
                    tier = "Bucket A"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 3, 45
                elif curr_to >= 5.0:
                    tier = "Bucket B"
                    pct_m, vol_m, min_c, base_w = 1.2, 1.0, 2, 15
                else:
                    tier = "Bucket C"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 4, 20
            else:
                tier = "Bucket C"
                pct_m, vol_m, min_c, base_w = 1.4, 1.0, 4, 20

            # 1. Trade Lifecycle
            if in_trade:
                gain = ((highs[i] - entry_price) / entry_price) * 100
                if gain > max_run_gain:
                    max_run_gain = gain

                if max_run_gain >= 15.0 and not partial_booked:
                    partial_booked = True
                    active_sl = entry_price

                # Dynamic Trails
                if "Bucket C" in active_bucket:
                    if max_run_gain >= 15.0 and i >= entry_idx + 10:
                        trail_15 = float(np.min(lows[i - 15 : i]))
                        if trail_15 > active_sl:
                            active_sl = trail_15
                elif "Bucket B" in active_bucket:
                    if max_run_gain >= 20.0 and i >= entry_idx + 20:
                        trail_20 = float(np.min(lows[i - 20 : i]))
                        if trail_20 > active_sl:
                            active_sl = trail_20
                else:
                    if max_run_gain >= 25.0 and i >= entry_idx + 30:
                        trail_30 = float(np.min(lows[i - 30 : i]))
                        if trail_30 > active_sl:
                            active_sl = trail_30

                exit_triggered = False
                if lows[i] <= active_sl:
                    exit_triggered = True
                elif "Bucket C" in active_bucket and i > entry_idx + 2:
                    if gross_sma[i] > 0 and pct_50[i] > 0:
                        if gross_vols[i] >= (1.5 * gross_sma[i]) and pcts[i] <= (0.70 * pct_50[i]) and closes[i] <= df.at[i, "open"]:
                            exit_triggered = True
                elif "Bucket A" in active_bucket and (i - entry_idx) >= 30 and max_run_gain < 8.0:
                    exit_triggered = True

                if exit_triggered:
                    in_trade = False
                    cooldown_until = i + 5
                    continue

            # 2. Breakout Evaluation
            if not in_trade and i > cooldown_until and i >= base_w:
                base_start = i - base_w
                qualifying = (pcts[base_start:i] >= (pct_m * pct_50[base_start:i])) & (d_vols[base_start:i] >= (vol_m * deliv_sma[base_start:i]))
                if np.sum(qualifying) >= min_c:
                    base_highs = highs[base_start:i]
                    sw_idx = int(np.argmax(base_highs))
                    sw_high = base_highs[sw_idx]
                    sw_obv = obvs[base_start + sw_idx]

                    if closes[i] > sw_high and closes[i - 1] <= sw_high and obvs[i] > sw_obv:
                        entry_cand = closes[i]
                        pre_lookback = min(12, i - base_start)
                        recent_swing_low = float(np.min(lows[i - pre_lookback : i]))
                        sl_cand = round(recent_swing_low * 0.995, 2)
                        risk = ((entry_cand - sl_cand) / entry_cand) * 100

                        if risk <= MAX_RISK_PCT:
                            active_bucket = tier
                            in_trade = True
                            entry_idx = i
                            entry_price = entry_cand
                            active_sl = sl_cand
                            target_15 = round(entry_cand * 1.15, 2)
                            max_run_gain = 0.0
                            partial_booked = False

        # Current Status on Today's Bar (last_i)
        last_i = N - 1
        ltp = round(float(closes[last_i]), 2)
        base_start = last_i - base_w
        qualifying = (pcts[base_start:last_i] >= (pct_m * pct_50[base_start:last_i])) & (d_vols[base_start:last_i] >= (vol_m * deliv_sma[base_start:last_i]))
        current_dots = int(np.sum(qualifying))
        base_highs = highs[base_start:last_i]
        sw_high = round(float(np.max(base_highs)), 2)

        # Output Candidate ONLY if genuinely in an active trade or breaking out today
        if in_trade:
            if partial_booked:
                signal_type = "🎯 50% BOOKED (TRAIL REST)"
            else:
                signal_type = "🔵 ACTIVE HOLDING"
            
            risk_pct = round(((entry_price - active_sl) / entry_price) * 100, 1)

            results.append({
                "Symbol": sym,
                "Signal": signal_type,
                "Tier": active_bucket,
                "LTP (₹)": ltp,
                "Swing High (₹)": round(entry_price, 2),
                "Swing SL (₹)": round(active_sl, 2),
                "Target +15% (₹)": target_15,
                "Active Trail (₹)": round(active_sl, 2),
                "Risk %": f"{risk_pct}%",
                "Dot Cluster": f"{current_dots}/{min_c}",
                "Turnover (₹ Cr)": round(curr_to, 2)
            })
        elif current_dots >= min_c and ltp >= (sw_high * 0.985) and ltp <= sw_high:
            pre_lookback = min(12, last_i - base_start)
            recent_swing_low = float(np.min(lows[last_i - pre_lookback : last_i]))
            sl_cand = round(recent_swing_low * 0.995, 2)
            risk_pct = round(((ltp - sl_cand) / ltp) * 100, 1)

            results.append({
                "Symbol": sym,
                "Signal": "🟡 NEAR BREAKOUT",
                "Tier": tier,
                "LTP (₹)": ltp,
                "Swing High (₹)": sw_high,
                "Swing SL (₹)": sl_cand,
                "Target +15% (₹)": round(sw_high * 1.15, 2),
                "Active Trail (₹)": sl_cand,
                "Risk %": f"{risk_pct}%",
                "Dot Cluster": f"{current_dots}/{min_c}",
                "Turnover (₹ Cr)": round(curr_to, 2)
            })

    priority_map = {"🟢": 1, "🟡": 2, "🎯": 3, "🔵": 4}
    results.sort(key=lambda x: (priority_map.get(x["Signal"][:1], 5), -x["Turnover (₹ Cr)"]))

    final_payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Nifty Regime": nifty_regime_str,
        "Candidates Count": len(results),
        "Candidates": results
    }

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(final_payload, fp, indent=2)

    print(f"🎉 Synchronized scanA Complete! {len(results)} valid active setups saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_scan_a()
