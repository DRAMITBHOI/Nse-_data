import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
MAX_PE = 35.0
OUTPUT_FILE = os.path.join(DATA_DIR, "scanA_results.json")

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
            print(f"⚠️ Warning fetching {u}: {e}")

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
            d_str = pd.to_datetime(raw_t).strftime("%Y-%m-%d")
            c = float(r.get("close", 0))
            if c <= 0:
                continue
            
            v = float(r.get("volume", 0) or 0)
            dv = float(r.get("delivery_vol", 0) or 0)
            pct = float(r.get("deliv_pct", 0) or 0)

            entry = {
                "time": d_str,
                "open": float(r.get("open", c)),
                "high": float(r.get("high", c)),
                "low": float(r.get("low", c)),
                "close": c,
                "delivery_vol": dv,
                "volume": v,
                "deliv_pct": pct
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
                r["low"] == prev["low"] and r["close"] == prev["close"] and r["volume"] == 0):
                continue
        clean.append(r)

    # Corporate actions adjustment
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

    # Forward-fill missing delivery gaps
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

def run_scan():
    print("🚀 Running Daily scanA Engine (Synchronized Framework)...")
    if not os.path.exists(DATA_DIR):
        print(f"❌ '{DATA_DIR}' directory does not exist.")
        return

    nifty_750_set = get_nifty_750_universe()
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
            "wyckoff_screener_results.json", "nifty750.json",
            "scanA_results.json"
        ]
    ]

    print(f"📊 Scanning {len(stock_files)} stocks...")
    candidates = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        pe_float = None
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
        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        vols = df["delivery_vol"].values
        t_vols = df["volume"].values
        pcts = df["deliv_pct"].values
        N = len(closes)

        turnovers = (closes * t_vols) / 1e7
        turnover_50d = float(np.mean(turnovers[max(0, N - 50):]))
        mean_deliv_50d = float(np.mean(pcts[max(0, N - 50):]))
        if mean_deliv_50d <= 0:
            continue

        deliv_sma20 = float(np.mean(vols[max(0, N - 20):]))
        gross_vol_sma20 = float(np.mean(t_vols[max(0, N - 20):]))

        is_n750 = sym in nifty_750_set

        # Tier Parameters (Bucket A widened to 45d for multi-month bases)
        if is_n750:
            if turnover_50d >= 30.0:
                tier_name = "Bucket A (>30 Cr)"
                pct_mult, vol_mult, min_cluster, base_window = 1.4, 1.0, 3, 45
                trail_mode = "Open / Trend Riding (Base SL Only)"
            elif turnover_50d >= 5.0:
                tier_name = "Bucket B (5-30 Cr)"
                pct_mult, vol_mult, min_cluster, base_window = 1.2, 1.0, 2, 15
                trail_mode = "Trail 20D Low (at +20% gain)"
            else:
                tier_name = "Bucket C (<5 Cr)"
                pct_mult, vol_mult, min_cluster, base_window = 1.4, 1.0, 4, 20
                trail_mode = "Trail 10D Low (at +15% gain) + Climax Exit"
        else:
            tier_name = "Non-Universe (Bucket C Rules)"
            pct_mult, vol_mult, min_cluster, base_window = 1.4, 1.0, 4, 20
            trail_mode = "Trail 10D Low (at +15% gain) + Climax Exit"

        # Continuous True Delivery OBV
        cur_obv = 0
        obvs = np.zeros(N)
        for idx in range(N):
            dv = vols[idx]
            if idx > 0:
                if closes[idx] > closes[idx - 1]:
                    cur_obv += dv
                elif closes[idx] < closes[idx - 1]:
                    cur_obv -= dv
            else:
                cur_obv = dv
            obvs[idx] = cur_obv

        # Accumulation Base Evaluation
        base_start = max(0, N - 1 - base_window)
        base_slice_pcts = pcts[base_start : N - 1]
        base_slice_vols = vols[base_start : N - 1]

        qualifying_days = (base_slice_pcts >= (pct_mult * mean_deliv_50d)) & (base_slice_vols >= (vol_mult * deliv_sma20))
        cluster_count = int(np.sum(qualifying_days))

        if cluster_count < min_cluster:
            continue

        base_highs = highs[base_start : N - 1]
        base_lows = lows[base_start : N - 1]
        if len(base_highs) == 0:
            continue

        swing_high_rel_idx = int(np.argmax(base_highs))
        swing_high_abs_idx = base_start + swing_high_rel_idx
        swing_high_price = float(base_highs[swing_high_rel_idx])
        base_low_price = float(np.min(base_lows))
        
        obv_at_swing_high = obvs[swing_high_abs_idx]
        current_close = float(closes[-1])
        current_open = float(opens[-1])
        current_obv = obvs[-1]
        initial_sl = round(base_low_price * 0.995, 2)
        risk_pct = round(((current_close - initial_sl) / current_close) * 100, 2) if current_close > 0 else 0

        # Breakout Condition
        is_breakout = bool((current_close >= swing_high_price) and (current_obv > obv_at_swing_high))

        # Model 2B Climax Churn Check
        is_climax_distribution = False
        if "Bucket C" in tier_name and gross_vol_sma20 > 0:
            is_climax_distribution = bool(
                t_vols[-1] >= (1.5 * gross_vol_sma20) and 
                pcts[-1] <= (0.70 * mean_deliv_50d) and 
                current_close <= (current_open * 1.002)
            )

        trail_10d_low = round(float(np.min(lows[max(0, N - 10):])), 2)
        trail_20d_low = round(float(np.min(lows[max(0, N - 20):])), 2)

        if risk_pct > 15.0:
            continue

        signal_type = "🟢 BUY BREAKOUT" if is_breakout else "🟡 ACCUMULATION BASE"
        if is_climax_distribution:
            signal_type = "🔴 CLIMAX DUMP ALERT"

        candidates.append({
            "Symbol": sym,
            "Signal": signal_type,
            "Tier": tier_name,
            "LTP (₹)": round(current_close, 2),
            "P/E": f"{pe_float:.1f}" if pe_float is not None else "N/A",
            "Swing High (₹)": round(swing_high_price, 2),
            "Initial Base SL (₹)": initial_sl,
            "Trail 10D Low (₹)": trail_10d_low,
            "Trail 20D Low (₹)": trail_20d_low,
            "Exit Management": trail_mode,
            "Risk %": f"{risk_pct}%",
            "Base Span": f"{base_window}d Base ({cluster_count} Dots)",
            "Turnover (₹ Cr)": round(turnover_50d, 1)
        })

    candidates.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk %"].replace("%", ""))), reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"🎉 scanA Complete! Saved {len(candidates)} candidate(s) to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_scan()
