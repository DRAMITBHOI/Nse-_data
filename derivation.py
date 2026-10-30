import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
START_DATE = "2021-01-01"
END_DATE = "2026-10-28"

OUTPUT_JSON = os.path.join(DATA_DIR, "nifty100_multivariate_summary.json")

NIFTY100_SYMBOLS = {
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", 
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "ATGL", "AUBANK", "AUROPHARMA", 
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BANKBARODA", 
    "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BOSCHLTD", "BPCL", 
    "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL", "CUMMINSIND", 
    "DABUR", "DIVISLAB", "DLF", "DMART", "DRREDDY", "EICHERMOT", "GAIL", "GODREJCP", 
    "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", 
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "HINDZINC", "ICICIBANK", "ICICIGI", 
    "ICICIPRULI", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", 
    "INFY", "INTERGLOBE", "IOC", "IRCTC", "IRFC", "ITC", "JINDALSTEL", "JIOFIN", 
    "JSWENERGY", "JSWSTEEL", "KOTAKBANK", "LICI", "LODHA", "LT", "LTIM", "LUPIN", 
    "M&M", "MARICO", "MARUTI", "MAXHEALTH", "MOTHERSON", "NAUKRI", "NESTLEIND", 
    "NTPC", "ONGC", "PFC", "PIDILITIND", "PNB", "POWERGRID", "RECLTD", "RELIANCE", 
    "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SRF", 
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", 
    "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNITDSPR", 
    "VBL", "VEDL", "VOLTAS", "WIPRO", "ZOMATO", "ZYDUSLIFE"
}

def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list): return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict): continue
        t = str(r.get("time", ""))[:10]
        c = float(r.get("close", 0) or 0)
        if not t or c <= 0: continue
        v = float(r.get("volume", 0) or 0)
        dv = float(r.get("delivery_vol", 0) or 0)
        dp = float(r.get("deliv_pct", 0) or 0)
        if dp > 0 and dv == 0: dv = v * (dp / 100.0)
        if v > 0 and dv > v: dv = v
        date_map[t] = {
            "time": t, "open": float(r.get("open", c) or c),
            "high": float(r.get("high", c) or c), "low": float(r.get("low", c) or c),
            "close": c, "volume": v, "delivery_vol": dv,
            "deliv_pct": (dv / v * 100.0) if v > 0 else 0.0
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
                    if abs(ratio - k) / k < 0.15: adj = k; break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
            if adj:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj
                    clean[j]["volume"] = clean[j]["volume"] * adj
    return clean

def simulate_trade(df, entry_idx, entry_price, hard_sl, dist_rule="3_in_10", apply_day25_rule=True):
    n = len(df)
    peak_price = max(entry_price, df["high"].iloc[entry_idx])
    peak_gain_pct = ((peak_price - entry_price) / entry_price) * 100.0
    vdu_under_count = 0
    consecutive_under_20 = 0

    for j in range(entry_idx + 1, n):
        bars_held = j - entry_idx
        c = df["close"].iloc[j]
        prev_c = df["close"].iloc[j - 1]
        h = df["high"].iloc[j]
        v = df["volume"].iloc[j]
        ema20 = df["ema_20"].iloc[j]
        atr14 = df["atr_14"].iloc[j]
        sma20_v = df["gross_sma20"].iloc[j]
        rc = df["range_closeness"].iloc[j]

        peak_price = max(peak_price, h)
        peak_gain_pct = max(peak_gain_pct, ((peak_price - entry_price) / entry_price) * 100.0)
        vol_pace = v / max(1.0, sma20_v)

        # 1. Catastrophe Stop
        if c < hard_sl:
            return ((c - entry_price) / entry_price) * 100.0, bars_held, "Hard Stop"

        # 2. 25-Day Momentum Stagnation Rule
        if apply_day25_rule and bars_held == 25:
            current_gain = ((c - entry_price) / entry_price) * 100.0
            if peak_gain_pct < 8.0 and current_gain < 8.0:
                return current_gain, bars_held, "Day 25 Stagnation Exit"

        # 3. Climactic Churn Harvest
        if peak_gain_pct >= 20.0 and vol_pace >= 2.0 and rc <= 0.35:
            return ((c - entry_price) / entry_price) * 100.0, bars_held, "Climactic Churn"

        # 4. Distribution Cluster Logic
        trigger_dist = False
        if dist_rule == "3_in_10" and df["dist_10d"].iloc[j] >= 3 and c < ema20:
            trigger_dist = True
        elif dist_rule == "4_in_12" and df["dist_12d"].iloc[j] >= 4 and c < ema20:
            trigger_dist = True
        elif dist_rule == "5_in_15" and df["dist_15d"].iloc[j] >= 5 and c < ema20:
            trigger_dist = True

        if trigger_dist:
            return ((c - entry_price) / entry_price) * 100.0, bars_held, "Distribution Cluster"

        # 5. Mega-Runner Cushion Lock
        if peak_gain_pct >= 50.0:
            cushion = max(ema20, peak_price * 0.85)
            if c < cushion:
                return ((c - entry_price) / entry_price) * 100.0, bars_held, "Mega Runner Lock"

        # 6. 20 EMA Trailing
        if c < ema20:
            consecutive_under_20 += 1
            is_heavy = (vol_pace >= 1.40) and (c < prev_c)
            if is_heavy or consecutive_under_20 >= 2:
                return ((c - entry_price) / entry_price) * 100.0, bars_held, "20 EMA Breach"
            else:
                vdu_under_count += 1
                dynamic_stop = min(ema20 * 0.95, ema20 - (1.5 * atr14))
                if peak_gain_pct >= 6.0:
                    dynamic_stop = max(entry_price * 1.01, dynamic_stop)
                if c < dynamic_stop or vdu_under_count > 5:
                    return ((c - entry_price) / entry_price) * 100.0, bars_held, "EMA Dynamic Floor"
        else:
            consecutive_under_20 = 0
            vdu_under_count = 0

    return ((df["close"].iloc[-1] - entry_price) / entry_price) * 100.0, n - 1 - entry_idx, "End of Data"

def run_derivation():
    print(f"🔬 Deriving Nifty 100 Breakout Mechanics ({START_DATE} to {END_DATE})...")
    all_json = glob.glob("**/*.json", recursive=True)
    stock_files = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if sym in NIFTY100_SYMBOLS:
            if sym not in stock_files or os.path.getsize(p) > os.path.getsize(stock_files[sym]):
                stock_files[sym] = p

    print(f"📦 Loaded {len(stock_files)} Verified Stock Files.")
    breakouts = []

    for sym, fpath in sorted(stock_files.items()):
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue
        if len(clean) < 80: continue

        df = pd.DataFrame(clean)
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(200, min_periods=40).mean()
        df["gross_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
        df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)

        prev_c = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()
        df["range_closeness"] = np.where(df["high"] > df["low"], (df["close"] - df["low"]) / (df["high"] - df["low"]), 0.5)

        dp_avg = df["deliv_pct"].rolling(50, min_periods=5).mean().fillna(35.0)
        df["is_dist"] = (
            (df["close"] < df["close"].shift(1)) & 
            ((df["delivery_vol"] >= 1.30 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * dp_avg)) & 
            (df["range_closeness"] <= 0.45)
        ).astype(int)
        df["dist_10d"] = df["is_dist"].rolling(10, min_periods=1).sum()
        df["dist_12d"] = df["is_dist"].rolling(12, min_periods=1).sum()
        df["dist_15d"] = df["is_dist"].rolling(15, min_periods=1).sum()

        n = len(df)
        for i in range(50, n - 30):
            t_curr = df["time"].iloc[i]
            if t_curr < START_DATE or t_curr > END_DATE: continue
            if df["turnover_50d"].iloc[i] < 10.0: continue
            if pd.notnull(df["sma_200"].iloc[i]) and df["close"].iloc[i] < df["sma_200"].iloc[i]: continue

            h0, c0, l0, v0, dv0 = df["high"].iloc[i], df["close"].iloc[i], df["low"].iloc[i], df["volume"].iloc[i], df["delivery_vol"].iloc[i]

            for s_len in [10, 15, 20]:
                lp_h = df["high"].iloc[i - s_len:i].max()
                lp_l = df["low"].iloc[i - s_len:i].min()
                if lp_l <= 0: continue
                spread = ((lp_h - lp_l) / lp_l) * 100.0

                if h0 >= lp_h * 1.0015 and c0 >= lp_h:
                    v_min = df["volume"].iloc[i - s_len:i].min()
                    vdu_depth = v_min / max(1.0, df["gross_sma20"].iloc[i - 1])

                    up_dv = df.loc[i - s_len:i - 1].apply(lambda r: r["delivery_vol"] if r["close"] >= r["open"] else 0, axis=1).sum()
                    dn_dv = df.loc[i - s_len:i - 1].apply(lambda r: r["delivery_vol"] if r["close"] < r["open"] else 0, axis=1).sum()
                    acc_ratio = (up_dv / dn_dv) if dn_dv > 0 else 2.50

                    dv_surge = dv0 / max(1.0, df["deliv_sma20"].iloc[i - 1])
                    rc = df["range_closeness"].iloc[i]
                    clearance_pct = ((h0 - lp_h) / lp_h) * 100.0

                    fwd_25_h = df["high"].iloc[i + 1:i + 26].max()
                    fwd_25_l = df["low"].iloc[i + 1:i + 26].min()
                    mfe = ((fwd_25_h - c0) / c0) * 100.0
                    mae = ((fwd_25_l - c0) / c0) * 100.0
                    hard_sl = min(l0, lp_l)

                    ret_std, h_std, r_std = simulate_trade(df, i, c0, hard_sl, dist_rule="3_in_10", apply_day25_rule=False)
                    ret_d25, h_d25, r_d25 = simulate_trade(df, i, c0, hard_sl, dist_rule="3_in_10", apply_day25_rule=True)
                    ret_4_12, _, r_4_12 = simulate_trade(df, i, c0, hard_sl, dist_rule="4_in_12", apply_day25_rule=True)
                    ret_5_15, _, r_5_15 = simulate_trade(df, i, c0, hard_sl, dist_rule="5_in_15", apply_day25_rule=True)

                    breakouts.append({
                        "symbol": sym, "date": t_curr, "base_len": s_len, "spread": spread,
                        "vdu_depth": vdu_depth, "acc_ratio": acc_ratio, "dv_surge": dv_surge,
                        "range_closeness": rc, "clearance_pct": clearance_pct,
                        "mfe_25": mfe, "mae_25": mae, "success_hurdle": 1 if (mfe >= 8.0 and mae >= -6.0) else 0,
                        "ret_std": ret_std, "ret_d25": ret_d25, "reason_d25": r_d25,
                        "ret_4_12": ret_4_12, "reason_4_12": r_4_12,
                        "ret_5_15": ret_5_15, "reason_5_15": r_5_15
                    })
                    break

    bdf = pd.DataFrame(breakouts)
    print(f"✅ Audited {len(bdf)} Total Breakouts.")

    def calc_table(df_subset, col, bins, labels):
        df_subset = df_subset.copy()
        df_subset["bin"] = pd.cut(df_subset[col], bins=bins, labels=labels, include_lowest=True)
        res = []
        for b in labels:
            sub = df_subset[df_subset["bin"] == b]
            if len(sub) == 0: continue
            wr_hurdle = (sub["success_hurdle"].mean()) * 100.0
            wr_std = (sub["ret_std"] > 0).mean() * 100.0
            wr_d25 = (sub["ret_d25"] > 0).mean() * 100.0
            pf_d25 = sub[sub["ret_d25"] > 0]["ret_d25"].sum() / abs(sub[sub["ret_d25"] <= 0]["ret_d25"].sum() or 1.0)
            res.append({
                "Range": b, "Trades": len(sub),
                "Hurdle_WR_%": round(wr_hurdle, 1),
                "Std_Exit2.2_WR_%": round(wr_std, 1),
                "D25_Exit2.2_WR_%": round(wr_d25, 1),
                "D25_PF": round(pf_d25, 2),
                "Avg_Return_%": round(sub["ret_d25"].mean(), 2)
            })
        return res

    summary = {
        "Total_Breakouts": len(bdf),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Shelf_Spread": calc_table(bdf, "spread", [0, 5, 8.5, 11, 25], ["<5% (Ultra-Tight)", "5.0-8.5% (Optimal)", "8.5-11.0% (Wide)", ">11% (Loose)"]),
        "Clearance_Pct": calc_table(bdf, "clearance_pct", [0, 0.5, 1.0, 1.5, 15.0], ["<0.5% (Poke)", "0.5-1.0% (Light)", "1.0-1.5% (Firm)", ">=1.5% (Impulse)"]),
        "Range_Closeness": calc_table(bdf, "range_closeness", [0, 0.50, 0.65, 0.80, 1.0], ["<0.50 (Wick Trap)", "0.50-0.65 (Average)", "0.65-0.80 (Strong)", ">=0.80 (Marubozu)"]),
        "VDU_Depth": calc_table(bdf, "vdu_depth", [0, 0.40, 0.65, 0.85, 5.0], ["<0.40x (Extreme VDU)", "0.40-0.65x (Healthy)", "0.65-0.85x (Mild)", ">0.85x (No VDU)"]),
        "Accumulation_Ratio": calc_table(bdf, "acc_ratio", [0, 1.0, 1.3, 1.5, 1.8, 99.0], ["<1.0 (Dist)", "1.0-1.3 (Neutral)", "1.3-1.5 (Mild Acc)", "1.5-1.8 (Strong Acc)", ">1.8 (Institutional)"]),
        "Distribution_Rule_Comparison": [
            {
                "Rule": "Baseline (3 in 10 Days)",
                "Win_Rate_%": round((bdf["ret_d25"] > 0).mean() * 100.0, 1),
                "Profit_Factor": round(bdf[bdf["ret_d25"] > 0]["ret_d25"].sum() / abs(bdf[bdf["ret_d25"] <= 0]["ret_d25"].sum() or 1.0), 2),
                "Avg_Return_%": round(bdf["ret_d25"].mean(), 2),
                "Dist_Exits": int((bdf["reason_d25"] == "Distribution Cluster").sum())
            },
            {
                "Rule": "Balanced (4 in 12 Days)",
                "Win_Rate_%": round((bdf["ret_4_12"] > 0).mean() * 100.0, 1),
                "Profit_Factor": round(bdf[bdf["ret_4_12"] > 0]["ret_4_12"].sum() / abs(bdf[bdf["ret_4_12"] <= 0]["ret_4_12"].sum() or 1.0), 2),
                "Avg_Return_%": round(bdf["ret_4_12"].mean(), 2),
                "Dist_Exits": int((bdf["reason_4_12"] == "Distribution Cluster").sum())
            },
            {
                "Rule": "Institutional Leeway (5 in 15 Days)",
                "Win_Rate_%": round((bdf["ret_5_15"] > 0).mean() * 100.0, 1),
                "Profit_Factor": round(bdf[bdf["ret_5_15"] > 0]["ret_5_15"].sum() / abs(bdf[bdf["ret_5_15"] <= 0]["ret_5_15"].sum() or 1.0), 2),
                "Avg_Return_%": round(bdf["ret_5_15"].mean(), 2),
                "Dist_Exits": int((bdf["reason_5_15"] == "Distribution Cluster").sum())
            }
        ]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"🎉 Successfully written to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_derivation()
