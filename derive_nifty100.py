import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

# --------------------------------------------------------------------------
# CONFIGURATION & UNIVERSE (NIFTY 100)
# --------------------------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
START_DATE = "2021-01-01"
END_DATE = "2026-10-28"

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

def run_nifty100_derivation():
    print(f"🚀 Re-running Nifty 100 Empirical Derivation ({START_DATE} to {END_DATE})...")
    all_json = glob.glob("**/*.json", recursive=True)
    stock_files = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if any(k in bname.lower() for k in ["nifty", "audit", "result", "trade", "secret", "package", "checkpoint", "live"]):
            continue
        if sym in NIFTY100_SYMBOLS:
            if sym not in stock_files or os.path.getsize(p) > os.path.getsize(stock_files[sym]):
                stock_files[sym] = p

    print(f"📦 Filtered {len(stock_files)} Verified Nifty 100 Stock Files.")

    breakouts = []

    for sym, fpath in stock_files.items():
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue
        if len(clean) < 80: continue

        df = pd.DataFrame(clean)
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(200, min_periods=40).mean()
        df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
        df["vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
        
        prev_c = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()
        df["range_closeness"] = np.where(df["high"] > df["low"], (df["close"] - df["low"]) / (df["high"] - df["low"]), 0.5)

        # Distribution count
        dp_avg = df["deliv_pct"].rolling(50, min_periods=5).mean().fillna(35.0)
        df["is_dist"] = (
            (df["close"] < df["close"].shift(1)) & 
            ((df["delivery_vol"] >= 1.30 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * dp_avg)) & 
            (df["range_closeness"] <= 0.45)
        ).astype(int)
        df["dist_10d"] = df["is_dist"].rolling(10, min_periods=1).sum()

        n = len(df)
        for i in range(60, n - 25):
            t0 = df["time"].iloc[i]
            if t0 < START_DATE or t0 > END_DATE: continue
            c0 = df["close"].iloc[i]
            h0 = df["high"].iloc[i]
            l0 = df["low"].iloc[i]
            v0 = df["volume"].iloc[i]
            dv0 = df["delivery_vol"].iloc[i]

            # Find 10-22 bar base
            highs = df["high"].values
            lows = df["low"].values
            delivs = df["delivery_vol"].values
            closes = df["close"].values
            volumes = df["volume"].values

            best_spread, lp_h, lp_l, best_len = 999.0, 0, 0, 15
            shelf_found = False
            for s_len in range(10, 23):
                s_h = highs[max(0, i - s_len):i].max()
                s_l = lows[max(0, i - s_len):i].min()
                if s_l > 0:
                    spread = ((s_h - s_l) / s_l) * 100.0
                    if spread < best_spread:
                        best_spread = spread
                        lp_h, lp_l, best_len = s_h, s_l, s_len
                        shelf_found = True

            if not shelf_found: continue

            # Breakout definition: Price clears ceiling
            if c0 >= lp_h * 1.0015:
                # Accumulation ratio
                base_up = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] >= closes[k - 1])
                base_dn = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] < closes[k - 1])
                acc_ratio = round(base_up / base_dn, 2) if base_dn > 0 else 2.50

                # Pre-breakout VDU depth
                base_min_vol = min(volumes[max(0, i - best_len):i])
                sma_v_pre = df["vol_sma20"].iloc[i - 1]
                vdu_depth = round(base_min_vol / sma_v_pre, 2) if sma_v_pre > 0 else 1.0

                # Breakout volume surge & RC
                dv_prev = df["deliv_sma20"].iloc[i - 1]
                dv_surge = round(dv0 / dv_prev, 2) if dv_prev > 0 else 1.0
                rc = round(df["range_closeness"].iloc[i], 2)

                # 20-bar forward targets
                fwd_highs = df["high"].iloc[i + 1: i + 25].values
                fwd_lows = df["low"].iloc[i + 1: i + 25].values
                mfe = ((fwd_highs.max() - c0) / c0) * 100.0 if len(fwd_highs) > 0 else 0.0
                mae = ((fwd_lows.min() - c0) / c0) * 100.0 if len(fwd_lows) > 0 else 0.0

                # ---------------------------------------------------------
                # Trailing Engine 1: Exit 2.2
                # ---------------------------------------------------------
                entry = c0
                hard_sl = min(l0, lp_l)
                peak = entry
                pnl_exit2_2 = 0.0
                bars_exit2_2 = 0
                vdu_under = 0

                for j in range(i + 1, min(n, i + 80)):
                    cj = df["close"].iloc[j]
                    hj = df["high"].iloc[j]
                    vj = df["volume"].iloc[j]
                    emaj = df["ema_20"].iloc[j]
                    atrj = df["atr_14"].iloc[j]
                    v_smaj = df["vol_sma20"].iloc[j]
                    rcj = df["range_closeness"].iloc[j]
                    distj = df["dist_10d"].iloc[j]

                    peak = max(peak, hj)
                    peak_gain = ((peak - entry) / entry) * 100.0
                    vol_pace = vj / max(1.0, v_smaj)

                    if cj < hard_sl:
                        pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                    elif peak_gain >= 20.0 and vol_pace >= 2.0 and rcj <= 0.35:
                        pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                    elif distj >= 3 and cj < emaj:
                        pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                    elif peak_gain >= 50.0 and cj < max(emaj, peak * 0.85):
                        pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                    elif cj < emaj:
                        if vol_pace >= 1.25 and cj < df["close"].iloc[j - 1]:
                            pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                        else:
                            vdu_under += 1
                            atr_floor = emaj - (1.5 * atrj)
                            pct_floor = emaj * 0.95
                            active_stop = min(pct_floor, atr_floor)
                            if peak_gain >= 5.0:
                                active_stop = max(entry * 1.01, active_stop)
                            if cj < active_stop:
                                pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                            elif vdu_under > 5:
                                pnl_exit2_2 = ((cj - entry) / entry) * 100.0; bars_exit2_2 = j - i; break
                    else:
                        vdu_under = 0
                else:
                    pnl_exit2_2 = ((df["close"].iloc[min(n - 1, i + 80)] - entry) / entry) * 100.0
                    bars_exit2_2 = min(n - 1, i + 80) - i

                # ---------------------------------------------------------
                # Trailing Engine 2: Fast 10 EMA Cross
                # ---------------------------------------------------------
                pnl_fast10 = 0.0
                bars_fast10 = 0
                for j in range(i + 1, min(n, i + 80)):
                    if df["close"].iloc[j] < df["ema_10"].iloc[j]:
                        pnl_fast10 = ((df["close"].iloc[j] - entry) / entry) * 100.0
                        bars_fast10 = j - i
                        break
                else:
                    pnl_fast10 = ((df["close"].iloc[min(n - 1, i + 80)] - entry) / entry) * 100.0
                    bars_fast10 = min(n - 1, i + 80) - i

                # ---------------------------------------------------------
                # Trailing Engine 3: ATR Chandelier (2.5x ATR)
                # ---------------------------------------------------------
                pnl_atr = 0.0
                bars_atr = 0
                pk = entry
                for j in range(i + 1, min(n, i + 80)):
                    pk = max(pk, df["high"].iloc[j])
                    if df["close"].iloc[j] < (pk - 2.5 * df["atr_14"].iloc[j]):
                        pnl_atr = ((df["close"].iloc[j] - entry) / entry) * 100.0
                        bars_atr = j - i
                        break
                else:
                    pnl_atr = ((df["close"].iloc[min(n - 1, i + 80)] - entry) / entry) * 100.0
                    bars_atr = min(n - 1, i + 80) - i

                # ---------------------------------------------------------
                # Trailing Engine 4: Institutional Climax / Distribution
                # ---------------------------------------------------------
                pnl_climax = 0.0
                bars_climax = 0
                for j in range(i + 1, min(n, i + 80)):
                    if df["dist_10d"].iloc[j] >= 3:
                        pnl_climax = ((df["close"].iloc[j] - entry) / entry) * 100.0
                        bars_climax = j - i
                        break
                else:
                    pnl_climax = ((df["close"].iloc[min(n - 1, i + 80)] - entry) / entry) * 100.0
                    bars_climax = min(n - 1, i + 80) - i

                breakouts.append({
                    "symbol": sym, "time": t0, "acc_ratio": acc_ratio, "vdu_depth": vdu_depth,
                    "dv_surge": dv_surge, "rc": rc, "spread": best_spread,
                    "mfe": mfe, "mae": mae,
                    "pnl_exit2_2": pnl_exit2_2, "bars_exit2_2": bars_exit2_2,
                    "pnl_fast10": pnl_fast10, "bars_fast10": bars_fast10,
                    "pnl_atr": pnl_atr, "bars_atr": bars_atr,
                    "pnl_climax": pnl_climax, "bars_climax": bars_climax,
                    "raw_win": int(mfe >= 8.0 and mae >= -6.0)
                })

        df_res = pd.DataFrame(breakouts)
    print("\n" + "=" * 90)
    print(f"📊 NIFTY 100 EMPIRICAL FEATURE ANALYSIS ({len(df_res):,} TOTAL BREAKOUTS)")
    print("=" * 90)

    # 1. Exit Engine Comparison
    engines = [
        ("Exit 2.2 (Multi-Tiered Trail)", "pnl_exit2_2", "bars_exit2_2"),
        ("Fast 10 EMA Cross", "pnl_fast10", "bars_fast10"),
        ("ATR Chandelier (2.5x ATR)", "pnl_atr", "bars_atr"),
        ("Institutional Climax/Dist", "pnl_climax", "bars_climax")
    ]
    eng_rows = []
    total_breakouts = len(df_res)
    for name, p_col, b_col in engines:
        wins = df_res[df_res[p_col] > 0][p_col]
        losses = df_res[df_res[p_col] <= 0][p_col]
        wr = (len(wins) / total_breakouts) * 100.0
        avg_ret = df_res[p_col].mean()
        pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 99.0
        avg_bars = df_res[b_col].mean()
        eng_rows.append({"Exit Engine": name, "Win Rate %": f"{wr:.1f}%", "Avg Return %": f"{avg_ret:+.2f}%", "Profit Factor": f"{pf:.2f}x", "Avg Holding": f"{avg_bars:.1f} bars"})

    # 2. Accumulation Ratio
    df_res["acc_bin"] = pd.cut(df_res["acc_ratio"], bins=[-np.inf, 1.0, 1.3, 1.5, 1.8, 2.2, np.inf],
                               labels=["< 1.0 (Dist)", "1.0 - 1.3", "1.3 - 1.5", "1.5 - 1.8", "1.8 - 2.2", "> 2.2 (High Acc)"])
    acc_t = df_res.groupby("acc_bin", observed=False).agg(
        Trades=("raw_win", "count"),
        Winners=("raw_win", "sum"),
        Win_Rate=("raw_win", lambda s: f"{s.mean() * 100:.1f}%"),
        Avg_MFE=("mfe", lambda s: f"+{s.mean():.2f}%"),
        Avg_Exit2_2=("pnl_exit2_2", lambda s: f"{s.mean():+.2f}%")
    ).reset_index().to_dict(orient="records")

    # 3. VDU Depth
    df_res["vdu_bin"] = pd.cut(df_res["vdu_depth"], bins=[-np.inf, 0.45, 0.65, 0.85, 1.10, np.inf],
                               labels=["< 0.45x (Ultra VDU)", "0.45x - 0.65x", "0.65x - 0.85x", "0.85x - 1.10x", "> 1.10x (Elevated)"])
    vdu_t = df_res.groupby("vdu_bin", observed=False).agg(
        Trades=("raw_win", "count"),
        Winners=("raw_win", "sum"),
        Win_Rate=("raw_win", lambda s: f"{s.mean() * 100:.1f}%"),
        Avg_MFE=("mfe", lambda s: f"+{s.mean():.2f}%"),
        Avg_Exit2_2=("pnl_exit2_2", lambda s: f"{s.mean():+.2f}%")
    ).reset_index().to_dict(orient="records")

    # 4. Delivery Surge
    df_res["deliv_bin"] = pd.cut(df_res["dv_surge"], bins=[-np.inf, 1.0, 1.25, 1.6, 2.2, np.inf],
                                  labels=["< 1.0x (Low Deliv)", "1.0x - 1.25x", "1.25x - 1.60x", "1.60x - 2.20x", "> 2.20x (Institutional Surge)"])
    deliv_t = df_res.groupby("deliv_bin", observed=False).agg(
        Trades=("raw_win", "count"),
        Winners=("raw_win", "sum"),
        Win_Rate=("raw_win", lambda s: f"{s.mean() * 100:.1f}%"),
        Avg_MFE=("mfe", lambda s: f"+{s.mean():.2f}%"),
        Avg_Exit2_2=("pnl_exit2_2", lambda s: f"{s.mean():+.2f}%")
    ).reset_index().to_dict(orient="records")

    # 5. Range Closeness
    df_res["rc_bin"] = pd.cut(df_res["rc"], bins=[-np.inf, 0.50, 0.65, 0.80, np.inf],
                                labels=["< 0.50 (Wick Trap)", "0.50 - 0.65", "0.65 - 0.80", "0.80 - 1.00 (Marubozu)"])
    rc_t = df_res.groupby("rc_bin", observed=False).agg(
        Trades=("raw_win", "count"),
        Winners=("raw_win", "sum"),
        Win_Rate=("raw_win", lambda s: f"{s.mean() * 100:.1f}%"),
        Avg_MFE=("mfe", lambda s: f"+{s.mean():.2f}%"),
        Avg_MAE=("mae", lambda s: f"{s.mean():.2f}%"),
        Avg_Exit2_2=("pnl_exit2_2", lambda s: f"{s.mean():+.2f}%")
    ).reset_index().to_dict(orient="records")

    # 6. Base Shelf Spread
    df_res["spread_bin"] = pd.cut(df_res["spread"], bins=[-np.inf, 5.0, 7.0, 8.5, 11.0, np.inf],
                                    labels=["< 5% (Ultra Tight)", "5% - 7%", "7% - 8.5%", "8.5% - 11%", "> 11% (Loose Base)"])
    spread_t = df_res.groupby("spread_bin", observed=False).agg(
        Trades=("raw_win", "count"),
        Winners=("raw_win", "sum"),
        Win_Rate=("raw_win", lambda s: f"{s.mean() * 100:.1f}%"),
        Avg_MFE=("mfe", lambda s: f"+{s.mean():.2f}%"),
        Avg_Exit2_2=("pnl_exit2_2", lambda s: f"{s.mean():+.2f}%")
    ).reset_index().to_dict(orient="records")

    # Package into a lightweight (< 2 KB) JSON payload
    summary_payload = {
        "Total_Breakouts": total_breakouts,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Exit_Engines": eng_rows,
        "Accumulation_Ratio": acc_t,
        "VDU_Depth": vdu_t,
        "Delivery_Surge": deliv_t,
        "Range_Closeness": rc_t,
        "Base_Spread": spread_t
    }

    with open("data/nifty100_derivation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    df_res.to_csv("data/nifty100_derivation_dump.csv", index=False)
    print("✅ Saved summary matrix to `data/nifty100_derivation_summary.json`.")
if __name__ == "__main__":
    run_nifty100_derivation()
