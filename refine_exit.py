import os
import glob
import json
import datetime
import numpy as np
import pandas as pd

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "refine_exit_results.json")
TRADES_CSV = os.path.join(DATA_DIR, "refine_exit_trades.csv")
MIN_TURNOVER_CR = 10.0

EXCLUDED = {
    "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
    "fundamentals.json", "backtest_results.json", "swing3_results.json",
    "swing3_run2_results.json", "swing3_run2_global_results.json",
    "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
    "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
    "swing_ultra750_results.json", "swing_ultraN750_results.json",
    "refine_exit_results.json", "multibagger_trades.csv"
}

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

def load_benchmark():
    for f in ["nifty.json", "nifty50.json", "NIFTY.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) > 50:
                    df = pd.DataFrame(clean)
                    df["perf_60d"] = df["close"].pct_change(60).fillna(0)
                    return {r["time"]: float(r["perf_60d"]) for _, r in df.iterrows()}
            except Exception:
                continue
    return {}

def backtest_tri_exits(symbol, clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return [], [], []

    df = pd.DataFrame(clean_data)
    for col in ["open", "high", "low", "close", "volume", "delivery_vol", "deliv_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Technical Suite
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    # Range Closeness
    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

    # ATR 14
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=1).mean()

    # Pre-calculate Distribution Days
    dist_days = np.zeros(len(df), dtype=int)
    for k in range(1, len(df)):
        c_down = df["close"].iloc[k] < df["close"].iloc[k - 1]
        heavy_deliv = df["delivery_vol"].iloc[k] >= (1.35 * df["deliv_sma20"].iloc[k])
        heavy_dp = df["deliv_pct"].iloc[k] >= (1.15 * df["deliv_pct_50d"].iloc[k])
        weak_close = df["range_closeness"].iloc[k] <= 0.45
        if c_down and (heavy_deliv or heavy_dp) and weak_close:
            dist_days[k] = 1

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_avgs = df["gross_vol_sma20"].values
    deliv_vols = df["delivery_vol"].values
    deliv_avgs = df["deliv_sma20"].values
    to_50d = df["turnover_50d"].values
    range_closes = df["range_closeness"].values
    ema10s = df["ema_10"].values
    ema20s = df["ema_20"].values
    sma200s = df["sma_200"].values
    perf_60s = df["perf_60d"].values
    atr14s = df["atr_14"].values
    times = df["time"].values

    t1_trades = []
    t2_trades = []
    t3_trades = []

    # Position 1: Pure 20 EMA
    in_t1 = False
    ep1, sl1, ed1, idx1, setup1 = 0.0, 0.0, "", 0, ""

    # Position 2: Exit 2 (Original VDU 5% Buffer)
    in_t2 = False
    ep2, sl2, ed2, idx2, setup2, hc2 = 0.0, 0.0, "", 0, "", 0.0

    # Position 3: Exit 2.1 (Adaptive ATR Buffer + 5-Session Reclaim + 50% Lock)
    in_t3 = False
    ep3, sl3, ed3, idx3, setup3, hc3 = 0.0, 0.0, "", 0, "", 0.0
    vdu_under_count3 = 0

    for i in range(40, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        to = to_50d[i]
        rc = range_closes[i]
        ema10 = ema10s[i]
        ema20 = ema20s[i]
        sma200 = sma200s[i]
        stock_p60 = perf_60s[i]
        atr14 = atr14s[i]
        t = times[i]

        is_vdu = (dv <= 0.70 * dv_avg) and (v <= 0.80 * v_avg)
        dist_cluster = int(dist_days[max(0, i - 12):i + 1].sum())

        # ==============================================================
        # 1. PURE 20 EMA (Rank 1 Baseline)
        # ==============================================================
        if in_t1:
            cur_sl1 = max(sl1, round(float(ema20), 2))
            if c < cur_sl1:
                pnl = round(((cur_sl1 - ep1) / ep1) * 100.0, 2)
                t1_trades.append({
                    "symbol": symbol, "engine": "Pure 20 EMA", "setup": setup1,
                    "entry_date": ed1, "exit_date": t, "pnl_%": pnl,
                    "holding_days": i - idx1, "is_win": pnl > 0
                })
                in_t1 = False

        # ==============================================================
        # 2. EXIT 2 (Original VDU 5% Buffer)
        # ==============================================================
        if in_t2:
            hc2 = max(hc2, c)
            unrealized_gain2 = (hc2 - ep2) / ep2

            if c < sl2:
                pnl = round(((sl2 - ep2) / ep2) * 100.0, 2)
                t2_trades.append({
                    "symbol": symbol, "engine": "Exit 2 (VDU 5% Buffer)", "setup": setup2,
                    "entry_date": ed2, "exit_date": t, "pnl_%": pnl,
                    "holding_days": i - idx2, "is_win": pnl > 0
                })
                in_t2 = False
                continue

            is_climax2 = (dv >= 2.0 * dv_avg) and (rc <= 0.35) and (unrealized_gain2 >= 0.20)
            active_ema2 = ema10 if unrealized_gain2 >= 0.25 else ema20

            exit_trig2 = False
            exit_px2 = c

            if is_climax2:
                exit_trig2 = True
            elif dist_cluster >= 3 and c < active_ema2:
                exit_trig2 = True
            elif c < ema20:
                if not is_vdu:
                    exit_trig2 = True
                    exit_px2 = round(float(ema20), 2)
                else:
                    guaranteed_breakeven = round(ep2 * 1.015, 2)
                    vdu_buffer_stop = max(guaranteed_breakeven, round(float(ema20 * 0.95), 2))
                    if c < vdu_buffer_stop:
                        exit_trig2 = True
                        exit_px2 = vdu_buffer_stop

            if exit_trig2:
                pnl = round(((exit_px2 - ep2) / ep2) * 100.0, 2)
                t2_trades.append({
                    "symbol": symbol, "engine": "Exit 2 (VDU 5% Buffer)", "setup": setup2,
                    "entry_date": ed2, "exit_date": t, "pnl_%": pnl,
                    "holding_days": i - idx2, "is_win": pnl > 0
                })
                in_t2 = False
                continue

        # ==============================================================
        # 3. EXIT 2.1 (VDU 5% + ATR Adaptive + 5-Session Reclaim + 50% Lock)
        # ==============================================================
        if in_t3:
            hc3 = max(hc3, c)
            unrealized_gain3 = (hc3 - ep3) / ep3

            # Hard catastrophe stop
            if c < sl3:
                pnl = round(((sl3 - ep3) / ep3) * 100.0, 2)
                t3_trades.append({
                    "symbol": symbol, "engine": "Exit 2.1 (5D Reclaim + ATR)", "setup": setup3,
                    "entry_date": ed3, "exit_date": t, "exit_reason": "INITIAL SL",
                    "pnl_%": pnl, "holding_days": i - idx3, "is_win": pnl > 0
                })
                in_t3 = False
                continue

            is_climax3 = (dv >= 2.0 * dv_avg) and (rc <= 0.35) and (unrealized_gain3 >= 0.20)
            active_ema3 = ema10 if unrealized_gain3 >= 0.25 else ema20

            exit_trig3 = False
            exit_reason3 = ""
            exit_px3 = c

            # Mega-Runner Profit Lock (Gain >= 50%)
            if unrealized_gain3 >= 0.50:
                mega_floor = max(round(float(ema10), 2), round(hc3 * 0.90, 2))
                if c < mega_floor:
                    exit_trig3 = True
                    exit_px3 = mega_floor
                    exit_reason3 = "MEGA-RUNNER 50%+ LOCK"

            # Climax Churn Exit
            elif is_climax3:
                exit_trig3 = True
                exit_reason3 = "CLIMAX CHURN"

            # Clustered Distribution Breakdown
            elif dist_cluster >= 3 and c < active_ema3:
                exit_trig3 = True
                exit_reason3 = f"DIST CLUSTER ({dist_cluster}D) + EMA"

            # 20 EMA Breach Engine
            elif c < ema20:
                if not is_vdu:
                    # Heavy volume breach: Immediate exit at 20 EMA
                    exit_trig3 = True
                    exit_px3 = round(float(ema20), 2)
                    exit_reason3 = "EMA BREAK (HEAVY VOL)"
                else:
                    # Volume is DRY (VDU)
                    vdu_under_count3 += 1

                    # Adaptive Stop: Min of 5% and 1.5 * ATR_14
                    atr_buffer_price = round(float(ema20 - (1.5 * atr14)), 2)
                    pct_buffer_price = round(float(ema20 * 0.95), 2)
                    dynamic_buffer_floor = min(pct_buffer_price, atr_buffer_price)

                    # Guaranteed Breakeven clamp
                    guaranteed_breakeven = round(ep3 * 1.015, 2)
                    vdu_buffer_stop = max(guaranteed_breakeven, dynamic_buffer_floor)

                    if c < vdu_buffer_stop:
                        exit_trig3 = True
                        exit_px3 = vdu_buffer_stop
                        exit_reason3 = "VDU ADAPTIVE BUFFER BREAK"
                    elif vdu_under_count3 > 5:
                        exit_trig3 = True
                        exit_px3 = c
                        exit_reason3 = "VDU 5-SESSION RECLAIM FAILED"
            else:
                # Price reclaimed back above 20 EMA: Reset reclaim counter
                vdu_under_count3 = 0

            if exit_trig3:
                pnl = round(((exit_px3 - ep3) / ep3) * 100.0, 2)
                t3_trades.append({
                    "symbol": symbol, "engine": "Exit 2.1 (5D Reclaim + ATR)", "setup": setup3,
                    "entry_date": ed3, "exit_date": t, "exit_reason": exit_reason3,
                    "pnl_%": pnl, "holding_days": i - idx3, "is_win": pnl > 0
                })
                in_t3 = False
                continue

        # ==============================================================
        # SWING ULTRA N750 ENTRY CRITERIA
        # ==============================================================
        if to < MIN_TURNOVER_CR:
            continue

        is_st2 = bool(c >= sma200) if pd.notnull(sma200) else False
        n_p = nifty_perf_map.get(t)
        if n_p is None:
            prior = [d for d in nifty_perf_map.keys() if d <= t]
            n_p = nifty_perf_map[max(prior)] if prior else 0.001
        is_rs = bool(stock_p60 > n_p)

        if not (is_st2 and is_rs):
            continue

        entry_trig = False
        calc_entry, calc_sl, cur_setup = 0.0, 0.0, ""

        # Engine 1: V-Reversal Reclaim
        r20_h = highs[i - 20:i].max()
        if (r20_h - l) / r20_h >= 0.18:
            trough = lows[i - 5:i].min()
            p3_h = highs[i - 4:i].max()
            if (c >= p3_h) and (c > closes[i - 1]) and (v >= 1.3 * v_avg) and (dv >= 1.25 * dv_avg) and (rc >= 0.55):
                entry_trig = True
                calc_entry = round(p3_h, 2)
                calc_sl = round(trough * 0.995, 2)
                cur_setup = "V-REVERSAL"

        # Engine 2: Launchpad Breakout
        if not entry_trig:
            for s_len in range(10, 19):
                s_h = highs[i - s_len:i].max()
                s_l = lows[i - s_len:i].min()
                if s_l > 0 and ((s_h - s_l) / s_l) <= 0.13:
                    u_dv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                    d_dv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                    r_ratio = (u_dv / d_dv) if d_dv > 0 else 1.15
                    if (c >= s_h) and (h >= s_h) and (v >= 1.4 * v_avg) and (dv >= 1.25 * dv_avg) and (rc >= 0.65) and (r_ratio >= 1.15):
                        entry_trig = True
                        calc_entry = round(float(s_h), 2)
                        calc_sl = round(min(l, s_l), 2)
                        cur_setup = "LAUNCHPAD"
                        break

        if entry_trig:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                if not in_t1:
                    in_t1 = True; ep1 = calc_entry; sl1 = calc_sl; ed1 = t; idx1 = i; setup1 = cur_setup
                if not in_t2:
                    in_t2 = True; ep2 = calc_entry; sl2 = calc_sl; ed2 = t; idx2 = i; setup2 = cur_setup; hc2 = c
                if not in_t3:
                    in_t3 = True; ep3 = calc_entry; sl3 = calc_sl; ed3 = t; idx3 = i; setup3 = cur_setup; hc3 = c; vdu_under_count3 = 0

    return t1_trades, t2_trades, t3_trades

def run_pipeline():
    nifty_map = load_benchmark()
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    target_files = [f for f in json_files if os.path.basename(f) not in EXCLUDED]
    print(f"Scanning {len(target_files)} NSE stocks across Pure 20 EMA, Exit 2, and Exit 2.1...")

    all_t1, all_t2, all_t3 = [], [], []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                t1, t2, t3 = backtest_tri_exits(sym, clean, nifty_map)
                all_t1.extend(t1)
                all_t2.extend(t2)
                all_t3.extend(t3)
        except Exception:
            continue

    def calc_stats(trades):
        if not trades:
            return {}
        df = pd.DataFrame(trades)
        wins = df[df["is_win"]]
        losses = df[~df["is_win"]]
        tot = len(df)
        wr = (len(wins) / tot) * 100.0 if tot else 0
        g_gain = wins["pnl_%"].sum()
        g_loss = abs(losses["pnl_%"].sum())
        pf = round(g_gain / g_loss, 2) if g_loss > 0 else 999.0
        return {
            "Total_Trades": tot,
            "Win_Rate_%": round(wr, 2),
            "Profit_Factor": pf,
            "Average_Gain_%": round(wins["pnl_%"].mean(), 2) if len(wins) else 0,
            "Average_Loss_%": round(losses["pnl_%"].mean(), 2) if len(losses) else 0,
            "Average_Holding_Days": round(df["holding_days"].mean(), 1) if tot else 0,
            "Max_Gain_%": round(df["pnl_%"].max(), 2) if tot else 0,
            "Exit_Reasons": df["exit_reason"].value_counts().to_dict() if "exit_reason" in df.columns else {}
        }

    s1 = calc_stats(all_t1)
    s2 = calc_stats(all_t2)
    s3 = calc_stats(all_t3)

    output = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Pure_20_EMA": s1,
        "Exit_2_VDU_5pct": s2,
        "Exit_2_1_Refined": s3,
        "Top_15_Exit_2_1_Winners": pd.DataFrame(all_t3).sort_values(by="pnl_%", ascending=False).head(15).to_dict(orient="records") if all_t3 else []
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    pd.DataFrame(all_t3).to_csv(TRADES_CSV, index=False)

    print("\n" + "=" * 84)
    print("      TRI-STRATEGY BENCHMARK: PURE 20 EMA vs. EXIT 2 vs. EXIT 2.1")
    print("=" * 84)
    print(f"{'Metric':<22} | {'Pure 20 EMA':<16} | {'Exit 2 (5% Buffer)':<18} | {'Exit 2.1 (Refined)':<18}")
    print("-" * 84)
    print(f"{'Total Trades':<22} | {s1.get('Total_Trades',0):<16,} | {s2.get('Total_Trades',0):<18,} | {s3.get('Total_Trades',0):<18,}")
    print(f"{'Win Rate (%)':<22} | {s1.get('Win_Rate_%',0):<16.2f}% | {s2.get('Win_Rate_%',0):<18.2f}% | {s3.get('Win_Rate_%',0):<18.2f}%")
    print(f"{'Profit Factor (PF)':<22} | {s1.get('Profit_Factor',0):<16.2f} | {s2.get('Profit_Factor',0):<18.2f} | {s3.get('Profit_Factor',0):<18.2f}")
    print(f"{'Avg Gain / Win':<22} | +{s1.get('Average_Gain_%',0):<15.2f}% | +{s2.get('Average_Gain_%',0):<17.2f}% | +{s3.get('Average_Gain_%',0):<17.2f}%")
    print(f"{'Avg Loss / Loss':<22} | {s1.get('Average_Loss_%',0):<16.2f}% | {s2.get('Average_Loss_%',0):<18.2f}% | {s3.get('Average_Loss_%',0):<18.2f}%")
    print(f"{'Max Gain':<22} | +{s1.get('Max_Gain_%',0):<15.2f}% | +{s2.get('Max_Gain_%',0):<17.2f}% | +{s3.get('Max_Gain_%',0):<17.2f}%")
    print(f"{'Avg Hold (Days)':<22} | {s1.get('Average_Holding_Days',0):<16.1f} | {s2.get('Average_Holding_Days',0):<18.1f} | {s3.get('Average_Holding_Days',0):<18.1f}")
    print("=" * 84)

if __name__ == "__main__":
    run_pipeline()
