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

def backtest_both_exits(symbol, clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return [], []

    df = pd.DataFrame(clean_data)
    for col in ["open", "high", "low", "close", "volume", "delivery_vol", "deliv_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

    # Pre-calculate Institutional Distribution Days
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
    ema20s = df["ema_20"].values
    sma200s = df["sma_200"].values
    perf_60s = df["perf_60d"].values
    times = df["time"].values

    ema_trades = []
    mod_trades = []

    # Position Tracker: Pure EMA 20
    in_trade_ema = False
    entry_p_ema = 0.0
    initial_sl_ema = 0.0
    entry_d_ema = ""
    entry_idx_ema = 0
    setup_ema = ""

    # Position Tracker: Modified Exit
    in_trade_mod = False
    entry_p_mod = 0.0
    initial_sl_mod = 0.0
    entry_d_mod = ""
    entry_idx_mod = 0
    setup_mod = ""
    highest_close_mod = 0.0

    for i in range(40, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        turnover = to_50d[i]
        rc = range_closes[i]
        ema = ema20s[i]
        sma200 = sma200s[i]
        stock_p60 = perf_60s[i]
        t = times[i]

        # -------------------------------------------------------------
        # 1. PURE 20 EMA EXIT (Rank 1: Hold initial SL until EMA > Entry)
        # -------------------------------------------------------------
        if in_trade_ema:
            cur_sl_ema = max(initial_sl_ema, round(float(ema), 2))
            if c < cur_sl_ema:
                exit_price = cur_sl_ema
                pnl_pts = exit_price - entry_p_ema
                ema_trades.append({
                    "symbol": symbol,
                    "engine": "EMA 20",
                    "setup": setup_ema,
                    "entry_date": entry_d_ema,
                    "entry_price": entry_p_ema,
                    "exit_date": t,
                    "exit_price": exit_price,
                    "exit_reason": "EMA 20 BREAK",
                    "pnl_%": round((pnl_pts / entry_p_ema) * 100.0, 2),
                    "holding_days": i - entry_idx_ema,
                    "is_win": pnl_pts > 0
                })
                in_trade_ema = False

        # -------------------------------------------------------------
        # 2. MODIFIED EXIT (Persistent Latch + Protected Breakeven Floor)
        # -------------------------------------------------------------
        if in_trade_mod:
            highest_close_mod = max(highest_close_mod, c)

            # A. Fixed initial disaster stop (No rolling 10-day floor)
            if c < initial_sl_mod:
                exit_px = initial_sl_mod
                pnl_pts = exit_px - entry_p_mod
                mod_trades.append({
                    "symbol": symbol,
                    "engine": "MODIFIED EXIT",
                    "setup": setup_mod,
                    "entry_date": entry_d_mod,
                    "entry_price": entry_p_mod,
                    "exit_date": t,
                    "exit_price": exit_px,
                    "exit_reason": "INITIAL SL",
                    "pnl_%": round((pnl_pts / entry_p_mod) * 100.0, 2),
                    "holding_days": i - entry_idx_mod,
                    "is_win": pnl_pts > 0
                })
                in_trade_mod = False
                continue

            # B. Institutional Distribution Latch (10-session lookback window)
            dist_in_window = int(dist_days[max(0, i - 10):i + 1].sum())
            is_climactic = (dv >= 1.80 * dv_avg) and (rc <= 0.35) and (c > entry_p_mod * 1.15)
            has_dist_memory = (dist_in_window >= 1) or is_climactic

            # C. Protected Drawdown Retracement (Never falls below Entry + 1.5%)
            gain_from_entry = (highest_close_mod - entry_p_mod) / entry_p_mod
            is_profit_protected = False
            if gain_from_entry >= 0.15:
                guaranteed_floor = round(entry_p_mod * 1.015, 2)
                drawdown_trigger_price = max(guaranteed_floor, round(highest_close_mod * 0.88, 2))
                if c < drawdown_trigger_price:
                    is_profit_protected = True

            # D. Volume Dry Check
            is_vdu = (dv <= 0.70 * dv_avg) and (v <= 0.80 * v_avg)

            exit_trig = False
            exit_reason = ""
            exit_px = c

            if c < ema:
                if has_dist_memory:
                    exit_trig = True
                    exit_reason = f"EMA + DIST ({dist_in_window}D)"
                elif is_profit_protected:
                    exit_trig = True
                    exit_reason = "PROFIT RETRACEMENT CAP"
                elif not is_vdu:
                    exit_trig = True
                    exit_reason = "EMA BREAK (HEAVY VOL)"
                else:
                    # Low-volume VDU pullback without prior distribution: HOLD
                    pass

            if exit_trig:
                pnl_pts = exit_px - entry_p_mod
                mod_trades.append({
                    "symbol": symbol,
                    "engine": "MODIFIED EXIT",
                    "setup": setup_mod,
                    "entry_date": entry_d_mod,
                    "entry_price": entry_p_mod,
                    "exit_date": t,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "pnl_%": round((pnl_pts / entry_p_mod) * 100.0, 2),
                    "holding_days": i - entry_idx_mod,
                    "is_win": pnl_pts > 0
                })
                in_trade_mod = False
                continue

        # -------------------------------------------------------------
        # 3. SWING ULTRA N750 ENTRY (IDENTICAL FOR BOTH)
        # -------------------------------------------------------------
        if turnover < MIN_TURNOVER_CR:
            continue

        is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
        n_perf = nifty_perf_map.get(t)
        if n_perf is None:
            prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
            n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
        is_rs = bool(stock_p60 > n_perf)

        if not (is_stage2 and is_rs):
            continue

        entry_triggered = False
        calc_entry = 0.0
        calc_sl = 0.0
        cur_setup = ""

        # Engine 1: V-Reversal Reclaim
        recent_20_high = highs[i - 20:i].max()
        if (recent_20_high - l) / recent_20_high >= 0.18:
            recent_trough = lows[i - 5:i].min()
            prior_3d_high = highs[i - 4:i].max()
            c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
            c_vol_rev = v >= (1.3 * v_avg)
            c_deliv_rev = (dv >= 1.25 * dv_avg)
            c_candle_rev = rc >= 0.55

            if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                entry_triggered = True
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)
                cur_setup = "V-REVERSAL"

        # Engine 2: Launchpad Breakout (Shelf <= 13%)
        if not entry_triggered:
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= 0.13:
                    valid_launchpad = True
                    launchpad_high = round(float(s_high), 2)
                    launchpad_low = round(float(s_low), 2)
                    break

            if valid_launchpad:
                base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.15

                c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg)
                c_candle_bo = rc >= 0.65

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.15):
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)
                    cur_setup = "LAUNCHPAD"

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                if not in_trade_ema:
                    in_trade_ema = True
                    entry_p_ema = calc_entry
                    initial_sl_ema = calc_sl
                    entry_d_ema = t
                    entry_idx_ema = i
                    setup_ema = cur_setup

                if not in_trade_mod:
                    in_trade_mod = True
                    entry_p_mod = calc_entry
                    initial_sl_mod = calc_sl
                    entry_d_mod = t
                    entry_idx_mod = i
                    setup_mod = cur_setup
                    highest_close_mod = c

    return ema_trades, mod_trades

def run_pipeline():
    nifty_map = load_benchmark()
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    target_files = [f for f in json_files if os.path.basename(f) not in EXCLUDED]
    print(f"Scanning {len(target_files)} NSE stocks for Head-to-Head Exit Comparison...")

    all_ema = []
    all_mod = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                e_t, m_t = backtest_both_exits(sym, clean, nifty_map)
                all_ema.extend(e_t)
                all_mod.extend(m_t)
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

    s_ema = calc_stats(all_ema)
    s_mod = calc_stats(all_mod)

    output = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "EMA_20_Exit": s_ema,
        "Modified_Dual_Exit": s_mod,
        "Top_15_Modified_Winners": pd.DataFrame(all_mod).sort_values(by="pnl_%", ascending=False).head(15).to_dict(orient="records") if all_mod else []
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    pd.DataFrame(all_mod).to_csv(TRADES_CSV, index=False)

    print("\n" + "=" * 70)
    print("      HEAD-TO-HEAD BACKTEST: 20 EMA vs. MODIFIED DUAL EXIT")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Pure 20 EMA Exit':<18} | {'Modified Dual Exit':<18}")
    print("-" * 70)
    print(f"{'Total Trades':<25} | {s_ema.get('Total_Trades',0):<18,} | {s_mod.get('Total_Trades',0):<18,}")
    print(f"{'Win Rate (%)':<25} | {s_ema.get('Win_Rate_%',0):<18.2f}% | {s_mod.get('Win_Rate_%',0):<18.2f}%")
    print(f"{'Profit Factor (PF)':<25} | {s_ema.get('Profit_Factor',0):<18.2f} | {s_mod.get('Profit_Factor',0):<18.2f}")
    print(f"{'Avg Gain / Win':<25} | +{s_ema.get('Average_Gain_%',0):<17.2f}% | +{s_mod.get('Average_Gain_%',0):<17.2f}%")
    print(f"{'Avg Loss / Loss':<25} | {s_ema.get('Average_Loss_%',0):<18.2f}% | {s_mod.get('Average_Loss_%',0):<18.2f}%")
    print(f"{'Max Gain':<25} | +{s_ema.get('Max_Gain_%',0):<17.2f}% | +{s_mod.get('Max_Gain_%',0):<17.2f}%")
    print(f"{'Avg Hold (Days)':<25} | {s_ema.get('Average_Holding_Days',0):<18.1f} | {s_mod.get('Average_Holding_Days',0):<18.1f}")
    print("=" * 70)
    print("\nModified Exit Breakdown:")
    for reason, count in s_mod.get("Exit_Reasons", {}).items():
        pct = (count / s_mod.get('Total_Trades', 1)) * 100
        print(f"  • {reason:<28}: {count:>5,} trades ({pct:>5.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
