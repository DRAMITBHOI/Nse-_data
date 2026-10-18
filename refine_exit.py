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


def backtest_stock_distribution(symbol, clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return []

    df = pd.DataFrame(clean_data)
    for col in ["open", "high", "low", "close", "volume", "delivery_vol", "deliv_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Indicator Suite
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    # Range Closeness
    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

    # True Demat OBV
    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["dobv_sma20"] = df["deliv_obv"].rolling(20, min_periods=1).mean()

    # Pre-calculate Distribution Days (1 = Distribution Day, 0 = Clean)
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
    deliv_pcts = df["deliv_pct"].values
    deliv_pct_avgs = df["deliv_pct_50d"].values
    to_50d = df["turnover_50d"].values
    range_closes = df["range_closeness"].values
    ema20s = df["ema_20"].values
    sma200s = df["sma_200"].values
    perf_60s = df["perf_60d"].values
    times = df["time"].values

    trades = []
    in_trade = False
    entry_p = 0.0
    initial_sl = 0.0
    entry_d = ""
    entry_idx = 0
    trade_setup = ""

    for i in range(40, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        turnover = to_50d[i]
        dp = deliv_pcts[i]
        dp_avg = deliv_pct_avgs[i]
        rc = range_closes[i]
        ema = ema20s[i]
        sma200 = sma200s[i]
        stock_p60 = perf_60s[i]
        t = times[i]

        # ==========================================
        # INSTITUTIONAL DISTRIBUTION EXIT ENGINE
        # ==========================================
        if in_trade:
            # Floor Stop: trailing lowest low since entry (structural support)
            recent_low = lows[max(entry_idx, i - 10):i].min() if i > entry_idx else initial_sl
            structural_stop = max(initial_sl, round(float(recent_low * 0.99), 2))

            # Distribution Metrics
            dist_cluster = int(dist_days[max(0, i - 14):i + 1].sum())
            is_climactic = (dv >= 1.80 * dv_avg) and (rc <= 0.35) and (c > entry_p * 1.15)
            is_vdu = (dv <= 0.70 * dv_avg) and (v <= 0.80 * v_avg)

            exit_triggered = False
            exit_reason = ""
            exit_price = c

            if c < structural_stop:
                exit_triggered = True
                exit_price = structural_stop
                exit_reason = "STRUCTURAL FLOOR STOP"
            elif is_climactic:
                exit_triggered = True
                exit_reason = "CLIMAX CHURN (TOP LIQUIDATION)"
            elif dist_cluster >= 3:
                exit_triggered = True
                exit_reason = f"DISTRIBUTION CLUSTER ({dist_cluster}D)"
            elif c < ema and not is_vdu:
                exit_triggered = True
                exit_reason = "EMA BREAK (HEAVY VOLUME)"

            if exit_triggered:
                pnl_pts = exit_price - entry_p
                pnl_pct = round((pnl_pts / entry_p) * 100.0, 2)
                holding_days = i - entry_idx

                trades.append({
                    "symbol": symbol,
                    "setup": trade_setup,
                    "entry_date": entry_d,
                    "entry_price": round(entry_p, 2),
                    "exit_date": t,
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "pnl_%": pnl_pct,
                    "holding_days": holding_days,
                    "is_win": pnl_pts > 0
                })
                in_trade = False
            continue

        # ==========================================
        # SWING ULTRA N750 ENTRY CRITERIA
        # ==========================================
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
            c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg)
            c_candle_rev = rc >= 0.55

            if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                entry_triggered = True
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)
                cur_setup = "V-REVERSAL"

        # Engine 2: Launchpad Breakout (Shelf Spread <= 13%)
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
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg)
                c_candle_bo = rc >= 0.65

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.15):
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)
                    cur_setup = "LAUNCHPAD"

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                in_trade = True
                entry_p = calc_entry
                initial_sl = calc_sl
                entry_d = t
                entry_idx = i
                trade_setup = cur_setup

    return trades


def run_backtest_pipeline():
    nifty_perf_map = load_nifty_benchmark()
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing3_results.json",
        "swing3_run2_results.json", "swing3_run2_global_results.json",
        "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
        "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
        "swing_ultra750_results.json", "swing_ultraN750_results.json",
        "refine_exit_results.json"
    }

    # Applied across the whole NSE Universe
    target_files = [f for f in json_files if os.path.basename(f) not in excluded]
    print(f"🔍 Running Institutional Distribution Backtest across {len(target_files)} NSE stocks...")

    all_trades = []
    for idx, path in enumerate(sorted(target_files)):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                t_list = backtest_stock_distribution(sym, clean, nifty_perf_map)
                all_trades.extend(t_list)
        except Exception:
            continue

    if not all_trades:
        print("⚠ No trades generated. Check data directory.")
        return

    df_res = pd.DataFrame(all_trades)
    df_res.to_csv(TRADES_CSV, index=False)

    total_trades = len(df_res)
    win_trades = df_res[df_res["is_win"]]
    loss_trades = df_res[~df_res["is_win"]]
    win_rate = round((len(win_trades) / total_trades) * 100, 2)

    gross_profit = win_trades["pnl_%"].sum()
    gross_loss = abs(loss_trades["pnl_%"].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0

    avg_gain = round(win_trades["pnl_%"].mean(), 2) if len(win_trades) > 0 else 0.0
    avg_loss = round(loss_trades["pnl_%"].mean(), 2) if len(loss_trades) > 0 else 0.0
    avg_hold = round(df_res["holding_days"].mean(), 1)

    reason_breakdown = df_res["exit_reason"].value_counts().to_dict()

    summary = {
        "Backtest_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Universe": "Entire Liquid NSE Universe (Turnover >= 10 Cr)",
        "Strategy": "Swing Ultra N750 Entry + Institutional Distribution Machine Exit",
        "Total_Trades": total_trades,
        "Win_Rate_%": win_rate,
        "Profit_Factor": profit_factor,
        "Average_Gain_%": avg_gain,
        "Average_Loss_%": avg_loss,
        "Average_Holding_Days": avg_hold,
        "Exit_Reasons_Distribution": reason_breakdown,
        "Top_15_Winning_Trades": df_res.sort_values(by="pnl_%", ascending=False).head(15).to_dict(orient="records"),
        "Top_10_Losses": df_res.sort_values(by="pnl_%", ascending=True).head(10).to_dict(orient="records")
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 55)
    print(" 🚀 REFINED EXIT BACKTEST SUMMARY")
    print("=" * 55)
    print(f"Total Trades        : {total_trades}")
    print(f"Win Rate            : {win_rate}%")
    print(f"Profit Factor       : {profit_factor}")
    print(f"Average Win / Loss  : +{avg_gain}% / {avg_loss}%")
    print(f"Average Hold Period : {avg_hold} days")
    print("Exit Breakdown      :")
    for reason, count in reason_breakdown.items():
        print(f"  • {reason:<32}: {count} ({round(count / total_trades * 100, 1)}%)")
    print("=" * 55)
    print(f"Saved artifacts to {RESULTS_JSON} and {TRADES_CSV}\n")


if __name__ == "__main__":
    run_backtest_pipeline()
