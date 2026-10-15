import os
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
MIN_TURNOVER_CR = 10.0

def load_universe_lists():
    n750_symbols = set()
    for p in ["./data/nifty750.json", "./nifty750.json", "nifty750.json"]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        n750_symbols = {str(x).strip().upper() for x in d}
                    elif isinstance(d, dict):
                        n750_symbols = {str(x).strip().upper() for x in d.keys()}
                break
            except Exception:
                pass

    local_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing3_results.json",
        "swing3_run2_results.json", "swing3_run2_global_results.json",
        "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
        "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
        "backtest_hp3_report.json", "live_intraday_breakouts.json"
    }

    all_symbols = [
        os.path.splitext(os.path.basename(f))[0].upper()
        for f in local_files if os.path.basename(f) not in excluded
    ]

    if len(all_symbols) < 50:
        all_symbols = sorted(list(n750_symbols)) + [
            "SIRCA", "RAIN", "NITINSPIN", "TANLA", "STOVEKRAFT", "BEPL", "CEIGALL",
            "TEXRAIL", "RELINFRA", "URJA", "CYIENTDLM", "WEBELSOLAR", "LXCHEM", "MANINFRA"
        ]

    return n750_symbols, all_symbols

def prepare_ohlcv(df):
    if len(df) < 60:
        return None

    df = df.copy()
    if "Close" in df.columns:
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })

    if "delivery_vol" not in df.columns:
        df["deliv_pct"] = 45.0
        df["delivery_vol"] = df["volume"] * 0.45

    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

    return df

def simulate_engine(df, nifty_perf_map, mode="SWING3_LOCKED"):
    """
    mode:
      - 'SWING3_LOCKED': Spread <= 8%, Conviction Score >= 2.5
      - 'RUN2': Spread <= 13%, No Score Cutoff
    """
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
    demat_obvs = df["deliv_obv"].values
    demat_obv_maxes = df["dobv_20max"].values
    times = df.index.astype(str).values

    trades = []
    in_trade = False
    entry_p = 0.0
    initial_sl = 0.0
    current_sl = 0.0
    entry_idx = 0
    setup_type = ""

    for i in range(50, len(df)):
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
        t = times[i][:10]

        # Exact Production Exit: Close < current_sl (20 EMA Trail)
        if in_trade:
            current_sl = max(initial_sl, round(float(ema), 2))
            if c < current_sl:
                exit_price = c
                pnl = (exit_price - entry_p) / entry_p
                trades.append({
                    "setup": setup_type,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                    "is_win": pnl > 0
                })
                in_trade = False
            continue

        if turnover < MIN_TURNOVER_CR:
            continue

        is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
        n_perf = nifty_perf_map.get(t, 0.001)
        is_rs = bool(stock_p60 > n_perf)

        if not (is_stage2 and is_rs):
            continue

        entry_triggered = False
        calc_entry = 0.0
        calc_sl = 0.0
        setup_label = ""

        # 1. ENGINE 1: V-REVERSAL (FLUSH >= 18%)
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
                setup_label = "V-REVERSAL"

        # 2. ENGINE 2: LAUNCHPAD BREAKOUT
        if not entry_triggered:
            max_spread = 0.08 if mode == "SWING3_LOCKED" else 0.13
            valid_lp = False
            lp_high = 0.0
            lp_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= max_spread:
                    valid_lp = True
                    lp_high = round(float(s_high), 2)
                    lp_low = round(float(s_low), 2)
                    break

            if valid_lp:
                base_up = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                base_dn = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                req_ratio = 1.50 if mode == "SWING3_LOCKED" else 1.15
                base_cumul = (base_up / base_dn) if base_dn > 0 else req_ratio

                c_bo = (c >= lp_high) and (h >= lp_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg)
                c_candle_bo = rc >= 0.65
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i - 1]) if mode == "SWING3_LOCKED" else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul >= req_ratio) and obv_high:
                    entry_triggered = True
                    calc_entry = lp_high
                    calc_sl = round(min(l, lp_low), 2)
                    setup_label = "LAUNCHPAD"

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
                rs_rat = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                conv_score = round(deliv_surge * max(0.1, rs_rat), 3)

                # Gate check: Score >= 2.5 for Swing 3 Locked; Unconstrained for Run 2
                score_pass = (conv_score >= 2.5) if mode == "SWING3_LOCKED" else True

                if score_pass:
                    in_trade = True
                    entry_p = calc_entry
                    initial_sl = calc_sl
                    current_sl = initial_sl
                    entry_idx = i
                    setup_type = setup_label

    return trades

def main():
    print("🚀 Initializing Comparative Backtest (2021 – Present)...")
    n750_set, all_symbols = load_universe_lists()

    # Nifty 50 benchmark
    nifty_df = yf.download("^NSEI", start="2020-01-01", progress=False, auto_adjust=True)
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)
    nifty_df["perf_60d"] = nifty_df["Close"].pct_change(60).fillna(0)
    nifty_perf_map = {str(k)[:10]: float(v) for k, v in nifty_df["perf_60d"].items()}

    buckets = {
        "Swing 3 Locked - N750 (Spread<=8% • Score>=2.5)": [],
        "Swing 3 Run 2 - N750 (Spread<=13% • No Score Gate)": [],
        "Swing 3 Run 2 - Non-N750 (Spread<=13% • No Score Gate)": []
    }

    batch_size = 50
    for b in range(0, len(all_symbols), batch_size):
        chunk = all_symbols[b : b + batch_size]
        tickers = [f"{s}.NS" for s in chunk]
        print(f"Processing Batch {b // batch_size + 1}/{(len(all_symbols) + batch_size - 1) // batch_size}...")

        try:
            data = yf.download(tickers, start="2020-06-01", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue

        for sym in chunk:
            try:
                t_str = f"{sym}.NS"
                raw_df = data[t_str].dropna() if len(chunk) > 1 and t_str in data else (data.dropna() if len(chunk) == 1 else None)
                if raw_df is None or len(raw_df) < 80:
                    continue

                clean_df = prepare_ohlcv(raw_df)
                if clean_df is None:
                    continue

                is_n750 = sym in n750_set

                if is_n750:
                    # 1. Swing 3 Locked in Nifty 750
                    t_sw3 = simulate_engine(clean_df, nifty_perf_map, mode="SWING3_LOCKED")
                    buckets["Swing 3 Locked - N750 (Spread<=8% • Score>=2.5)"].extend(t_sw3)

                    # 2. Run 2 in Nifty 750
                    t_r2_n750 = simulate_engine(clean_df, nifty_perf_map, mode="RUN2")
                    buckets["Swing 3 Run 2 - N750 (Spread<=13% • No Score Gate)"].extend(t_r2_n750)
                else:
                    # 3. Run 2 in Non-Nifty 750
                    t_r2_non750 = simulate_engine(clean_df, nifty_perf_map, mode="RUN2")
                    buckets["Swing 3 Run 2 - Non-N750 (Spread<=13% • No Score Gate)"].extend(t_r2_non750)
            except Exception:
                continue

    rows = []
    for k, t_list in buckets.items():
        if not t_list:
            rows.append({
                "Configuration": k, "Trades": 0, "Win Rate": "0.0%", "Profit Factor": 0.0,
                "Avg Gain": "0.0%", "Avg Loss": "0.0%", "Max Gain": "0.0%", "Avg Days": 0.0
            })
            continue
        df_res = pd.DataFrame(t_list)
        wins = df_res[df_res["pnl"] > 0]
        losses = df_res[df_res["pnl"] <= 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

        rows.append({
            "Configuration": k,
            "Trades": len(df_res),
            "Win Rate": f"{(len(wins) / len(df_res)) * 100:.2f}%",
            "Profit Factor": pf,
            "Avg Gain": f"+{wins['pnl'].mean() * 100:.2f}%" if not wins.empty else "+0.0%",
            "Avg Loss": f"{losses['pnl'].mean() * 100:.2f}%" if not losses.empty else "0.0%",
            "Max Gain": f"+{df_res['pnl'].max() * 100:.2f}%",
            "Avg Days": round(df_res["holding_days"].mean(), 1)
        })

    report_df = pd.DataFrame(rows)
    print("\n" + "=" * 105)
    print(report_df.to_string(index=False))
    print("=" * 105)

    os.makedirs("./backtest_results", exist_ok=True)
    report_df.to_csv("./backtest_results/latest_backtest.csv", index=False)
    with open("./backtest_results/latest_backtest.md", "w") as f:
        f.write("# Strategy Backtest Comparison (2021 – Present)\n\n")
        f.write(report_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
