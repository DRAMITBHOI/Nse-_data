import os
import glob
import json
import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, display
except ImportError:
    HTML, display = None, None

DATA_DIR = "data"


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
    for f in ["nifty.json", "nifty50.json", "NIFTY.json", "NIFTY50.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) > 50:
                    df = pd.DataFrame(clean)
                    df["perf_60d"] = df["close"].pct_change(60).fillna(0)
                    return {str(r["time"])[:10]: float(r["perf_60d"]) for _, r in df.iterrows()}
            except Exception:
                continue
    return {}


def load_nifty750_symbols():
    p = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def simulate_symbol(df, nifty_perf_map, is_n750, exit_strategy_mode):
    # exit_strategy_mode: 'EXIT_3', 'EXIT_2_2', 'COMBINATION_7'
    trades = []

    min_turnover = 10.0 if is_n750 else 5.0
    base_spread_min = 6.0 if is_n750 else 8.0
    base_spread_max = 12.0 if is_n750 else 18.0
    vdu_thresh = 0.45 if is_n750 else 0.42
    prox_max = 0.028 if is_n750 else 0.030
    vr_drop_min = 14.5 if is_n750 else 16.0
    vr_drop_max = 25.0 if is_n750 else 28.0
    vr_rc_min = 0.80 if is_n750 else 0.78
    vr_deliv_min = 1.15 if is_n750 else 1.20

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
    ema10s = df["ema_10"].values
    ema20s = df["ema_20"].values
    sma200s = df["sma_200"].values
    perf_60s = df["perf_60d"].values
    atr14s = df["atr_14"].values
    times = df["time"].values
    dist_counts = df["dist_count_12d"].values

    nifty_dates = sorted(nifty_perf_map.keys())

    in_trade = False
    entry_p = 0.0
    calc_sl = 0.0
    active_stop = 0.0
    trade_setup = ""
    hit_milestone = False
    t1_target_p = 0.0
    entry_d = ""

    # Exit 2.2 trackers
    peak_close = 0.0
    vdu_under_count = 0

    for i in range(40, len(df)):
        t = str(times[i])[:10]
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = vol_avgs[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        rc = range_closes[i]
        ema20 = ema20s[i]
        atr = atr14s[i]
        dist_cnt = dist_counts[i]

        if in_trade:
            exit_triggered = False
            exit_p = c
            exit_reason = ""

            peak_close = max(peak_close, c)
            peak_gain_pct = ((peak_close - entry_p) / entry_p) * 100.0

            # -----------------------------------------------------------------
            # NO GAP-DOWN OPEN CUTS: EXIT AT HARD SL OR TRAILING EMA / RULES
            # -----------------------------------------------------------------
            if exit_strategy_mode == "EXIT_2_2":
                # 1. Catastrophe Stop
                if c < calc_sl or l <= calc_sl:
                    exit_triggered = True
                    exit_p = calc_sl
                    exit_reason = "Catastrophe Stop (Base Structural Low)"

                # 2. Mega-Runner 15% Cushion Lock (Peak Unrealized Gain >= +50%)
                elif peak_gain_pct >= 50.0:
                    trailing_stop = max(ema20, peak_close * 0.85)
                    active_stop = max(active_stop, trailing_stop)
                    if c < trailing_stop:
                        exit_triggered = True
                        exit_p = trailing_stop
                        exit_reason = "Mega-Runner 15% Cushion Lock (Peak >= +50%)"

                # 3. Climactic Churn Exit (Top Harvesting)
                elif (dv >= 2.0 * dv_avg) and (rc <= 0.35) and (peak_gain_pct >= 20.0):
                    exit_triggered = True
                    exit_p = c
                    exit_reason = "Climactic Churn Top Harvest (DV >= 2.0x & RC <= 0.35)"

                # 4. Persistent Distribution Cluster (Strict >= 3D cutoff)
                elif dist_cnt >= 3 and c < ema20:
                    exit_triggered = True
                    exit_p = c
                    exit_reason = "Persistent Distribution Cluster (>= 3D in 12s & Close < 20 EMA)"

                # 5. 20 EMA Breach & 5-Session VDU Reclaim Engine (Gain < +50%)
                elif c < ema20:
                    is_heavy_vol = (v >= 1.25 * v_avg) or (dv >= 1.25 * dv_avg)
                    if is_heavy_vol:
                        exit_triggered = True
                        exit_p = ema20
                        exit_reason = "Heavy Volume 20 EMA Breakdown (Vol >= 1.25x)"
                    else:
                        atr_buffer_floor = ema20 - (1.5 * atr)
                        pct_buffer_floor = ema20 * 0.95
                        dyn_leeway_stop = min(pct_buffer_floor, atr_buffer_floor)
                        guaranteed_be_floor = entry_p * 1.015
                        active_vdu_stop = max(guaranteed_be_floor, dyn_leeway_stop)

                        vdu_under_count += 1
                        if c < active_vdu_stop:
                            exit_triggered = True
                            exit_p = c
                            exit_reason = "Active VDU Stop Violated (< Max(BE+1.5%, Leeway))"
                        elif vdu_under_count > 5:
                            exit_triggered = True
                            exit_p = c
                            exit_reason = "Failed to Reclaim 20 EMA within 5 Sessions"
                else:
                    vdu_under_count = 0
                    if c >= entry_p * 1.05:
                        active_stop = max(active_stop, entry_p * 1.015)

            elif exit_strategy_mode == "COMBINATION_7":
                if not hit_milestone and l <= active_stop:
                    exit_triggered = True
                    exit_p = active_stop
                    exit_reason = "Hard SL Hit (Base Low - 0.5%)"
                else:
                    if not hit_milestone and h >= t1_target_p:
                        hit_milestone = True
                        active_stop = max(active_stop, round(entry_p * 1.005, 2))

                    if hit_milestone:
                        active_stop = max(active_stop, ema20)
                        if c < ema20:
                            exit_triggered = True
                            exit_p = c
                            exit_reason = "Daily Close < 20 EMA"

            elif exit_strategy_mode == "EXIT_3":
                # EMA 20 Takeover (once EMA 20 > Hard SL) + BE+0.5% at +1.5R + 20 EMA Trail
                if not hit_milestone and h >= t1_target_p:
                    hit_milestone = True
                    active_stop = max(active_stop, round(entry_p * 1.005, 2))

                if ema20 > calc_sl:
                    if hit_milestone:
                        effective_stop = max(round(entry_p * 1.005, 2), ema20)
                    else:
                        effective_stop = ema20
                    active_stop = max(active_stop, effective_stop)

                if hit_milestone:
                    if c < active_stop:
                        exit_triggered = True
                        exit_p = c
                        exit_reason = "Close < Trailing Stop (Max(BE+0.5%, 20 EMA))"
                else:
                    if l <= active_stop or c < active_stop:
                        exit_triggered = True
                        exit_p = active_stop
                        exit_reason = "Stop Loss Hit (Pre-Milestone EMA20/Hard SL)"

            if exit_triggered:
                hold_days = (pd.to_datetime(t) - pd.to_datetime(entry_d)).days
                risk_pct = round(((entry_p - calc_sl) / entry_p) * 100.0, 2)

                if exit_strategy_mode == "COMBINATION_7":
                    t1_gain = ((t1_target_p - entry_p) / entry_p) * 100.0 if hit_milestone else 0.0
                    runner_gain = ((exit_p - entry_p) / entry_p) * 100.0
                    final_gain = round((0.20 * t1_gain) + (0.80 * runner_gain), 2) if hit_milestone else round(runner_gain, 2)
                else:
                    final_gain = round(((exit_p - entry_p) / entry_p) * 100.0, 2)

                r_mult = round(final_gain / risk_pct, 2) if risk_pct > 0 else 0.0

                trades.append({
                    "Symbol": df.get("symbol", "SYM"),
                    "Setup": trade_setup,
                    "Entry_Date": entry_d,
                    "Exit_Date": t,
                    "Gain_%": final_gain,
                    "R_Mult": r_mult,
                    "Hold_Days": hold_days,
                    "Exit_Reason": exit_reason
                })
                in_trade = False
            continue

        # ---------------------------------------------------------------------
        # ENTRY SCANNERS (NO GAP-UP FILTER: ENTER AT NR CANDLE CEILING)
        # ---------------------------------------------------------------------
        if to_50d[i] < min_turnover:
            continue

        sma200 = sma200s[i]
        is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
        n_perf = nifty_perf_map.get(t)
        if n_perf is None:
            p_dates = [d for d in nifty_dates if d <= t]
            n_perf = nifty_perf_map[p_dates[-1]] if p_dates else 0.001
        is_rs = bool(perf_60s[i] > n_perf)

        if not (is_stage2 and is_rs):
            continue

        entry_triggered = False
        c_entry = 0.0
        c_sl = 0.0
        c_setup = ""

        # Engine 1: V-Reversal Setup
        recent_20_h = highs[max(0, i - 20) : i].max()
        if recent_20_h > 0:
            drop_depth = ((recent_20_h - l) / recent_20_h) * 100.0
            if vr_drop_min <= drop_depth <= vr_drop_max:
                prior_3d_h = highs[max(0, i - 3) : i].max()
                prior_3d_l = lows[max(0, i - 3) : i].min()
                c_reclaim = (c >= prior_3d_h) and (c > closes[i - 1])
                c_deliv = (dv >= vr_deliv_min * dv_avg) or (deliv_pcts[i] >= 1.15 * deliv_pct_avgs[i])
                c_candle = rc >= vr_rc_min

                if c_reclaim and c_deliv and c_candle:
                    v_sl = round(prior_3d_l * 0.995, 2)
                    v_risk = ((prior_3d_h - v_sl) / prior_3d_h) * 100.0
                    if 0.5 <= v_risk <= 5.0:
                        entry_triggered = True
                        c_entry = round(prior_3d_h + 0.05, 2)
                        c_sl = v_sl
                        c_setup = "V-Reversal"

        # Engine 2: Launchpad Cheat Entry
        if not entry_triggered:
            for s_len in range(10, 19):
                s_h = highs[max(0, i - s_len) : i].max()
                s_l = lows[max(0, i - s_len) : i].min()
                spread = ((s_h - s_l) / s_l) * 100.0 if s_l > 0 else 99.0
                if s_l > 0 and (base_spread_min <= spread <= base_spread_max):
                    base_vols = volumes[max(0, i - 15) : i]
                    lowest_5d = pd.Series(base_vols).rolling(5, min_periods=1).mean().min() if len(base_vols) > 0 else v_avg
                    vdu_ok = (lowest_5d / v_avg <= vdu_thresh) if v_avg > 0 else False

                    prev_h = highs[i - 1]
                    prev_l = lows[i - 1]
                    prev_atr = atr14s[i - 1]
                    cheat_ok = (prev_atr > 0) and ((prev_h - prev_l) / prev_atr <= 0.65)
                    prox = (s_h - closes[i - 1]) / s_h if s_h > 0 else 1.0
                    prox_ok = 0.005 <= prox <= prox_max
                    above_10ema = closes[i - 1] >= ema10s[i - 1]

                    if vdu_ok and cheat_ok and prox_ok and above_10ema:
                        cheat_e = round(prev_h + 0.05, 2)
                        cheat_s = round(prev_l * 0.995, 2)

                        if h < cheat_e:
                            continue
                        effective_entry = cheat_e

                        comp_risk = ((effective_entry - cheat_s) / effective_entry) * 100.0
                        if 0.5 <= comp_risk <= 4.5:
                            entry_triggered = True
                            c_entry = effective_entry
                            c_sl = cheat_s
                            c_setup = "Launchpad Cheat"
                            break

        if entry_triggered:
            deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
            rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
            conv_score = round(deliv_surge * max(0.1, rs_ratio), 3)

            if is_n750 and c_setup == "Launchpad Cheat" and conv_score < 2.30:
                continue

            r_dist = c_entry - c_sl
            risk_pct = (r_dist / c_entry) * 100.0

            in_trade = True
            entry_p = c_entry
            calc_sl = c_sl
            active_stop = c_sl
            trade_setup = c_setup
            hit_milestone = False
            entry_d = t
            t1_pct = min(1.5 * risk_pct, 6.0)
            t1_target_p = round(entry_p * (1.0 + t1_pct / 100.0), 2)
            peak_close = entry_p
            vdu_under_count = 0

    return trades


def run_nogap_3_models():
    n750_set = load_nifty750_symbols()
    nifty_perf_map = load_nifty_benchmark()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing_ultra750_results.json",
        "swing_ultraN750_results.json", "live_intraday_breakouts.json",
        "live_intraday_breakouts_v2.json", "accuracy_audit_log.json", "my_trades.json",
        "gap_down_audit_results.json", "gap_down_audit_full_results.json",
        "gap_up_empirical_audit_results.json", "active_watchlist.json",
        "combination7_master_audited.json", "quick_backtest_summary.json",
        "comparison_3_exit_models_results.json", "comb7_nogap_results.json",
        "comparison_3_models_nogap_results.json"
    }

    target_files = [p for p in sorted(json_files) if os.path.basename(p) not in excluded]
    if not target_files:
        print("❌ No stock data files found in 'data/' directory.")
        return

    print(f"⏳ Pre-loading {len(target_files)} NSE stocks (No-Gap Baseline Comparison)...")
    prepared_stocks = []
    for path in target_files:
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if len(clean) < 60:
                continue

            df = pd.DataFrame(clean)
            df["symbol"] = sym
            df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
            df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
            df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
            df["perf_60d"] = df["close"].pct_change(60).fillna(0)

            prev_c = df["close"].shift(1)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()
            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            is_dist = (
                (df["close"] < df["close"].shift(1))
                & ((df["delivery_vol"] >= 1.35 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * df["deliv_pct_50d"]))
                & (df["range_closeness"] <= 0.45)
            ).astype(int)
            df["dist_count_12d"] = is_dist.rolling(12, min_periods=1).sum()

            is_n750 = sym in n750_set if n750_set else True
            prepared_stocks.append((df, is_n750))
        except Exception:
            continue

    print(f"✅ Pre-loaded {len(prepared_stocks)} clean stock histories.")

    models = [
        ("EXIT_3", "1. Exit 3: EMA 20 Takeover (Crosses SL) + BE+0.5% @ 1.5R + 20 EMA Trail"),
        ("EXIT_2_2", "2. Exit 2.2: Calibrated Institutional Harvest (Churn + 3D Dist + VDU Reclaim)"),
        ("COMBINATION_7", "3. Combination 7: 20% Harvest @ 1.5R + 80% 20 EMA Runner")
    ]

    all_model_results = []

    def compute_metrics(sub_df, label):
        n = len(sub_df)
        if n == 0:
            return {
                "Segment": label, "Trades": "0", "Win Rate": "0.0%", "Profit Factor": "0.00",
                "Avg Win": "0.00%", "Avg Loss": "0.00%", "Max Gain": "0.00%",
                "Avg Hold": "0.0 d", "Total R": "0.0R", "Exp (R/t)": "0.00R"
            }
        # Any gain > 0 is a winning trade
        w = sub_df[sub_df["Gain_%"] > 0]
        l = sub_df[sub_df["Gain_%"] <= 0]
        win_rate = (len(w) / n) * 100.0
        gross_win = w["Gain_%"].sum()
        gross_loss = abs(l["Gain_%"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_gain = round(w["Gain_%"].mean(), 2) if len(w) > 0 else 0.0
        avg_loss = round(l["Gain_%"].mean(), 2) if len(l) > 0 else 0.0
        max_gain = round(sub_df["Gain_%"].max(), 2) if n > 0 else 0.0
        avg_hold = round(sub_df["Hold_Days"].mean(), 1) if n > 0 else 0.0
        tot_r = round(sub_df["R_Mult"].sum(), 1)
        exp_r = round(tot_r / n, 2) if n > 0 else 0.0

        return {
            "Segment": label,
            "Trades": f"{n:,}",
            "Win Rate": f"{win_rate:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Avg Win": f"+{avg_gain:.2f}%",
            "Avg Loss": f"{avg_loss:.2f}%",
            "Max Gain": f"+{max_gain:.2f}%",
            "Avg Hold": f"{avg_hold} d",
            "Total R": f"+{tot_r:,.1f}R",
            "Exp (R/t)": f"+{exp_r:.2f}R"
        }

    for mode_id, mode_title in models:
        print(f"🚀 Running No-Gap Backtest for: {mode_title}...")
        model_trades = []
        for df, is_n750 in prepared_stocks:
            univ_tag = "Nifty 750" if is_n750 else "Non-N750"
            t_list = simulate_symbol(df, nifty_perf_map, is_n750, mode_id)
            for t in t_list:
                t["Universe"] = univ_tag
                model_trades.append(t)

        df_res = pd.DataFrame(model_trades)
        if len(df_res) == 0:
            continue

        seg1 = df_res[(df_res["Universe"] == "Nifty 750") & (df_res["Setup"] == "Launchpad Cheat")]
        seg2 = df_res[(df_res["Universe"] == "Nifty 750") & (df_res["Setup"] == "V-Reversal")]
        seg3 = df_res[(df_res["Universe"] == "Non-N750") & (df_res["Setup"] == "Launchpad Cheat")]
        seg4 = df_res[(df_res["Universe"] == "Non-N750") & (df_res["Setup"] == "V-Reversal")]

        rows = [
            compute_metrics(seg1, "1. Nifty 750 — Launchpad Cheat"),
            compute_metrics(seg2, "2. Nifty 750 — V-Reversal"),
            compute_metrics(seg3, "3. Non-Nifty 750 — Launchpad Cheat"),
            compute_metrics(seg4, "4. Non-Nifty 750 — V-Reversal"),
            compute_metrics(df_res, "⭐ Overall Portfolio (Combined)")
        ]

        all_model_results.append((mode_title, rows))

        print("\n" + "=" * 95)
        print(f"📊 {mode_title}")
        print("=" * 95)
        print(pd.DataFrame(rows).to_string(index=False))
        print("=" * 95)

    with open("comparison_3_models_nogap_results.json", "w", encoding="utf-8") as fp:
        json.dump(all_model_results, fp, indent=2)

    if display and HTML:
        html_out = ""
        for mode_title, rows in all_model_results:
            table_rows = ""
            for r in rows:
                is_overall = "Overall" in r["Segment"]
                bg = "#f1f8ff" if is_overall else "#ffffff"
                fw = "bold" if is_overall else "600"
                col = "#0366d6" if is_overall else "#24292e"

                table_rows += f"""
                <tr style="background: {bg}; border-bottom: 1px solid #e1e4e8; color: {col}; font-weight: {fw};">
                    <td style="padding: 9px 12px; font-size: 13px;">{r['Segment']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px;">{r['Trades']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px;">{r['Win Rate']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #28a745;">{r['Profit Factor']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #28a745;">{r['Avg Win']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #cb2431;">{r['Avg Loss']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #005cc5;">{r['Max Gain']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px;">{r['Avg Hold']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #0366d6;">{r['Total R']}</td>
                    <td style="padding: 9px 12px; text-align: center; font-size: 13px; color: #0366d6;">{r['Exp (R/t)']}</td>
                </tr>
                """

            html_out += f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #ffffff; color: #24292e; padding: 20px; border: 1px solid #d1d5da; border-radius: 8px; max-width: 1080px; margin: 15px 0;">
                <h3 style="margin-top: 0; color: #0366d6; border-bottom: 2px solid #eaecef; padding-bottom: 8px; font-size: 16px;">
                    {mode_title}
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 12.5px;">
                    <thead>
                        <tr style="background: #f6f8fa; border-bottom: 2px solid #d1d5da; color: #24292e;">
                            <th style="padding: 9px 12px; text-align: left;">Segment</th>
                            <th style="padding: 9px 12px; text-align: center;">Trades</th>
                            <th style="padding: 9px 12px; text-align: center;">Win Rate</th>
                            <th style="padding: 9px 12px; text-align: center;">Profit Factor</th>
                            <th style="padding: 9px 12px; text-align: center;">Avg Win</th>
                            <th style="padding: 9px 12px; text-align: center;">Avg Loss</th>
                            <th style="padding: 9px 12px; text-align: center;">Max Gain</th>
                            <th style="padding: 9px 12px; text-align: center;">Avg Hold</th>
                            <th style="padding: 9px 12px; text-align: center;">Total R</th>
                            <th style="padding: 9px 12px; text-align: center;">Exp (R/t)</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
            """
        display(HTML(html_out))

if __name__ == "__main__":
    run_nogap_3_models()
