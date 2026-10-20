import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
START_DATE = "2021-01-01"
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


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def run_segmented_backtest():
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing3_results.json",
        "swing3_run2_results.json", "swing3_run2_global_results.json",
        "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
        "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
        "swing_ultra750_results.json", "swing_ultraN750_results.json",
        "live_intraday_breakouts.json", "live_intraday_breakouts_v2.json",
        "accuracy_audit_log.json"
    }

    target_files = [f for f in json_files if os.path.basename(f) not in excluded]

    trades = []
    print(f"🚀 Starting Historical Exit 2.2 Segmented Backtest on {len(target_files)} symbols...")

    for idx, path in enumerate(sorted(target_files)):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        is_nifty750 = sym in nifty750_set
        universe_tag = "NIFTY 750" if is_nifty750 else "NON-NIFTY 750"

        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if len(clean) < 60:
                continue

            df = pd.DataFrame(clean)
            df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
            df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
            df["perf_60d"] = df["close"].pct_change(60).fillna(0)

            # Demat OBV for N750 Launchpad
            price_diff = df["close"].diff()
            direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
            df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
            df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()

            # ATR 14 calculation for VDU Buffer
            prev_close = df["close"].shift(1)
            tr1 = df["high"] - df["low"]
            tr2 = (df["high"] - prev_close).abs()
            tr3 = (df["low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()

            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            # Distribution Day Definition
            is_dist_day = (
                (df["close"] < df["close"].shift(1)) &
                ((df["delivery_vol"] >= 1.35 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * df["deliv_pct_50d"])) &
                (df["range_closeness"] <= 0.45)
            ).astype(int)
            df["dist_count_12d"] = is_dist_day.rolling(12, min_periods=1).sum()

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
            atr14s = df["atr_14"].values
            dist_12ds = df["dist_count_12d"].values
            times = df["time"].values

            # Simulation State Variables
            in_trade = False
            entry_p = 0.0
            calc_sl = 0.0
            entry_d = ""
            entry_idx = 0
            trade_setup = ""
            peak_gain_pct = 0.0
            peak_close = 0.0
            vdu_under_count = 0

            for i in range(40, len(df)):
                t = times[i]
                c = closes[i]
                h = highs[i]
                l = lows[i]
                v = volumes[i]
                v_avg = vol_avgs[i]
                dv = deliv_vols[i]
                dv_avg = deliv_avgs[i]
                rc = range_closes[i]
                ema = ema20s[i]
                atr = atr14s[i]

                # ==============================================================
                # EXIT 2.2 ENGINE (ACTIVE TRADE MONITORING)
                # ==============================================================
                if in_trade:
                    peak_close = max(peak_close, c)
                    current_gain_pct = ((c - entry_p) / entry_p) * 100.0
                    peak_gain_pct = max(peak_gain_pct, current_gain_pct)

                    exit_triggered = False
                    exit_price = c
                    exit_reason = ""

                    # 1. CATASTROPHE STOP (UNREALIZED GAIN < 0%)
                    if c < calc_sl:
                        exit_triggered = True
                        exit_price = calc_sl
                        exit_reason = "1. Catastrophe Stop"

                    # 2. CLIMACTIC CHURN EXIT (TOP HARVESTING)
                    elif (dv >= 2.0 * dv_avg) and (rc <= 0.35) and (peak_gain_pct >= 20.0):
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "2. Climactic Churn Harvest"

                    # 3. PERSISTENT DISTRIBUTION CLUSTER EXIT (>= 3D & Close < 20 EMA)
                    elif (dist_12ds[i] >= 3) and (c < ema):
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "3. Distribution Cluster"

                    # 4. MEGA-RUNNER 15% CUSHION LOCK (PEAK GAIN >= +50%)
                    elif peak_gain_pct >= 50.0:
                        trailing_cushion_stop = max(ema, peak_close * 0.85)
                        if c < trailing_cushion_stop:
                            exit_triggered = True
                            exit_price = trailing_cushion_stop
                            exit_reason = "4. Mega-Runner Cushion Lock"

                    # 5. 20 EMA BREACH & 5-SESSION VDU RECLAIM ENGINE (PEAK GAIN < +50%)
                    elif c < ema:
                        is_heavy_breakdown = (v >= 1.25 * v_avg) or (dv >= 1.25 * dv_avg)
                        if is_heavy_breakdown:
                            exit_triggered = True
                            exit_price = ema
                            exit_reason = "5A. Heavy Vol 20 EMA Breakdown"
                        else:
                            # 5B. VDU / Normal Volume Breach
                            vdu_under_count += 1
                            atr_floor = ema - (1.5 * atr)
                            pct_floor = ema * 0.95
                            dynamic_leeway = min(pct_floor, atr_floor)
                            guaranteed_be = entry_p * 1.015
                            active_vdu_stop = max(guaranteed_be, dynamic_leeway)

                            if c < active_vdu_stop:
                                exit_triggered = True
                                exit_price = active_vdu_stop
                                exit_reason = "5B. Dynamic VDU Buffer Floor Breach"
                            elif vdu_under_count > 5:
                                exit_triggered = True
                                exit_price = c
                                exit_reason = "5B. VDU 5-Session Reclaim Failure"

                    # Reset VDU counter if closed above/on 20 EMA
                    if not exit_triggered and c >= ema:
                        vdu_under_count = 0

                    if exit_triggered:
                        pnl_pct = round(((exit_price - entry_p) / entry_p) * 100.0, 2)
                        hold_days = i - entry_idx
                        trades.append({
                            "symbol": sym,
                            "universe": universe_tag,
                            "setup": trade_setup,
                            "segment": f"{universe_tag} | {trade_setup}",
                            "entry_date": entry_d,
                            "exit_date": t,
                            "entry_price": round(entry_p, 2),
                            "exit_price": round(exit_price, 2),
                            "initial_sl": round(calc_sl, 2),
                            "pnl_%": pnl_pct,
                            "peak_gain_%": round(peak_gain_pct, 2),
                            "holding_days": hold_days,
                            "exit_reason": exit_reason
                        })
                        in_trade = False
                    continue

                # ==============================================================
                # ENTRY SCANNER (BACKTESTING FROM 2021 ONWARD)
                # ==============================================================
                if t < START_DATE:
                    continue

                if to_50d[i] < MIN_TURNOVER_CR:
                    continue

                sma200 = sma200s[i]
                is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False

                n_perf = nifty_perf_map.get(t)
                if n_perf is None:
                    prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
                    n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
                is_rs = bool(perf_60s[i] > n_perf)

                if not (is_stage2 and is_rs):
                    continue

                entry_triggered = False
                c_entry = 0.0
                c_sl = 0.0
                c_setup = ""

                # --- ENGINE 1: V-REVERSAL ---
                r20_h = highs[i - 20:i].max()
                if (r20_h - l) / r20_h >= 0.18:
                    prior_3d_high = highs[i - 4:i].max()
                    c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
                    c_vol_rev = v >= (1.3 * v_avg)
                    c_deliv_rev = (dv >= 1.25 * dv_avg) or (deliv_pcts[i] >= 1.15 * deliv_pct_avgs[i])
                    c_candle_rev = rc >= 0.55

                    if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                        entry_triggered = True
                        c_entry = round(prior_3d_high, 2)
                        c_sl = round(lows[i - 5:i].min() * 0.995, 2)
                        c_setup = "V-REVERSAL"

                # --- ENGINE 2: LAUNCHPAD BREAKOUT ---
                if not entry_triggered:
                    valid_shelf = False
                    lp_high = 0.0
                    lp_low = 0.0
                    max_spread = 0.08 if is_nifty750 else 0.13

                    for shelf_len in range(10, 19):
                        s_h = highs[i - shelf_len:i].max()
                        s_l = lows[i - shelf_len:i].min()
                        if s_l > 0 and ((s_h - s_l) / s_l) <= max_spread:
                            valid_shelf = True
                            lp_high = round(float(s_h), 2)
                            lp_low = round(float(s_l), 2)
                            break

                    if valid_shelf:
                        req_cumul_ratio = 1.50 if is_nifty750 else 1.15
                        base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                        base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                        base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else req_cumul_ratio

                        c_bo = (c >= lp_high) and (h >= lp_high)
                        c_vol_bo = v >= (1.4 * v_avg)
                        c_deliv_bo = (dv >= 1.25 * dv_avg) or (deliv_pcts[i] >= 1.25 * deliv_pct_avgs[i])
                        c_candle_bo = rc >= 0.65
                        obv_condition = (demat_obvs[i] >= demat_obv_maxes[i - 1]) if (is_nifty750 and i > 0) else True

                        if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= req_cumul_ratio) and obv_condition:
                            entry_triggered = True
                            c_entry = lp_high
                            c_sl = round(min(l, lp_low), 2)
                            c_setup = "LAUNCHPAD"

                # Stop-loss Gate Check
                if entry_triggered:
                    r_dist = c_entry - c_sl
                    if r_dist > 0.05 and (r_dist / c_entry) <= 0.12:
                        deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                        rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
                        conv_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                        # Nifty 750 Launchpad requires conviction >= 2.5
                        if is_nifty750 and c_setup == "LAUNCHPAD" and conv_score < 2.5:
                            continue

                        in_trade = True
                        entry_p = c_entry
                        calc_sl = c_sl
                        entry_d = t
                        entry_idx = i
                        trade_setup = c_setup
                        peak_gain_pct = 0.0
                        peak_close = c
                        vdu_under_count = 0
        except Exception:
            continue

    # ==========================================================================
    # AGGREGATE RESULTS & PERFORMANCE METRICS
    # ==========================================================================
    if not trades:
        print("❌ No historical trades triggered.")
        return

    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("data/segmented_backtest_trades_exit2_2.csv", index=False)

    print("\n" + "=" * 120)
    print("🏆 HISTORICAL MULTI-UNIVERSE BACKTEST PERFORMANCE: EXIT 2.2 ENGINE (2021 - PRESENT)")
    print("=" * 120)

    summary_rows = []
    segments = [
        ("NIFTY 750", "LAUNCHPAD"),
        ("NIFTY 750", "V-REVERSAL"),
        ("NON-NIFTY 750", "LAUNCHPAD"),
        ("NON-NIFTY 750", "V-REVERSAL"),
        ("ALL", "COMBINED PORTFOLIO")
    ]

    for univ, setup in segments:
        if univ == "ALL":
            sub = df_trades
            label = "TOTAL COMBINED"
        else:
            sub = df_trades[(df_trades["universe"] == univ) & (df_trades["setup"] == setup)]
            label = f"{univ} | {setup}"

        count = len(sub)
        if count == 0:
            continue

        wins = sub[sub["pnl_%"] > 0]
        losses = sub[sub["pnl_%"] <= 0]
        win_rate = round((len(wins) / count) * 100.0, 2)
        avg_gain = round(wins["pnl_%"].mean(), 2) if len(wins) > 0 else 0.0
        avg_loss = round(losses["pnl_%"].mean(), 2) if len(losses) > 0 else 0.0
        tot_win = wins["pnl_%"].sum()
        tot_loss = abs(losses["pnl_%"].sum())
        pf = round(tot_win / tot_loss, 2) if tot_loss > 0 else 99.99
        max_gain = round(sub["pnl_%"].max(), 2)
        avg_hold = round(sub["holding_days"].mean(), 1)

        summary_rows.append({
            "Segment": label,
            "Trade Count": count,
            "Win Rate %": f"{win_rate}%",
            "Avg Gain %": f"+{avg_gain}%",
            "Avg Loss %": f"{avg_loss}%",
            "Profit Factor": pf,
            "Max Gain %": f"+{max_gain}%",
            "Avg Hold (Days)": avg_hold
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 120)

    # Breakdown by Exit Reason
    print("\n📊 EXIT TRIGGER REASON DISTRIBUTION:")
    reason_counts = df_trades["exit_reason"].value_counts().reset_index()
    reason_counts.columns = ["Exit Reason", "Trades Triggered"]
    print(reason_counts.to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    run_segmented_backtest()
