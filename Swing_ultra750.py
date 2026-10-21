import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
RESULTS_OUT = os.path.join(DATA_DIR, "swing_ultra750_results.json")
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


def run_swing_ultra750():
    nifty_perf_map = load_nifty_benchmark()
    nifty_dates = sorted(nifty_perf_map.keys())
    n750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing_ultra750_results.json",
        "swing_ultraN750_results.json", "live_intraday_breakouts.json",
        "live_intraday_breakouts_v2.json", "accuracy_audit_log.json", "my_trades.json"
    }

    fresh_triggers = []
    watchlist = []
    active_trailing_positions = []

    print(f"🚀 Running Swing Ultra Nifty 750 Engine across {len(json_files)} symbols...")

    for path in sorted(json_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym in excluded or (n750_set and sym not in n750_set):
            continue

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

            # Demat OBV
            price_diff = df["close"].diff()
            direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
            df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
            df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()

            # ATR 14
            prev_close = df["close"].shift(1)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()

            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            # Distribution Days (Rolling 12)
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

            # Simulation state to track active trailing positions
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
                t = str(times[i])[:10]
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
                # EXIT 2.2 ENGINE (ACTIVE TRADE TRAILING)
                # ==============================================================
                if in_trade:
                    peak_close = max(peak_close, c)
                    current_gain_pct = ((c - entry_p) / entry_p) * 100.0
                    peak_gain_pct = max(peak_gain_pct, current_gain_pct)

                    exit_triggered = False
                    exit_reason = ""

                    # 1. Catastrophe Stop
                    if c < calc_sl:
                        exit_triggered = True
                        exit_reason = "1. Catastrophe Stop"

                    # 2. Climactic Churn Harvest
                    elif (dv >= 2.0 * dv_avg) and (rc <= 0.35) and (peak_gain_pct >= 20.0):
                        exit_triggered = True
                        exit_reason = "2. Climactic Churn Harvest"

                    # 3. Distribution Cluster Breakdown
                    elif (dist_12ds[i] >= 3) and (c < ema):
                        exit_triggered = True
                        exit_reason = "3. Distribution Cluster"

                    # 4. Mega-Runner 15% Cushion Lock
                    elif peak_gain_pct >= 50.0:
                        trailing_cushion = max(ema, peak_close * 0.85)
                        if c < trailing_cushion:
                            exit_triggered = True
                            exit_reason = "4. Mega-Runner Cushion Lock"

                    # 5. 20 EMA Breach Engine
                    elif c < ema:
                        is_heavy = (v >= 1.25 * v_avg) or (dv >= 1.25 * dv_avg)
                        if is_heavy:
                            exit_triggered = True
                            exit_reason = "5A. Heavy Vol Breakdown"
                        else:
                            vdu_under_count += 1
                            atr_floor = ema - (1.5 * atr)
                            pct_floor = ema * 0.95
                            dynamic_leeway = min(pct_floor, atr_floor)
                            guaranteed_be = entry_p * 1.015
                            active_vdu_stop = max(guaranteed_be, dynamic_leeway)

                            if c < active_vdu_stop:
                                exit_triggered = True
                                exit_reason = "5B. Dynamic VDU Floor Breach"
                            elif vdu_under_count > 5:
                                exit_triggered = True
                                exit_reason = "5B. 5-Session Reclaim Failure"

                    if not exit_triggered and c >= ema:
                        vdu_under_count = 0

                    # If this is the current (latest) bar and trade is still open
                    if i == len(df) - 1 and not exit_triggered:
                        holding_days = i - entry_idx
                        active_trailing_positions.append({
                            "symbol": sym,
                            "entry_date": entry_d,
                            "entry_price": round(float(entry_p), 2),
                            "current_close": round(float(c), 2),
                            "trailing_stop_20ema": round(float(ema), 2),      # Exact key needed by dashboard
                            "trailing_stop": round(float(ema), 2),
                            "hard_sl": round(float(calc_sl), 2),
                            "pnl_%": round(float(current_gain_pct), 2),
                            "days": int(holding_days),
                            "holding_days": int(holding_days),
                            "status": "HOLD"
                        })

                    if exit_triggered:
                        in_trade = False
                    continue

                # ==============================================================
                # ENTRY SCANNER (ONLY EVALUATE CANDIDATES FOR TODAY / LATEST BAR)
                # ==============================================================
                if to_50d[i] < MIN_TURNOVER_CR:
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

                # Engine 1: V-Reversal
                recent_20_h = highs[max(0, i - 20):i].max()
                if (recent_20_h - l) / recent_20_h >= 0.18:
                    prior_3d_h = highs[max(0, i - 4):i].max()
                    c_reclaim = (c >= prior_3d_h) and (c > closes[i - 1])
                    c_vol_rev = v >= (1.3 * v_avg)
                    c_deliv_rev = (dv >= 1.25 * dv_avg) or (deliv_pcts[i] >= 1.15 * deliv_pct_avgs[i])
                    c_candle_rev = rc >= 0.55

                    if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                        entry_triggered = True
                        c_entry = round(prior_3d_h, 2)
                        c_sl = round(lows[max(0, i - 5):i].min() * 0.995, 2)
                        c_setup = "V-REVERSAL"

                # Engine 2: Launchpad
                shelf_found = False
                lp_h = 0.0
                lp_l = 0.0
                found_spread = 6.0

                for shelf_len in range(10, 19):
                    s_h = highs[max(0, i - shelf_len):i].max()
                    s_l = lows[max(0, i - shelf_len):i].min()
                    spread = ((s_h - s_l) / s_l) * 100.0 if s_l > 0 else 99.0
                    if s_l > 0 and spread <= 8.0:
                        shelf_found = True
                        lp_h = round(float(s_h), 2)
                        lp_l = round(float(s_l), 2)
                        found_spread = round(spread, 2)
                        break

                if not entry_triggered and shelf_found:
                    base_up_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] >= closes[k - 1])
                    base_down_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] < closes[k - 1])
                    cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.50

                    c_bo = (c >= lp_h) and (h >= lp_h)
                    c_vol_bo = v >= (1.4 * v_avg)
                    c_deliv_bo = (dv >= 1.25 * dv_avg) or (deliv_pcts[i] >= 1.25 * deliv_pct_avgs[i])
                    c_candle_bo = rc >= 0.65
                    obv_condition = (demat_obvs[i] >= demat_obv_maxes[i - 1]) if i > 0 else True

                    if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (cumul_ratio >= 1.50) and obv_condition:
                        entry_triggered = True
                        c_entry = lp_h
                        c_sl = round(min(l, lp_l), 2)
                        c_setup = "LAUNCHPAD COIL"

                # Sizing & Gate
                if entry_triggered:
                    r_dist = c_entry - c_sl
                    if r_dist > 0.05 and (r_dist / c_entry) <= 0.12:
                        deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
                        rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
                        conv_score = round(deliv_surge * max(0.1, rs_ratio), 3)

                        if c_setup == "LAUNCHPAD COIL" and conv_score < 2.5:
                            pass
                        else:
                            in_trade = True
                            entry_p = c_entry
                            calc_sl = c_sl
                            entry_d = t
                            entry_idx = i
                            trade_setup = c_setup
                            peak_gain_pct = 0.0
                            peak_close = c
                            vdu_under_count = 0

                            # If this fired on the latest bar, log as Fresh Trigger
                            if i == len(df) - 1:
                                ext_pct = round(((c - c_entry) / c_entry) * 100.0, 2)
                                risk_pct = round((r_dist / c_entry) * 100.0, 2)
                                action_tag = "BUY AT OPEN" if ext_pct <= 2.0 else "LIMIT RETEST"
                                fresh_triggers.append({
                                    "symbol": sym,
                                    "type": c_setup,
                                    "pivot_ceiling": f"₹{c_entry:.2f}",
                                    "close": f"₹{c:.2f}",
                                    "extended_%": f"{ext_pct:+.1f}%",
                                    "stop_loss": f"₹{c_sl:.2f}",
                                    "risk_%": f"{risk_pct:.2f}%",
                                    "conviction_score": conv_score,
                                    "turnover": f"₹{to_50d[i]:.1f} Cr",
                                    "action": action_tag
                                })

                # If on latest bar and not triggered, evaluate Pre-Breakout Watchlist
                if i == len(df) - 1 and not in_trade and shelf_found:
                    dist_to_p = round(((lp_h - c) / c) * 100.0, 2)
                    if 0.0 <= dist_to_p <= 3.0:
                        deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
                        rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
                        conv_score = round(deliv_surge * max(0.1, rs_ratio), 3)
                        watchlist.append({
                            "symbol": sym,
                            "setup_class": "LAUNCHPAD COIL",
                            "close": f"₹{c:.2f}",
                            "pivot_target": f"₹{lp_h:.2f}",
                            "dist_to_pivot": f"{dist_to_p:.2f}%",
                            "spread_%": f"{found_spread:.2f}%",
                            "conviction_score": conv_score,
                            "turnover": f"₹{to_50d[i]:.1f} Cr"
                        })
        except Exception:
            continue

    fresh_triggers.sort(key=lambda x: x["conviction_score"], reverse=True)
    watchlist.sort(key=lambda x: x["conviction_score"], reverse=True)
    active_trailing_positions.sort(key=lambda x: x["pnl_%"], reverse=True)

    payload = {
        "Scan_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Universe": "NIFTY 750",
        "Fresh_Triggers": fresh_triggers,
        "Watchlist": watchlist,
        "Active_Positions": active_trailing_positions,
        "active_trailing_positions": active_trailing_positions
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Generated {RESULTS_OUT}: {len(fresh_triggers)} Fresh, {len(watchlist)} Watchlist, {len(active_trailing_positions)} Active Positions.")


if __name__ == "__main__":
    run_swing_ultra750()
