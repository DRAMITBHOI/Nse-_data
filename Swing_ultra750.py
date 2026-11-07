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

CONFIG_750 = {
    "min_turnover_cr": 10.0,
    # Strategy 2: Launchpad Coiling & Cheat Entry
    "lp_base_min": 6.0,
    "lp_base_max": 12.0,
    "lp_base_len_min": 10,
    "lp_base_len_max": 18,
    "lp_vdu_ratio": 0.45,
    "lp_cheat_atr_ratio": 0.65,
    "lp_cheat_prox_min": 0.010,
    "lp_cheat_prox_max": 0.028,
    "lp_risk_cap": 4.5,
    
    # Coiling Pipeline Disqualification Gates
    "coiling_max_staleness_days": 25,
    "coiling_max_dist_days": 2,
    
    # V-Shaped Recovery
    "vr_drop_min": 14.5,
    "vr_drop_max": 25.0,
    "vr_drop_len_min": 8,
    "vr_drop_len_max": 22,
    "vr_range_close_min": 0.80,
    "vr_conv_min": 2.00,
    "vr_deliv_surge_min": 1.15,
    "vr_risk_cap": 5.0,
    
    # Combination 7 Exit Rules (Strategy 1 + Exit 3)
    "escalator_harvest_pct": 0.20,
    "escalator_r_mult": 1.5,
    "escalator_cap_pct": 6.0,
    "breakeven_cushion": 0.005,  # Ratchet stop to Entry + 0.5%
    "runner_trail_ema": 20
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
    active_coiling_watchlist = []
    active_trailing_positions = []

    print(f"🚀 Running Swing Ultra Nifty 750 (Combination 7 + Coiling State Pipeline) across {len(json_files)} symbols...")

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
            df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
            df["perf_60d"] = df["close"].pct_change(60).fillna(0)

            # ATR 14
            prev_close = df["close"].shift(1)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()

            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            highs = df["high"].values
            lows = df["low"].values
            opens = df["open"].values
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
            ema50s = df["ema_50"].values
            sma200s = df["sma_200"].values
            perf_60s = df["perf_60d"].values
            atr14s = df["atr_14"].values
            times = df["time"].values

            # Active position simulation
            in_trade = False
            entry_p = 0.0
            calc_sl = 0.0
            active_stop = 0.0
            entry_d = ""
            entry_idx = 0
            trade_setup = ""
            conv_score_val = 1.0
            conv_mult_str = "1.0x"
            hit_milestone = False
            harvested_pct = 0.0

            for i in range(40, len(df)):
                t = str(times[i])[:10]
                o = opens[i]
                c = closes[i]
                h = highs[i]
                l = lows[i]
                v = volumes[i]
                v_avg = vol_avgs[i]
                dv = deliv_vols[i]
                dv_avg = deliv_avgs[i]
                rc = range_closes[i]
                ema20 = ema20s[i]
                ema10 = ema10s[i]
                ema50 = ema50s[i]
                atr = atr14s[i]

                # ==============================================================
                # 1. COMBINATION 7 TRAILING & EXIT ENGINE
                # ==============================================================
                if in_trade:
                    current_gain_pct = ((c - entry_p) / entry_p) * 100.0
                    r_dist = entry_p - calc_sl
                    t1_pct = min(CONFIG_750["escalator_r_mult"] * ((r_dist / entry_p) * 100.0), CONFIG_750["escalator_cap_pct"])
                    t1_target_price = entry_p * (1.0 + t1_pct / 100.0)

                    exit_triggered = False
                    exit_reason = ""

                    # 1. Gap-down breach at 9:15 AM Open
                    if o < active_stop:
                        exit_triggered = True
                        exit_reason = "1. Gap-Down Breach at 9:15 AM Open"
                    # 2. Hard stop intraday breach (pre-1.5R)
                    elif not hit_milestone and l <= active_stop:
                        exit_triggered = True
                        exit_reason = "2. Initial Hard Stop Hit"
                    else:
                        # 3. Strategy 1: Milestone 1 (+1.5R) Harvest
                        if not hit_milestone and h >= t1_target_price:
                            hit_milestone = True
                            harvested_pct = CONFIG_750["escalator_harvest_pct"]
                            active_stop = max(active_stop, round(entry_p * (1.0 + CONFIG_750["breakeven_cushion"]), 2))

                        # 4. Strategy 1 + Exit 3: Trail remaining 80% on Daily 20 EMA
                        if hit_milestone:
                            active_stop = max(active_stop, ema20)
                            if c < ema20:
                                exit_triggered = True
                                exit_reason = "3. Daily Session Close Below 20 EMA"

                    if i == len(df) - 1 and not exit_triggered:
                        holding_days = i - entry_idx
                        active_trailing_positions.append({
                            "symbol": sym,
                            "setup": trade_setup,
                            "entry_date": entry_d,
                            "entry_price": round(float(entry_p), 2),
                            "current_close": round(float(c), 2),
                            "active_stop": round(float(active_stop), 2),
                            "trailing_stop_20ema": round(float(ema20), 2),
                            "hard_sl": round(float(calc_sl), 2),
                            "pnl_%": round(float(current_gain_pct), 2),
                            "hit_milestone_15r": hit_milestone,
                            "harvested_pct": f"{int(harvested_pct * 100)}%",
                            "conviction_score": conv_score_val,
                            "conv_multiplier": conv_mult_str,
                            "days": int(holding_days),
                            "holding_days": int(holding_days),
                            "status": "RUNNER TRAILING (20 EMA)" if hit_milestone else "ACTIVE (PRE-1.5R)"
                        })

                    if exit_triggered:
                        in_trade = False
                    continue

                # ==============================================================
                # 2. ENTRY SCANNER
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

                # --- V-REVERSAL ENGINE ---
                recent_20_h = highs[max(0, i - 20):i].max()
                if recent_20_h > 0:
                    drop_depth = ((recent_20_h - l) / recent_20_h) * 100.0
                    if CONFIG_750["vr_drop_min"] <= drop_depth <= CONFIG_750["vr_drop_max"]:
                        prior_3d_h = highs[max(0, i - 3):i].max()
                        prior_3d_l = lows[max(0, i - 3):i].min()
                        c_reclaim = (c >= prior_3d_h) and (c > closes[i - 1])
                        c_deliv_rev = (dv >= CONFIG_750["vr_deliv_surge_min"] * dv_avg) or (deliv_pcts[i] >= 1.15 * deliv_pct_avgs[i])
                        c_candle_rev = rc >= CONFIG_750["vr_range_close_min"]

                        if c_reclaim and c_deliv_rev and c_candle_rev:
                            v_sl = round(prior_3d_l * 0.995, 2)
                            v_risk = ((prior_3d_h - v_sl) / prior_3d_h) * 100.0
                            if 0.5 <= v_risk <= CONFIG_750["vr_risk_cap"]:
                                entry_triggered = True
                                c_entry = round(prior_3d_h, 2)
                                c_sl = v_sl
                                c_setup = "V-REVERSAL"

                # --- STRATEGY 2: LAUNCHPAD SPRINGBOARD ("CHEAT") TRIGGER ---
                # Checks for the cheat contraction candle breaking today
                for shelf_len in range(CONFIG_750["lp_base_len_min"], CONFIG_750["lp_base_len_max"] + 1):
                    s_h = highs[max(0, i - shelf_len):i].max()
                    s_l = lows[max(0, i - shelf_len):i].min()
                    spread = ((s_h - s_l) / s_l) * 100.0 if s_l > 0 else 99.0
                    if s_l > 0 and (CONFIG_750["lp_base_min"] <= spread <= CONFIG_750["lp_base_max"]):
                        base_df_vols = volumes[max(0, i - 15):i]
                        lowest_5d_vol = pd.Series(base_df_vols).rolling(5, min_periods=1).mean().min() if len(base_df_vols) > 0 else v_avg
                        vdu_ok = (lowest_5d_vol / v_avg <= CONFIG_750["lp_vdu_ratio"]) if v_avg > 0 else False

                        # Prior candle was Cheat Bar (coiled narrow range under ceiling)
                        prev_range = highs[i - 1] - lows[i - 1]
                        prev_atr = atr14s[i - 1]
                        cheat_bar_ok = (prev_atr > 0) and (prev_range / prev_atr <= CONFIG_750["lp_cheat_atr_ratio"])
                        prox = (s_h - closes[i - 1]) / s_h if s_h > 0 else 1.0
                        prox_ok = CONFIG_750["lp_cheat_prox_min"] <= prox <= CONFIG_750["lp_cheat_prox_max"]
                        above_10ema = closes[i - 1] >= ema10s[i - 1]

                        # Trigger: Today's high breaks prior cheat bar high
                        cheat_entry = round(highs[i - 1] + 0.05, 2)
                        if vdu_ok and cheat_bar_ok and prox_ok and above_10ema and (h >= cheat_entry):
                            cheat_sl = round(lows[i - 1] * 0.995, 2)
                            compressed_risk = ((cheat_entry - cheat_sl) / cheat_entry) * 100.0
                            if 0.5 <= compressed_risk <= CONFIG_750["lp_risk_cap"]:
                                entry_triggered = True
                                c_entry = cheat_entry
                                c_sl = cheat_sl
                                c_setup = "LAUNCHPAD CHEAT"
                                break

                # Register execution
                if entry_triggered:
                    deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
                    rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
                    conv_score = round(deliv_surge * max(0.1, rs_ratio), 3)

                    risk_mult_tag = "2.0x" if conv_score >= 3.20 else ("1.5x" if conv_score >= 2.60 else "1.0x")

                    if c_setup == "LAUNCHPAD CHEAT" and conv_score < 2.30:
                        pass
                    else:
                        in_trade = True
                        entry_p = c_entry
                        calc_sl = c_sl
                        active_stop = c_sl
                        entry_d = t
                        entry_idx = i
                        trade_setup = c_setup
                        conv_score_val = conv_score
                        conv_mult_str = risk_mult_tag
                        hit_milestone = False
                        harvested_pct = 0.0

                        if i == len(df) - 1:
                            ext_pct = round(((c - c_entry) / c_entry) * 100.0, 2)
                            risk_pct = round(((c_entry - c_sl) / c_entry) * 100.0, 2)
                            t1_harvest_p = round(c_entry * (1.0 + min(1.5 * risk_pct, 6.0) / 100.0), 2)
                            action_tag = "BUY STOP-LIMIT" if ext_pct <= 1.5 else "LIMIT RETEST"

                            fresh_triggers.append({
                                "symbol": sym,
                                "type": c_setup,
                                "entry_trigger": f"₹{c_entry:.2f}",
                                "close": f"₹{c:.2f}",
                                "extended_%": f"{ext_pct:+.1f}%",
                                "cheat_stop_loss": f"₹{c_sl:.2f}",
                                "risk_%": f"{risk_pct:.2f}%",
                                "t1_harvest_target": f"₹{t1_harvest_p:.2f} (+1.5R)",
                                "conviction_score": conv_score,
                                "conv_multiplier": risk_mult_tag,
                                "turnover": f"₹{to_50d[i]:.1f} Cr",
                                "action": action_tag
                            })

            # ==================================================================
            # 3. STATE-TRACKED PERSISTENT COILING WATCHLIST PIPELINE
            # Scans historical coiling bases and validates against 4 Disqualification Gates
            # ==================================================================
            if not in_trade and len(df) >= 30:
                last_idx = len(df) - 1
                curr_c = closes[last_idx]
                curr_ema50 = ema50s[last_idx]

                # Scan looking back up to 25 sessions for an active base formation
                for base_end_offset in range(0, 16):
                    b_idx = last_idx - base_end_offset
                    if b_idx < CONFIG_750["lp_base_len_max"]:
                        continue

                    # Test base lengths (10 to 18 sessions)
                    for b_len in range(CONFIG_750["lp_base_len_min"], CONFIG_750["lp_base_len_max"] + 1):
                        base_highs = highs[b_idx - b_len : b_idx]
                        base_lows = lows[b_idx - b_len : b_idx]
                        ceil = base_highs.max()
                        floor = base_lows.min()
                        spread = ((ceil - floor) / floor) * 100.0 if floor > 0 else 99.0

                        if not (CONFIG_750["lp_base_min"] <= spread <= CONFIG_750["lp_base_max"]):
                            continue

                        # Check VDU inside base
                        base_vols = volumes[b_idx - b_len : b_idx]
                        vdu_val = pd.Series(base_vols).rolling(5, min_periods=1).mean().min()
                        v_ref = vol_avgs[b_idx]
                        if v_ref <= 0 or (vdu_val / v_ref > CONFIG_750["lp_vdu_ratio"]):
                            continue

                        # Coiling found. Now evaluate the 4 Disqualification Gates:
                        post_base_closes = closes[b_idx : last_idx + 1]
                        post_base_vols = volumes[b_idx : last_idx + 1]
                        post_base_vol_avgs = vol_avgs[b_idx : last_idx + 1]
                        coiling_duration = (last_idx - (b_idx - b_len))

                        # Gate 1: Floor Breakdown
                        if curr_c < (floor * 0.99):
                            continue

                        # Gate 2: Max Staleness (> 25 sessions)
                        if coiling_duration > CONFIG_750["coiling_max_staleness_days"]:
                            continue

                        # Gate 3: Heavy Distribution Contamination (> 2 days)
                        dist_days = 0
                        for k in range(len(post_base_closes)):
                            k_idx = b_idx + k
                            if (post_base_closes[k] < closes[k_idx - 1]) and (post_base_vols[k] >= 1.35 * post_base_vol_avgs[k]) and (range_closes[k_idx] <= 0.35):
                                dist_days += 1
                        if dist_days > CONFIG_750["coiling_max_dist_days"]:
                            continue

                        # Gate 4: Daily Close < Daily 50 EMA
                        if curr_c < curr_ema50:
                            continue

                        # Candidate is QUALIFIED & ACTIVE in Coiling State
                        dist_to_ceiling = round(((ceil - curr_c) / curr_c) * 100.0, 2)
                        
                        # Find tightest inside contraction bar for Cheat Level
                        recent_h = highs[max(0, last_idx - 3) : last_idx + 1]
                        recent_l = lows[max(0, last_idx - 3) : last_idx + 1]
                        cheat_entry_lvl = round(recent_h.min() + 0.05, 2)
                        cheat_sl_lvl = round(recent_l.min() * 0.995, 2)
                        cheat_risk = round(((cheat_entry_lvl - cheat_sl_lvl) / cheat_entry_lvl) * 100.0, 2)

                        deliv_surge = deliv_vols[last_idx] / deliv_avgs[last_idx] if deliv_avgs[last_idx] > 0 else 1.0
                        rs_ratio = (perf_60s[last_idx] / n_perf) if n_perf > 0 else 1.0
                        conv_score = round(deliv_surge * max(0.1, rs_ratio), 2)

                        active_coiling_watchlist.append({
                            "symbol": sym,
                            "setup_class": "LAUNCHPAD ACTIVE COIL",
                            "close": f"₹{curr_c:.2f}",
                            "pivot_ceiling": f"₹{ceil:.2f}",
                            "base_floor": f"₹{floor:.2f}",
                            "cheat_entry_level": f"₹{cheat_entry_lvl:.2f}",
                            "cheat_stop_loss": f"₹{cheat_sl_lvl:.2f}",
                            "compressed_risk_%": f"{cheat_risk:.2f}%",
                            "dist_to_ceiling_%": f"{dist_to_ceiling:+.2f}%",
                            "base_spread_%": f"{spread:.2f}%",
                            "coiling_duration_days": int(coiling_duration),
                            "conviction_score": conv_score,
                            "turnover": f"₹{to_50d[last_idx]:.1f} Cr",
                            "status": "MONITORING_FOR_LIVE_SCANNER"
                        })
                        break
                    if len(active_coiling_watchlist) > 0 and active_coiling_watchlist[-1]["symbol"] == sym:
                        break
        except Exception:
            continue

    fresh_triggers.sort(key=lambda x: x["conviction_score"], reverse=True)
    active_coiling_watchlist.sort(key=lambda x: float(x["dist_to_ceiling_%"].replace("%", "").replace("+", "")))
    active_trailing_positions.sort(key=lambda x: x["pnl_%"], reverse=True)

    payload = {
        "Scan_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Universe": "NIFTY 750",
        "Strategy_Framework": "Combination 7 (Strategy 1 + Strategy 2 Cheat + Strategy 3 Sizing + Exit 3)",
        "Fresh_Triggers": fresh_triggers,
        "Active_Coiling_Watchlist": active_coiling_watchlist,
        "Watchlist": active_coiling_watchlist,
        "Active_Positions": active_trailing_positions,
        "active_trailing_positions": active_trailing_positions
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Generated {RESULTS_OUT}: {len(fresh_triggers)} Fresh Triggers, {len(active_coiling_watchlist)} Active Coiling Candidates, {len(active_trailing_positions)} Active Positions.")


if __name__ == "__main__":
    run_swing_ultra750()
