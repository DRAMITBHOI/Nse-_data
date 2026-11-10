import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
RESULTS_FILE = "swing_ultra750_results.json"
ACTIVE_WATCHLIST_FILE = os.path.join(DATA_DIR, "active_watchlist.json")


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


def merge_active_watchlist(new_items, current_universe):
    """Merges new pre-breakout candidates into active_watchlist.json while preserving other universe items."""
    existing = {}
    if os.path.exists(ACTIVE_WATCHLIST_FILE):
        try:
            with open(ACTIVE_WATCHLIST_FILE, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
        except Exception:
            existing = {}

    # Remove stale items for the current universe, replace with freshly validated list
    cleaned = {k: v for k, v in existing.items() if v.get("universe") != current_universe}
    for item in new_items:
        sym = item["symbol"]
        cleaned[sym] = item

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ACTIVE_WATCHLIST_FILE, "w", encoding="utf-8") as fp:
        json.dump(cleaned, fp, indent=2)


def run_swing_ultra750():
    print("=" * 85)
    print("🚀 SWING ULTRA 750 (COMBINATION 7 - GAP RULES & WATCHLIST GENERATOR)")
    print("   Universe: Nifty 750 Core (Turnover >= ₹10 Cr/d)")
    print("=" * 85)

    n750_set = load_nifty750_symbols()
    nifty_perf_map = load_nifty_benchmark()
    nifty_dates = sorted(nifty_perf_map.keys())

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing_ultra750_results.json",
        "swing_ultraN750_results.json", "live_intraday_breakouts.json",
        "live_intraday_breakouts_v2.json", "accuracy_audit_log.json", "my_trades.json",
        "gap_down_audit_results.json", "gap_down_audit_full_results.json",
        "gap_up_empirical_audit_results.json", "active_watchlist.json"
    }

    target_files = [p for p in sorted(json_files) if os.path.basename(p) not in excluded]
    if n750_set:
        target_files = [p for p in target_files if os.path.splitext(os.path.basename(p))[0].upper() in n750_set]

    print(f"📊 Processing {len(target_files)} Nifty 750 stocks...")

    trades = []
    active_prebreakouts = []
    shakeouts_absorbed = 0
    severe_gaps_cut = 0
    normal_entries = 0
    modest_gaps_entered = 0
    extended_gaps_skipped = 0

    for path in target_files:
        sym = os.path.splitext(os.path.basename(path))[0].upper()
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

            prev_c = df["close"].shift(1)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
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

            in_trade = False
            entry_p = 0.0
            calc_sl = 0.0
            active_stop = 0.0
            trade_setup = ""
            hit_milestone = False
            t1_target_p = 0.0
            entry_d = ""
            conv_tag = ""

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
                ema50 = ema50s[i]
                atr = atr14s[i]

                if in_trade:
                    exit_triggered = False
                    exit_p = c
                    exit_reason = ""

                    # BIFURCATED GAP-DOWN PROTOCOL
                    if o < active_stop:
                        gap_depth_pct = ((active_stop - o) / active_stop) * 100.0
                        is_severe = (gap_depth_pct > 2.0) or (o < ema50)

                        if is_severe:
                            exit_triggered = True
                            exit_p = o
                            exit_reason = "Gap-Down Severe / 50 EMA Breakdown (9:15 AM)"
                            severe_gaps_cut += 1
                        else:
                            if c >= active_stop:
                                shakeouts_absorbed += 1
                            else:
                                exit_triggered = True
                                exit_p = round(o * 0.988, 2)
                                exit_reason = "Gap-Down Leeway Failed (9:30 AM)"

                    if not exit_triggered:
                        if not hit_milestone and l <= active_stop:
                            exit_triggered = True
                            exit_p = active_stop
                            exit_reason = "Hard SL Hit"
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

                    if exit_triggered:
                        if hit_milestone:
                            t1_gain = ((t1_target_p - entry_p) / entry_p) * 100.0
                            runner_gain = ((exit_p - entry_p) / entry_p) * 100.0
                            blended_gain = (0.20 * t1_gain) + (0.80 * runner_gain)
                        else:
                            blended_gain = ((exit_p - entry_p) / entry_p) * 100.0

                        risk_pct = ((entry_p - calc_sl) / entry_p) * 100.0
                        r_mult = blended_gain / risk_pct if risk_pct > 0 else 0.0

                        trades.append({
                            "symbol": sym,
                            "entry_date": entry_d,
                            "exit_date": t,
                            "setup": trade_setup,
                            "conviction": conv_tag,
                            "entry_price": entry_p,
                            "exit_price": exit_p,
                            "gain_pct": round(blended_gain, 2),
                            "r_mult": round(r_mult, 2),
                            "exit_reason": exit_reason,
                            "hold_days": (pd.to_datetime(t) - pd.to_datetime(entry_d)).days,
                            "hit_milestone": hit_milestone,
                        })
                        in_trade = False
                    continue

                # -------------------------------------------------------------
                # ENTRY SCANNERS
                # -------------------------------------------------------------
                if to_50d[i] < 10.0:
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
                    if 14.5 <= drop_depth <= 25.0:
                        prior_3d_h = highs[max(0, i - 3) : i].max()
                        prior_3d_l = lows[max(0, i - 3) : i].min()
                        c_reclaim = (c >= prior_3d_h) and (c > closes[i - 1])
                        c_deliv = (dv >= 1.15 * dv_avg) or (deliv_pcts[i] >= 1.15 * deliv_pct_avgs[i])
                        c_candle = rc >= 0.80

                        if c_reclaim and c_deliv and c_candle:
                            v_sl = round(prior_3d_l * 0.995, 2)
                            v_risk = ((prior_3d_h - v_sl) / prior_3d_h) * 100.0
                            if 0.5 <= v_risk <= 5.0:
                                entry_triggered = True
                                c_entry = round(prior_3d_h + 0.05, 2)
                                c_sl = v_sl
                                c_setup = "V-REVERSAL"

                # Engine 2: Launchpad Cheat Entry
                if not entry_triggered:
                    for s_len in range(10, 19):
                        s_h = highs[max(0, i - s_len) : i].max()
                        s_l = lows[max(0, i - s_len) : i].min()
                        spread = ((s_h - s_l) / s_l) * 100.0 if s_l > 0 else 99.0
                        if s_l > 0 and (6.0 <= spread <= 12.0):
                            base_vols = volumes[max(0, i - 15) : i]
                            lowest_5d = pd.Series(base_vols).rolling(5, min_periods=1).mean().min() if len(base_vols) > 0 else v_avg
                            vdu_ok = (lowest_5d / v_avg <= 0.45) if v_avg > 0 else False

                            prev_h = highs[i - 1]
                            prev_l = lows[i - 1]
                            prev_atr = atr14s[i - 1]
                            cheat_ok = (prev_atr > 0) and ((prev_h - prev_l) / prev_atr <= 0.65)
                            prox = (s_h - closes[i - 1]) / s_h if s_h > 0 else 1.0
                            prox_ok = 0.005 <= prox <= 0.028
                            above_10ema = closes[i - 1] >= ema10s[i - 1]

                            if vdu_ok and cheat_ok and prox_ok and above_10ema:
                                cheat_e = round(prev_h + 0.05, 2)
                                cheat_s = round(prev_l * 0.995, 2)

                                # GAP-UP EVALUATION
                                gap_pct = ((o - cheat_e) / cheat_e) * 100.0 if cheat_e > 0 else 0.0
                                if gap_pct > 2.0:
                                    extended_gaps_skipped += 1
                                    continue
                                elif gap_pct > 0.0:
                                    if h <= o:
                                        continue
                                    effective_entry = o
                                    modest_gaps_entered += 1
                                else:
                                    if h < cheat_e:
                                        continue
                                    effective_entry = cheat_e
                                    normal_entries += 1

                                comp_risk = ((effective_entry - cheat_s) / effective_entry) * 100.0
                                if 0.5 <= comp_risk <= 4.5:
                                    entry_triggered = True
                                    c_entry = effective_entry
                                    c_sl = cheat_s
                                    c_setup = "CHEAT BREAKOUT"
                                    break

                if entry_triggered:
                    deliv_surge = dv / dv_avg if dv_avg > 0 else 1.0
                    rs_ratio = (perf_60s[i] / n_perf) if n_perf > 0 else (1.0 + abs(perf_60s[i]))
                    conv_score = round(deliv_surge * max(0.1, rs_ratio), 3)

                    if c_setup == "CHEAT BREAKOUT" and conv_score < 2.30:
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
                    conv_tag = "2.0x" if conv_score >= 3.20 else ("1.5x" if conv_score >= 2.60 else "1.0x")

            # =================================================================
            # PRE-BREAKOUT WATCHLIST SCANNER (EVALUATING LATEST COMPLETED BAR)
            # =================================================================
            last_idx = len(df) - 1
            if last_idx >= 40 and not in_trade:
                last_c = closes[last_idx]
                last_h = highs[last_idx]
                last_l = lows[last_idx]
                last_v_avg = vol_avgs[last_idx]
                last_to = to_50d[last_idx]
                last_sma200 = sma200s[last_idx]
                last_perf = perf_60s[last_idx]
                last_atr = atr14s[last_idx]
                last_t = str(times[last_idx])[:10]

                n_perf_latest = nifty_perf_map.get(last_t)
                if n_perf_latest is None:
                    p_dates = [d for d in nifty_dates if d <= last_t]
                    n_perf_latest = nifty_perf_map[p_dates[-1]] if p_dates else 0.001

                stage2_ok = bool(last_c >= last_sma200) if pd.notnull(last_sma200) else False
                rs_ok = bool(last_perf > n_perf_latest)
                turnover_ok = bool(last_to >= 10.0)

                if stage2_ok and rs_ok and turnover_ok:
                    pb_found = False
                    pb_record = None

                    # Check Launchpad Cheat Setup on latest candle
                    for s_len in range(10, 19):
                        s_h = highs[max(0, last_idx - s_len + 1) : last_idx + 1].max()
                        s_l = lows[max(0, last_idx - s_len + 1) : last_idx + 1].min()
                        spread = ((s_h - s_l) / s_l) * 100.0 if s_l > 0 else 99.0
                        if s_l > 0 and (6.0 <= spread <= 12.0):
                            base_vols = volumes[max(0, last_idx - 14) : last_idx + 1]
                            lowest_5d = pd.Series(base_vols).rolling(5, min_periods=1).mean().min() if len(base_vols) > 0 else last_v_avg
                            vdu_ratio = round(lowest_5d / last_v_avg, 3) if last_v_avg > 0 else 1.0
                            vdu_ok = (vdu_ratio <= 0.45)

                            cheat_ok = (last_atr > 0) and ((last_h - last_l) / last_atr <= 0.65)
                            prox = (s_h - last_c) / s_h if s_h > 0 else 1.0
                            prox_ok = 0.005 <= prox <= 0.028
                            above_10ema = last_c >= ema10s[last_idx]

                            if vdu_ok and cheat_ok and prox_ok and above_10ema:
                                pivot_p = round(last_h + 0.05, 2)
                                sl_p = round(last_l * 0.995, 2)
                                r_pct = round(((pivot_p - sl_p) / pivot_p) * 100.0, 2)
                                if 0.5 <= r_pct <= 4.5:
                                    pb_record = {
                                        "symbol": sym,
                                        "universe": "NIFTY 750",
                                        "setup": "LAUNCHPAD CHEAT",
                                        "pivot_ceiling": pivot_p,
                                        "stop_loss": sl_p,
                                        "max_allowed_open": round(pivot_p * 1.02, 2),
                                        "shelf_high": round(s_h, 2),
                                        "shelf_spread_%": round(spread, 2),
                                        "last_close": round(last_c, 2),
                                        "last_atr": round(last_atr, 2),
                                        "risk_%": r_pct,
                                        "vdu_ratio": vdu_ratio,
                                        "qualify_date": last_t,
                                        "added_date": datetime.now().strftime("%Y-%m-%d")
                                    }
                                    pb_found = True
                                    break

                    # Check V-Reversal Setup on latest candle if Cheat not found
                    if not pb_found:
                        recent_20_h = highs[max(0, last_idx - 20) : last_idx + 1].max()
                        if recent_20_h > 0:
                            drop_depth = ((recent_20_h - last_l) / recent_20_h) * 100.0
                            if 14.5 <= drop_depth <= 25.0:
                                prior_3d_h = highs[max(0, last_idx - 2) : last_idx + 1].max()
                                prior_3d_l = lows[max(0, last_idx - 2) : last_idx + 1].min()
                                v_sl = round(prior_3d_l * 0.995, 2)
                                v_piv = round(prior_3d_h + 0.05, 2)
                                v_risk = round(((v_piv - v_sl) / v_piv) * 100.0, 2)
                                if 0.5 <= v_risk <= 5.0 and last_c >= ema10s[last_idx]:
                                    pb_record = {
                                        "symbol": sym,
                                        "universe": "NIFTY 750",
                                        "setup": "V-REVERSAL",
                                        "pivot_ceiling": v_piv,
                                        "stop_loss": v_sl,
                                        "max_allowed_open": round(v_piv * 1.02, 2),
                                        "shelf_high": round(recent_20_h, 2),
                                        "shelf_spread_%": round(drop_depth, 2),
                                        "last_close": round(last_c, 2),
                                        "last_atr": round(last_atr, 2),
                                        "risk_%": v_risk,
                                        "vdu_ratio": 1.0,
                                        "qualify_date": last_t,
                                        "added_date": datetime.now().strftime("%Y-%m-%d")
                                    }
                                    pb_found = True

                    if pb_found and pb_record:
                        active_prebreakouts.append(pb_record)

        except Exception:
            continue

    if not trades:
        print("⚠ No trades generated.")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["gain_pct"] > 0]
    losses = df_t[df_t["gain_pct"] <= 0]
    win_rate = (len(wins) / len(df_t)) * 100.0
    tot_r = df_t["r_mult"].sum()
    exp_r = tot_r / len(df_t)

    gross_win = wins["gain_pct"].sum()
    gross_loss = abs(losses["gain_pct"].sum())
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

    print("\n" + "=" * 85)
    print("📈 SWING ULTRA 750 (COMBINATION 7) AUDIT REPORT")
    print("=" * 85)
    print(f"Total Trades:           {len(df_t):,}")
    print(f"Win Rate:               {win_rate:.1f}%")
    print(f"Profit Factor:          {profit_factor}")
    print(f"Average Win:            +{wins['gain_pct'].mean():.2f}%")
    print(f"Average Loss:           {losses['gain_pct'].mean():.2f}%")
    print(f"Net Expectancy (%):     +{df_t['gain_pct'].mean():.2f}%")
    print(f"Total R-Multiple:       +{tot_r:,.1f} R")
    print(f"Expectancy (R/Trade):   +{exp_r:.2f} R")
    print(f"Average Hold:           {df_t['hold_days'].mean():.1f} days")
    print("-" * 85)
    print("🛡 GAP RISK MANAGEMENT STATS:")
    print(f"• Shakeouts Absorbed (Runners Retained): {shakeouts_absorbed}")
    print(f"• Severe Gaps Cut (Tail Loss Prevented): {severe_gaps_cut}")
    print(f"• Normal Intraday Entries:               {normal_entries}")
    print(f"• Modest Gap-Up Entries (<= 2.0%):       {modest_gaps_entered}")
    print(f"• Extended Gap-Ups Skipped (> 2.0%):     {extended_gaps_skipped} (Fade Traps Avoided)")
    print("-" * 85)
    print(f"🎯 Active Pre-Breakout Setups Qualified: {len(active_prebreakouts)}")
    print("=" * 85)

    # Save to swing_ultra750_results.json
    with open(RESULTS_FILE, "w", encoding="utf-8") as fp:
        json.dump({
            "universe": "NIFTY_750",
            "total_trades": len(df_t),
            "win_rate": round(win_rate, 2),
            "profit_factor": profit_factor,
            "total_r": round(tot_r, 2),
            "exp_r": round(exp_r, 2),
            "shakeouts_absorbed": shakeouts_absorbed,
            "severe_gaps_cut": severe_gaps_cut,
            "normal_entries": normal_entries,
            "modest_gaps_entered": modest_gaps_entered,
            "extended_gaps_skipped": extended_gaps_skipped,
            "Watchlist": active_prebreakouts,
            "recent_trades": trades[-50:]
        }, fp, indent=2)
    print(f"💾 Results & Watchlist saved to '{RESULTS_FILE}'.")

    # Update consolidated active_watchlist.json
    merge_active_watchlist(active_prebreakouts, "NIFTY 750")
    print(f"📋 Consolidated Active Watchlist updated in '{ACTIVE_WATCHLIST_FILE}'.")


if __name__ == "__main__":
    run_swing_ultra750()
