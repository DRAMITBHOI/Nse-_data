import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_exit_efficiency_results.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
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
                with open(p, "r") as fp:
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
            with open(NIFTY750_FILE, "r") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def extract_entry_candidates(clean_data, nifty_perf_map):
    if len(clean_data) < 60:
        return []

    df = pd.DataFrame(clean_data)
    df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["deliv_sma"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(0)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    direction = np.where(df["close"] >= df["close"].shift(1), 1.0, -1.0)
    df["demat_obv"] = (direction * df["delivery_vol"]).cumsum()
    df["demat_obv_sma20"] = df["demat_obv"].rolling(20, min_periods=5).mean()
    df["demat_obv_20max"] = df["demat_obv"].rolling(20, min_periods=5).max()

    c_range = df["high"] - df["low"]
    df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.0)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vol_avgs = df["gross_vol_sma20"].values
    deliv_vols = df["delivery_vol"].values
    deliv_avgs = df["deliv_sma"].values
    to_50d = df["turnover_50d"].values
    deliv_pcts = df["deliv_pct"].values
    deliv_pct_avgs = df["deliv_pct_50d"].values
    range_closes = df["range_closeness"].values
    ema20s = df["ema20"].values
    sma200s = df["sma200"].values
    perf_60s = df["perf_60d"].values
    demat_obvs = df["demat_obv"].values
    demat_obv_smas = df["demat_obv_sma20"].values
    demat_obv_maxes = df["demat_obv_20max"].values
    times = df["time"].values

    candidates = []

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
        sma200 = sma200s[i]
        stock_p60 = perf_60s[i]
        t = times[i]

        if turnover < MIN_TURNOVER_CR:
            continue

        entry_triggered = False
        calc_entry = 0.0
        calc_sl = 0.0

        # ENGINE 1: V-REVERSAL
        recent_20_high = highs[i - 20:i].max()
        if (recent_20_high - l) / recent_20_high >= 0.18:
            recent_trough = lows[i - 5:i].min()
            prior_3d_high = highs[i - 4:i].max()
            c_reclaim = (c >= prior_3d_high) and (c > closes[i - 1])
            c_vol_rev = v >= (1.3 * v_avg)
            c_deliv_rev = (dv >= 1.25 * dv_avg) or (dp >= 1.15 * dp_avg if dp_avg > 0 else False)
            c_candle_rev = rc >= 0.55

            if c_reclaim and c_vol_rev and c_deliv_rev and c_candle_rev:
                entry_triggered = True
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)

        # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT (SPREAD <= 8%)
        if not entry_triggered:
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0 and ((s_high - s_low) / s_low) <= 0.08:
                    valid_launchpad = True
                    launchpad_high = round(float(s_high), 2)
                    launchpad_low = round(float(s_low), 2)
                    break

            if valid_launchpad:
                base_up_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] >= closes[k - 1])
                base_down_deliv = sum(deliv_vols[k] for k in range(i - 15, i) if closes[k] < closes[k - 1])
                base_cumul_ratio = (base_up_deliv / base_down_deliv) if base_down_deliv > 0 else 1.5

                c_bo = (c >= launchpad_high) and (h >= launchpad_high)
                c_vol_bo = v >= (1.4 * v_avg)
                c_deliv_bo = (dv >= 1.25 * dv_avg) or (dp >= 1.25 * dp_avg if dp_avg > 0 else False)
                c_candle_bo = rc >= 0.65
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i - 1]) if i > 0 else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.50) and obv_high:
                    entry_triggered = True
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
                n_perf = nifty_perf_map.get(t, 0.001)
                is_rs = bool(stock_p60 > n_perf)

                if is_stage2 and is_rs:
                    deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                    rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                    conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                    if conviction_score >= 2.5:
                        candidates.append({
                            "entry_idx": i,
                            "entry_date": t,
                            "entry_price": calc_entry,
                            "initial_stop": calc_sl,
                            "r_unit": r_dist,
                            "conviction_score": conviction_score,
                            "highs": highs,
                            "lows": lows,
                            "closes": closes,
                            "volumes": volumes,
                            "deliv_vols": deliv_vols,
                            "deliv_avgs": deliv_avgs,
                            "ema20s": ema20s,
                            "demat_obvs": demat_obvs,
                            "demat_obv_smas": demat_obv_smas,
                            "times": times,
                        })

    return candidates


def simulate_exit(cand, mode="baseline_20ema"):
    entry_idx = cand["entry_idx"]
    entry_p = cand["entry_price"]
    initial_sl = cand["initial_stop"]
    r_unit = cand["r_unit"]
    highs = cand["highs"]
    lows = cand["lows"]
    closes = cand["closes"]
    deliv_vols = cand["deliv_vols"]
    deliv_avgs = cand["deliv_avgs"]
    ema20s = cand["ema20s"]
    demat_obvs = cand["demat_obvs"]
    demat_obv_smas = cand["demat_obv_smas"]
    times = cand["times"]

    days_below_ema = 0
    days_in_tight_chop = 0
    highest_close_since_entry = entry_p

    for i in range(entry_idx + 1, len(closes)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        ema = ema20s[i]
        dv = deliv_vols[i]
        dv_avg = deliv_avgs[i]
        dobv = demat_obvs[i]
        dobv_sma = demat_obv_smas[i]

        highest_close_since_entry = max(highest_close_since_entry, c)

        # -------------------------------------------------------------
        # VARIANT A: CURRENT BASELINE (CLOSE < MAX(INITIAL_SL, 20 EMA))
        # -------------------------------------------------------------
        if mode == "baseline_20ema":
            current_sl = max(initial_sl, round(float(ema), 2))
            if c < current_sl:
                exit_p = round(current_sl, 2)
                pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

        # -------------------------------------------------------------
        # VARIANT B: INSTITUTIONAL OBV DISTRIBUTION EXIT
        # -------------------------------------------------------------
        elif mode == "institutional_obv":
            # Hard floor: structural stop is always defended
            if c < initial_sl:
                pnl = round(((initial_sl - entry_p) / entry_p) * 100.0, 2)
                return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

            if c < ema:
                # Check for genuine institutional selling:
                # 1. Demat OBV is below its 20 SMA, OR
                # 2. Heavy red delivery volume printed (>= 1.3x SMA)
                is_distribution = (dobv < dobv_sma) or (c < closes[i - 1] and dv >= 1.3 * dv_avg)
                if is_distribution:
                    exit_p = round(c, 2)
                    pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                    return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

        # -------------------------------------------------------------
        # VARIANT C: BALANCED INSTITUTIONAL + 8-DAY SIDEWAYS TIME DECAY
        # -------------------------------------------------------------
        elif mode == "balanced_time_decay":
            # 1. Hard Floor: Structural stop loss must never be breached
            structural_stop = max(initial_sl, highest_close_since_entry * 0.88)  # Don't give up >12% from peak
            if c < structural_stop:
                exit_p = round(c, 2)
                pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

            # 2. Track sideways and below-EMA drift
            if c < ema:
                days_below_ema += 1
                is_inst_selling = (dobv < dobv_sma) or (c < closes[i - 1] and dv >= 1.3 * dv_avg)
                
                # Immediate exit if institutions are actively dumping
                if is_inst_selling:
                    exit_p = round(c, 2)
                    pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                    return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

                # Sideways Time Decay Rule: If below 20 EMA for > 6 consecutive bars without reclaiming, exit
                if days_below_ema >= 6:
                    exit_p = round(c, 2)
                    pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                    return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}
            else:
                days_below_ema = 0  # Reclaimed EMA: reset counter, let it run

            # 3. Flat Sideways Penalty (Dead Money Rule)
            # If after 15 days in trade, the stock hasn't moved more than +3% from entry, exit to free capital
            if (i - entry_idx) >= 15 and c <= (entry_p * 1.03):
                exit_p = round(c, 2)
                pnl = round(((exit_p - entry_p) / entry_p) * 100.0, 2)
                return {"pnl": pnl, "duration": i - entry_idx, "outcome": "WIN" if pnl > 0 else "LOSS"}

    last_pnl = round(((closes[-1] - entry_p) / entry_p) * 100.0, 2)
    return {"pnl": last_pnl, "duration": len(closes) - 1 - entry_idx, "outcome": "WIN" if last_pnl > 0 else "LOSS"}


def compute_metrics(trades_list, label):
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]

    win_rate = round((len(wins) / total) * 100.0, 2)
    avg_gain = round(wins["pnl"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["pnl"].mean(), 2) if not losses.empty else 0.0
    max_gain = round(df["pnl"].max(), 2) if not df.empty else 0.0
    avg_days = round(df["duration"].mean(), 1)

    gross_win = wins["pnl"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 1.0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "Exit Model": label,
        "Trades": total,
        "Win Rate %": win_rate,
        "Profit Factor": pf,
        "Avg Gain %": avg_gain,
        "Avg Loss %": avg_loss,
        "Max Profit %": max_gain,
        "Avg Hold (Days)": avg_days,
    }


def run_efficiency_study():
    nifty_perf_map = load_nifty_benchmark()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "gap_margin_candidates.json", "swing3_results.json", "backtest_results.json",
        "swing3_compare.json", "swing3_run2_results.json", "swing3_run2_compare_results.json",
        "swing3_run3_results.json", "swing3_target400_results.json", "swing3_exit_study_results.json",
        "swing3_scanner_results.json", "swing3_exit_efficiency_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"🚀 Scanning {len(target_files)} symbols in Nifty 750 across institutional exit variants...")
    all_raw = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym not in nifty750_set:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                cands = extract_entry_candidates(clean, nifty_perf_map)
                all_raw.extend(cands)
        except Exception:
            continue

    if not all_raw:
        print("⚠️ No qualifying candidates found.")
        return

    # Max 1/day by Conviction Score (Identical trade baseline)
    cands_by_date = {}
    for c in all_raw:
        cands_by_date.setdefault(c["entry_date"], []).append(c)

    filtered = []
    for d, d_cands in sorted(cands_by_date.items()):
        d_cands.sort(key=lambda x: x["conviction_score"], reverse=True)
        filtered.append(d_cands[0])

    print(f"✅ Filtered candidates ready: {len(filtered)} trades evaluated.")

    # 1. Baseline
    t_v1 = [simulate_exit(c, "baseline_20ema") for c in filtered]
    m_v1 = compute_metrics(t_v1, "1. Current: Strict 20 EMA Close Exit")

    # 2. Pure Institutional OBV
    t_v2 = [simulate_exit(c, "institutional_obv") for c in filtered]
    m_v2 = compute_metrics(t_v2, "2. Institutional: Close < 20 EMA + OBV Dump")

    # 3. Balanced Institutional + 8-Day Sideways Decay
    t_v3 = [simulate_exit(c, "balanced_time_decay") for c in filtered]
    m_v3 = compute_metrics(t_v3, "3. Balanced: OBV Buffer + 6d Decay + 15d Chop Exit")

    report = [m_v1, m_v2, m_v3]
    df_report = pd.DataFrame(report)
    df_sorted = df_report.sort_values(by=["Profit Factor", "Win Rate %"], ascending=[False, False]).reset_index(drop=True)
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    print("\n" + "=" * 115)
    print("🏆 SWING 3.0: EXIT EFFICIENCY & INSTITUTIONAL RETENTION STUDY")
    print("=" * 115)
    print(df_sorted.to_string(index=False))
    print("=" * 115 + "\n")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": df_sorted.to_dict(orient="records"),
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    run_efficiency_study()
