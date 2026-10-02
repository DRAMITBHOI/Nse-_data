import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request

DATA_DIR = "data"
RESULTS_JSON = os.path.join(DATA_DIR, "swing3_exit_study_results.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0


# -------------------------------------------------------------
# 1. DATA SANITIZATION & SPLIT ADJUSTMENT
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 2. BENCHMARK & UNIVERSE HELPERS
# -------------------------------------------------------------
def load_nifty_perf_map():
    nifty_candidates = [
        os.path.join(DATA_DIR, "nifty.json"),
        os.path.join(DATA_DIR, "nifty50.json"),
        os.path.join(DATA_DIR, "NIFTY.json"),
        os.path.join(DATA_DIR, "NIFTY50.json"),
    ]
    raw = None
    for path in nifty_candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, list) and len(data) > 50:
                        raw = data
                        break
            except Exception:
                continue

    if not raw:
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=1d&range=5y"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                yf_data = json.loads(resp.read().decode("utf-8"))
                res = yf_data["chart"]["result"][0]
                timestamps = res["timestamp"]
                quotes = res["indicators"]["quote"][0]
                raw = []
                for ts, o, h, l, c, v in zip(timestamps, quotes["open"], quotes["high"], quotes["low"], quotes["close"], quotes["volume"]):
                    if c is not None and c > 0:
                        d_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        raw.append({"time": d_str, "open": o, "high": h, "low": l, "close": c, "volume": v or 100000})
        except Exception:
            pass

    clean = clean_and_prepare(raw)
    if not clean:
        return {}

    df = pd.DataFrame(clean)
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    nifty_map = {}
    for _, row in df.iterrows():
        nifty_map[row["time"]] = float(row["perf_60d"]) if pd.notnull(row["perf_60d"]) else 0.001
    return nifty_map


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            elif isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


# -------------------------------------------------------------
# 3. RANK 5 ENTRY DETECTOR & CANDIDATE BUILDER
# -------------------------------------------------------------
def get_rank5_candidates(clean_data, nifty_perf_map):
    """
    Extracts setups adhering strictly to Rank 5:
    - Base cumulative delivery ratio >= 1.5x
    - Demat OBV at 20-day high (Lever B)
    - Micro-launchpad spread squeeze <= 8% (Lever C)
    - Conviction Score >= 2.5 (Lever D1)
    - Stage 2 (Price >= 200 SMA) and 60d RS > Nifty
    """
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
        curr_setup = ""
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
                curr_setup = "V-REVERSAL"
                calc_entry = round(prior_3d_high, 2)
                calc_sl = round(recent_trough * 0.995, 2)

        # ENGINE 2: MICRO-LAUNCHPAD BREAKOUT (LEVER C: SPREAD <= 8%)
        if not entry_triggered:
            valid_launchpad = False
            launchpad_high = 0.0
            launchpad_low = 0.0

            for shelf_len in range(10, 19):
                s_high = highs[i - shelf_len:i].max()
                s_low = lows[i - shelf_len:i].min()
                if s_low > 0:
                    spr = (s_high - s_low) / s_low
                    if spr <= 0.08:  # Lever C
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

                # Lever B: Demat OBV at 20-day high & Base Delivery >= 1.5x
                obv_high = bool(demat_obvs[i] >= demat_obv_maxes[i-1]) if i > 0 else True

                if c_bo and c_vol_bo and c_deliv_bo and c_candle_bo and (base_cumul_ratio >= 1.50) and obv_high:
                    entry_triggered = True
                    curr_setup = "LAUNCHPAD-BO"
                    calc_entry = launchpad_high
                    calc_sl = round(min(l, launchpad_low), 2)

        if entry_triggered:
            r_dist = calc_entry - calc_sl
            if r_dist > 0.05 and (r_dist / calc_entry) <= 0.12:
                is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
                n_perf = nifty_perf_map.get(t)
                if n_perf is None:
                    prior_dates = [d for d in nifty_perf_map.keys() if d <= t]
                    n_perf = nifty_perf_map[max(prior_dates)] if prior_dates else 0.001
                is_rs = bool(stock_p60 > n_perf)

                if is_stage2 and is_rs:
                    deliv_surge_ratio = dv / dv_avg if dv_avg > 0 else 1.0
                    rs_ratio = (stock_p60 / n_perf) if n_perf > 0 else (1.0 + abs(stock_p60))
                    conviction_score = round(deliv_surge_ratio * max(0.1, rs_ratio), 3)

                    # Lever D1: Score >= 2.5
                    if conviction_score >= 2.5:
                        candidates.append({
                            "entry_idx": i,
                            "entry_date": t,
                            "entry_price": calc_entry,
                            "initial_stop": calc_sl,
                            "r_unit": r_dist,
                            "setup": curr_setup,
                            "conviction_score": conviction_score,
                            "highs": highs,
                            "lows": lows,
                            "closes": closes,
                            "ema20s": ema20s,
                            "times": times,
                        })

    return candidates


# -------------------------------------------------------------
# 4. EXIT ARCHITECTURE SIMULATORS
# -------------------------------------------------------------
def simulate_exit(cand, mode="breakeven_then_trail"):
    """
    mode options:
      1. 'breakeven_then_trail': Move stop to BE at +1.5R, then trail 20 EMA.
      2. 'pure_structural_trail': Never move stop to BE. Stop stays at initial_stop until 20 EMA overtakes it.
      3. 'book_50_at_1_5r': Book 50% at +1.5R, move remaining 50% stop to BE, trail remainder on 20 EMA.
    """
    entry_idx = cand["entry_idx"]
    entry_p = cand["entry_price"]
    initial_sl = cand["initial_stop"]
    r_unit = cand["r_unit"]
    highs = cand["highs"]
    closes = cand["closes"]
    ema20s = cand["ema20s"]
    times = cand["times"]

    be_target = entry_p + (1.5 * r_unit)
    current_sl = initial_sl
    moved_to_be = False
    half_booked = False
    booked_pnl_pct = 0.0

    for i in range(entry_idx + 1, len(closes)):
        h = highs[i]
        c = closes[i]
        ema = ema20s[i]

        if mode == "breakeven_then_trail":
            if not moved_to_be and h >= be_target:
                current_sl = max(current_sl, entry_p)
                moved_to_be = True
            if moved_to_be:
                current_sl = max(current_sl, round(float(ema), 2))

            if c < current_sl:
                exit_p = round(current_sl, 2)
                pnl_pts = exit_p - entry_p
                pnl_pct = round((pnl_pts / entry_p) * 100.0, 2)
                r_mult = round(pnl_pts / r_unit, 2)
                return {
                    "entry_date": cand["entry_date"],
                    "exit_date": times[i],
                    "duration": i - entry_idx,
                    "pnl_pct": pnl_pct,
                    "r_mult": r_mult,
                    "outcome": "WIN" if pnl_pts > 0 else "LOSS"
                }

        elif mode == "pure_structural_trail":
            # Never shift early to breakeven. Only trail via 20 EMA once it moves above initial stop.
            current_sl = max(initial_sl, round(float(ema), 2))

            if c < current_sl:
                exit_p = round(current_sl, 2)
                pnl_pts = exit_p - entry_p
                pnl_pct = round((pnl_pts / entry_p) * 100.0, 2)
                r_mult = round(pnl_pts / r_unit, 2)
                return {
                    "entry_date": cand["entry_date"],
                    "exit_date": times[i],
                    "duration": i - entry_idx,
                    "pnl_pct": pnl_pct,
                    "r_mult": r_mult,
                    "outcome": "WIN" if pnl_pts > 0 else "LOSS"
                }

        elif mode == "book_50_at_1_5r":
            if not half_booked and h >= be_target:
                half_booked = True
                booked_pnl_pct = round(((be_target - entry_p) / entry_p) * 100.0, 2)
                current_sl = max(current_sl, entry_p)

            if half_booked:
                current_sl = max(current_sl, round(float(ema), 2))

            if c < current_sl:
                exit_p = round(current_sl, 2)
                rem_pnl_pct = round(((exit_p - entry_p) / entry_p) * 100.0, 2)

                if half_booked:
                    final_pnl_pct = round(0.5 * booked_pnl_pct + 0.5 * rem_pnl_pct, 2)
                    final_r_mult = round(0.5 * 1.5 + 0.5 * ((exit_p - entry_p) / r_unit), 2)
                else:
                    final_pnl_pct = rem_pnl_pct
                    final_r_mult = round((exit_p - entry_p) / r_unit, 2)

                return {
                    "entry_date": cand["entry_date"],
                    "exit_date": times[i],
                    "duration": i - entry_idx,
                    "pnl_pct": final_pnl_pct,
                    "r_mult": final_r_mult,
                    "outcome": "WIN" if final_pnl_pct > 0 else "LOSS"
                }

    # If position still active at end of historical data
    last_idx = len(closes) - 1
    last_c = closes[last_idx]
    last_pnl_pct = round(((last_c - entry_p) / entry_p) * 100.0, 2)
    last_r_mult = round((last_c - entry_p) / r_unit, 2)
    return {
        "entry_date": cand["entry_date"],
        "exit_date": times[last_idx],
        "duration": last_idx - entry_idx,
        "pnl_pct": last_pnl_pct,
        "r_mult": last_r_mult,
        "outcome": "WIN" if last_pnl_pct > 0 else "LOSS"
    }


# -------------------------------------------------------------
# 5. METRICS HELPER
# -------------------------------------------------------------
def compute_metrics(trades_list, label):
    if not trades_list:
        return {
            "Configuration": label,
            "Trades": 0, "Win Rate %": 0.0, "Profit Factor": 0.0,
            "Avg Gain %": 0.0, "Avg Loss %": 0.0, "Max Profit %": 0.0,
            "Avg Hold (Days)": 0.0
        }

    df = pd.DataFrame(trades_list)
    total = len(df)
    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]

    win_rate = round((len(wins) / total) * 100.0, 2)
    avg_gain = round(wins["pnl_pct"].mean(), 2) if not wins.empty else 0.0
    avg_loss = round(losses["pnl_pct"].mean(), 2) if not losses.empty else 0.0
    max_gain = round(df["pnl_pct"].max(), 2) if not df.empty else 0.0
    avg_duration = round(df["duration"].mean(), 1)

    gross_win = wins["pnl_pct"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl_pct"].sum()) if not losses.empty else 1.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "Configuration": label,
        "Trades": total,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,
        "Avg Gain %": avg_gain,
        "Avg Loss %": avg_loss,
        "Max Profit %": max_gain,
        "Avg Hold (Days)": avg_duration
    }


# -------------------------------------------------------------
# 6. RUNNER FOR EXIT STUDY
# -------------------------------------------------------------
def run_exit_study():
    nifty_perf_map = load_nifty_perf_map()
    nifty750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "gap_margin_candidates.json", "swing3_results.json",
        "backtest_results.json", "swing3_compare.json", "swing3_run2_results.json",
        "swing3_run2_compare_results.json", "swing3_run3_results.json",
        "swing3_target400_results.json", "swing3_exit_study_results.json"
    }
    target_files = [f for f in json_files if os.path.basename(f).lower() not in excluded]

    print(f"🚀 Scanning {len(target_files)} symbols in Nifty 750 for Swing 3.0 (Rank 5 Model)...")
    all_raw_candidates = []

    for path in sorted(target_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym not in nifty750_set:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if clean:
                cands = get_rank5_candidates(clean, nifty_perf_map)
                for c in cands:
                    c["symbol"] = sym
                all_raw_candidates.extend(cands)
        except Exception:
            continue

    if not all_raw_candidates:
        print("⚠️ No qualifying candidates found.")
        return

    # Apply Max 1/day by Conviction Score (Identical Trade Set across all 3 exit tests)
    cands_by_date = {}
    for c in all_raw_candidates:
        cands_by_date.setdefault(c["entry_date"], []).append(c)

    filtered_candidates = []
    for d, d_cands in sorted(cands_by_date.items()):
        d_cands.sort(key=lambda x: x["conviction_score"], reverse=True)
        filtered_candidates.append(d_cands[0])

    print(f"✅ Filtered candidates ready: {len(filtered_candidates)} trades for exit simulation.")

    # 1. Variant 1: Move SL to Breakeven @ +1.5R and Trail EMA
    t_v1 = [simulate_exit(c, "breakeven_then_trail") for c in filtered_candidates]
    m_v1 = compute_metrics(t_v1, "1. Move SL to BE @ 1.5R, then Trail 20 EMA")

    # 2. Variant 2: Never Move to BE (Pure Structural Hold + 20 EMA Trail)
    t_v2 = [simulate_exit(c, "pure_structural_trail") for c in filtered_candidates]
    m_v2 = compute_metrics(t_v2, "2. Never Move to BE (Hold Pivot SL, Trail EMA)")

    # 3. Variant 3: Book 50% @ 1.5R, Move Remainder to BE, Trail 50%
    t_v3 = [simulate_exit(c, "book_50_at_1_5r") for c in filtered_candidates]
    m_v3 = compute_metrics(t_v3, "3. Book 50% @ 1.5R + Trail 50% on EMA")

    report = [m_v1, m_v2, m_v3]
    df_report = pd.DataFrame(report)
    df_sorted = df_report.sort_values(by=["Profit Factor", "Win Rate %"], ascending=[False, False]).reset_index(drop=True)
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    print("\n" + "=" * 118)
    print("🏆 SWING 3.0 (RANK 5 MODEL): EXIT ARCHITECTURE COMPARISON STUDY")
    print("=" * 118)
    print(df_sorted.to_string(index=False))
    print("=" * 118 + "\n")

    payload = {
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Report": df_sorted.to_dict(orient="records"),
        "Sample Trades (Variant 1)": t_v1[-50:],
        "Sample Trades (Variant 2)": t_v2[-50:],
        "Sample Trades (Variant 3)": t_v3[-50:],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    run_exit_study()
