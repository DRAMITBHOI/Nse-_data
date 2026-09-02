import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "integrated_institutional_report.json")
MAX_PE = 35.0
MAX_RISK_PCT = 10.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def get_nifty_750_universe():
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fp:
                data = json.load(fp)
                if data:
                    return set(data)
        except Exception:
            pass

    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]
    symbols = set()
    for u in urls:
        try:
            req = urllib.request.Request(u, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Warning fetching universe from {u}: {e}")

    sorted_list = sorted(list(symbols))
    if sorted_list:
        try:
            with open(local_path, "w") as fp:
                json.dump(sorted_list, fp, indent=2)
        except Exception:
            pass
    return set(sorted_list)

def clean_data_fast(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = r.get("time", "")
        if not raw_t:
            continue
        d_str = str(raw_t)[:10]
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
            "deliv_pct": float(r.get("deliv_pct", 0) or 0)
        }
        if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
            date_map[d_str] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 50:
        return []

    clean = [date_map[k] for k in sorted_dates]

    filtered = []
    for r in clean:
        if filtered:
            prev = filtered[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    filtered.pop()
        filtered.append(r)

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(filtered) - 1, 0, -1):
        prev_c = filtered[i - 1]["close"]
        curr_o = filtered[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj_factor = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj_factor = k
                        break
                if not adj_factor and 1.70 <= ratio <= 2.30: adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50: adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50: adj_factor = 10.0
            if adj_factor:
                for j in range(0, i):
                    filtered[j]["open"] = round(filtered[j]["open"] / adj_factor, 2)
                    filtered[j]["high"] = round(filtered[j]["high"] / adj_factor, 2)
                    filtered[j]["low"] = round(filtered[j]["low"] / adj_factor, 2)
                    filtered[j]["close"] = round(filtered[j]["close"] / adj_factor, 2)
                    filtered[j]["delivery_vol"] = filtered[j]["delivery_vol"] * adj_factor
                    filtered[j]["volume"] = filtered[j]["volume"] * adj_factor

    running_vol = 50000.0
    for i in range(len(filtered)):
        v = filtered[i]["volume"]
        dv = filtered[i]["delivery_vol"]
        pct = filtered[i]["deliv_pct"]

        if v > 0:
            running_vol = 0.9 * running_vol + 0.1 * v
        else:
            filtered[i]["volume"] = running_vol
            v = running_vol

        if dv <= 0:
            filtered[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            filtered[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            filtered[i]["delivery_vol"] = v
            filtered[i]["deliv_pct"] = 100.0

    return filtered

def load_benchmark_and_regime():
    for f_name in ["NIFTY50.json", "NIFTY.json", "RELIANCE.json"]:
        path = os.path.join(DATA_DIR, f_name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                c = clean_data_fast(raw)
                if len(c) >= 50:
                    df = pd.DataFrame(c)
                    df["sma50"] = df["close"].rolling(50, min_periods=20).mean()
                    b_map = {}
                    regime_map = {}
                    for _, r in df.iterrows():
                        d = r["time"]
                        cls = float(r["close"])
                        sma = float(r["sma50"]) if not np.isnan(r["sma50"]) else cls
                        b_map[d] = cls
                        regime_map[d] = "Favourable" if cls >= sma else "Unfavourable"
                    return b_map, regime_map
            except Exception:
                pass
    return {}, {}

def fast_rolling_mean(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    result = np.empty_like(arr, dtype=float)
    result[:window-1] = np.nan
    result[window-1:] = ret[window-1:] / window
    first_val = result[window-1] if len(result) >= window else 1.0
    result[:window-1] = first_val
    return result

def run_integrated_backtest():
    print("🚀 Running Dual Institutional Backtest Engine...")
    nifty_750_set = get_nifty_750_universe()
    benchmark_map, nifty_regime = load_benchmark_and_regime()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as fp:
                fundamentals = json.load(fp)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "backtest_report.json", "segmented_backtest_report.json",
            "scanB_backtest_report.json", "integrated_institutional_report.json",
            "stealth_backtest_report.json", "wyckoff_screener_results.json",
            "active_trade_plan.json", "scanA_results.json",
            "nifty750.json", "NIFTY50.json", "NIFTY.json"
        ]
    ]

    all_trades = []
    processed = 0

    for f_name in stock_files:
        processed += 1
        if processed % 500 == 0:
            print(f"⏳ Processed {processed}/{len(stock_files)} stocks...")

        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        if pe_val is not None:
            try:
                pe_float = float(pe_val)
                if pe_float <= 0 or pe_float >= MAX_PE:
                    continue
            except (ValueError, TypeError):
                pass

        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception:
            continue

        clean = clean_data_fast(raw)
        if len(clean) < 60:
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        opens = np.array([r["open"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
        d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # Demat Delivery OBV
        cur_obv = 0.0
        obvs = []
        for i in range(N):
            dv = d_vols[i]
            if i > 0:
                if closes[i] > closes[i - 1]: cur_obv += dv
                elif closes[i] < closes[i - 1]: cur_obv -= dv
            else:
                cur_obv = dv
            obvs.append(cur_obv)
        obvs = np.array(obvs, dtype=float)

        turnover_cr = (closes * volumes) / 1e7
        to_50 = fast_rolling_mean(turnover_cr, 50)
        pct_50 = fast_rolling_mean(pcts, 50)
        deliv_sma20 = fast_rolling_mean(d_vols, 20)
        gross_sma20 = fast_rolling_mean(volumes, 20)

        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        natr10 = (fast_rolling_mean(tr, 10) / np.maximum(closes, 1e-4)) * 100.0
        natr50 = (fast_rolling_mean(tr, 50) / np.maximum(closes, 1e-4)) * 100.0
        vol_sma10 = fast_rolling_mean(volumes, 10)
        vol_sma50 = fast_rolling_mean(volumes, 50)

        candle_ranges = np.maximum(highs - lows, 1e-4)
        lower_wicks = np.minimum(opens, closes) - lows
        wick_absorption = ((lower_wicks / candle_ranges) >= 0.45) & (closes >= opens)
        rolling_absorption_count = fast_rolling_mean(wick_absorption.astype(float), 15) * 15.0

        is_n750 = sym in nifty_750_set

        in_trade = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = ""
        active_sl = 0.0
        active_bucket = ""
        setup_type = ""
        max_run_gain = 0.0
        partial_booked = False
        cooldown_until = 0

        for i in range(50, N):
            curr_to = to_50[i] if not np.isnan(to_50[i]) else 0.0

            if is_n750:
                if curr_to >= 30.0:
                    tier = "Bucket A (>30 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 3, 45
                elif curr_to >= 5.0:
                    tier = "Bucket B (5-30 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.2, 1.0, 2, 15
                else:
                    tier = "Bucket C (<5 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 4, 20
            else:
                tier = "Bucket C (<5 Cr)"
                pct_m, vol_m, min_c, base_w = 1.4, 1.0, 4, 20

            if in_trade:
                gain = ((highs[i] - entry_price) / entry_price) * 100.0
                if gain > max_run_gain:
                    max_run_gain = gain

                if max_run_gain >= 15.0 and not partial_booked:
                    partial_booked = True
                    active_sl = entry_price

                if "Bucket C" in active_bucket:
                    if max_run_gain >= 15.0 and i >= entry_idx + 10:
                        trail_15 = float(np.min(lows[i - 15 : i]))
                        if trail_15 > active_sl: active_sl = trail_15
                elif "Bucket B" in active_bucket:
                    if max_run_gain >= 20.0 and i >= entry_idx + 20:
                        trail_20 = float(np.min(lows[i - 20 : i]))
                        if trail_20 > active_sl: active_sl = trail_20
                else:
                    if max_run_gain >= 25.0 and i >= entry_idx + 30:
                        trail_30 = float(np.min(lows[i - 30 : i]))
                        if trail_30 > active_sl: active_sl = trail_30

                exit_triggered = False
                exit_price = closes[i]

                if lows[i] <= active_sl:
                    exit_triggered = True
                    exit_price = min(closes[i], active_sl)
                elif "Bucket C" in active_bucket and i > entry_idx + 2:
                    if gross_sma20[i] > 0 and pct_50[i] > 0:
                        if volumes[i] >= (1.5 * gross_sma20[i]) and pcts[i] <= (0.70 * pct_50[i]) and closes[i] <= opens[i]:
                            exit_triggered = True
                            exit_price = closes[i]
                elif "Bucket A" in active_bucket and (i - entry_idx) >= 30 and max_run_gain < 8.0:
                    exit_triggered = True
                    exit_price = closes[i]

                if exit_triggered:
                    base_ret = ((exit_price - entry_price) / entry_price) * 100.0
                    final_ret = round((15.0 * 0.50) + (base_ret * 0.50), 2) if partial_booked else round(base_ret, 2)
                    all_trades.append({
                        "Symbol": sym,
                        "Tier": active_bucket,
                        "Setup Type": setup_type,
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": times[i],
                        "Exit Price": round(exit_price, 2),
                        "Return %": final_ret,
                        "Max Run Gain %": round(max_run_gain, 2),
                        "Rally 20%": bool(max_run_gain >= 20.0),
                        "Holding Days": i - entry_idx,
                        "Is Win": bool(final_ret > 0)
                    })
                    in_trade = False
                    cooldown_until = i + 5
                    continue

            if not in_trade and i > cooldown_until and i >= base_w:
                base_start = i - base_w
                
                qualifying_dots = (pcts[base_start:i] >= (pct_m * pct_50[base_start:i])) & \
                                  (d_vols[base_start:i] >= (vol_m * deliv_sma20[base_start:i]))
                is_delivery_cluster = bool(np.sum(qualifying_dots) >= min_c)

                vcp_contracted = (natr50[i] > 0 and (natr10[i] / natr50[i]) <= 0.72) and \
                                 (vol_sma50[i] > 0 and vol_sma10[i] <= (0.85 * vol_sma50[i]))
                wick_absorption_present = bool(rolling_absorption_count[i] >= 3.0)
                
                d_now = times[i]
                d_past = times[i - 20]
                rs_divergent = True
                if benchmark_map and d_now in benchmark_map and d_past in benchmark_map:
                    stk_perf = (closes[i] - closes[i - 20]) / closes[i - 20]
                    idx_perf = (benchmark_map[d_now] - benchmark_map[d_past]) / benchmark_map[d_past]
                    rs_divergent = stk_perf >= (idx_perf + 0.025)

                prior_20_low = np.min(lows[i - 20 : i])
                prior_20_high = np.max(highs[i - 20 : i])
                range_tight = (((prior_20_high - prior_20_low) / max(prior_20_low, 1e-4)) * 100.0) <= 12.0
                is_stealth_setup = vcp_contracted and wick_absorption_present and rs_divergent and range_tight

                if is_delivery_cluster or is_stealth_setup:
                    base_highs = highs[base_start:i]
                    sw_idx = int(np.argmax(base_highs))
                    sw_high = base_highs[sw_idx]
                    sw_obv = obvs[base_start + sw_idx]

                    if closes[i] > sw_high and closes[i - 1] <= sw_high and obvs[i] > sw_obv:
                        entry_cand = closes[i]
                        pre_lookback = min(12, i - base_start)
                        recent_swing_low = float(np.min(lows[i - pre_lookback : i]))
                        sl_cand = round(recent_swing_low * 0.995, 2)
                        risk_pct = ((entry_cand - sl_cand) / entry_cand) * 100.0

                        if 0 < risk_pct <= MAX_RISK_PCT:
                            active_bucket = tier
                            in_trade = True
                            entry_idx = i
                            entry_date = times[i]
                            entry_price = entry_cand
                            active_sl = sl_cand
                            max_run_gain = 0.0
                            partial_booked = False
                            
                            if is_delivery_cluster and is_stealth_setup:
                                setup_type = "Dual (Cluster + Stealth)"
                            elif is_delivery_cluster:
                                setup_type = "Delivery Cluster"
                            else:
                                setup_type = "Stealth Absorption"

    df_trades = pd.DataFrame(all_trades)
    print(f"📊 Total Dual Strategy Trades: {len(df_trades)}")

    def compute_stats(df_sub):
        if df_sub.empty:
            return {"Trades": 0, "Win Rate %": "0%", "+20% Rally %": "0%", "Avg Return %": "0%", "Profit Factor": 0.0, "Avg Hold": "0 d"}
        total_t = len(df_sub)
        wins = len(df_sub[df_sub["Is Win"] == True])
        win_rate = round((wins / total_t) * 100, 1)
        r20 = round((len(df_sub[df_sub["Rally 20%"] == True]) / total_t) * 100, 1)
        avg_ret = round(float(df_sub["Return %"].mean()), 2)
        
        gross_win = df_sub[df_sub["Is Win"] == True]["Return %"].sum()
        gross_loss = abs(df_sub[df_sub["Is Win"] == False]["Return %"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_h = round(float(df_sub["Holding Days"].mean()), 1)

        return {
            "Trades": total_t,
            "Win Rate %": f"{win_rate}%",
            "+20% Rally %": f"{r20}%",
            "Avg Return %": f"{'+' if avg_ret >= 0 else ''}{avg_ret}%",
            "Profit Factor": pf,
            "Avg Hold": f"{avg_h} d"
        }

    tier_summary = {}
    for b in ["Bucket A (>30 Cr)", "Bucket B (5-30 Cr)", "Bucket C (<5 Cr)"]:
        sub = df_trades[df_trades["Tier"] == b] if not df_trades.empty else pd.DataFrame()
        tier_summary[b] = compute_stats(sub)

    setup_summary = {}
    for st in ["Delivery Cluster", "Stealth Absorption", "Dual (Cluster + Stealth)"]:
        sub = df_trades[df_trades["Setup Type"] == st] if not df_trades.empty else pd.DataFrame()
        setup_summary[st] = compute_stats(sub)

    final_payload = {
        "Overall Summary": compute_stats(df_trades),
        "Tier Breakdown": tier_summary,
        "Setup Type Comparison": setup_summary,
        "Recent Trades": all_trades[-25:] if len(all_trades) >= 25 else all_trades
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(final_payload, fp, indent=2)

    print(f"🎉 Integrated Report saved to '{OUTPUT_REPORT}'.")

if __name__ == "__main__":
    run_integrated_backtest()
