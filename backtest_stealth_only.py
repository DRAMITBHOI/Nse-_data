import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "stealth_backtest_report.json")
MAX_RISK_PCT = 10.0
MAX_HOLD_DAYS = 60
MAX_PE = 35.0

def clean_data_fast(raw_data):
    """Blazing-fast string-based data extraction without slow pd.to_datetime overhead."""
    if not raw_data or not isinstance(raw_data, list):
        return []
    
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = r.get("time", "")
        if not raw_t:
            continue
        
        # Pure string slice 'YYYY-MM-DD' (0.01 microsecond vs 50 microseconds for pd.to_datetime)
        d_str = str(raw_t)[:10]
        c = float(r.get("close", 0) or 0)
        if c <= 0:
            continue

        v = float(r.get("volume", 0) or 0)
        o = float(r.get("open", c) or c)
        h = float(r.get("high", c) or c)
        l = float(r.get("low", c) or c)

        if d_str not in date_map or v > date_map[d_str]["volume"]:
            date_map[d_str] = {
                "time": d_str,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v
            }

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 50:
        return []

    clean = [date_map[k] for k in sorted_dates]

    # Holiday twin candle deduplication
    deduped = []
    for r in clean:
        if deduped:
            prev = deduped[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    deduped.pop()
        deduped.append(r)

    # Split adjustments
    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(deduped) - 1, 0, -1):
        prev_c = deduped[i - 1]["close"]
        curr_o = deduped[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj = k
                        break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
                elif not adj and 8.50 <= ratio <= 11.50: adj = 10.0
            if adj:
                for j in range(0, i):
                    deduped[j]["open"] = round(deduped[j]["open"] / adj, 2)
                    deduped[j]["high"] = round(deduped[j]["high"] / adj, 2)
                    deduped[j]["low"] = round(deduped[j]["low"] / adj, 2)
                    deduped[j]["close"] = round(deduped[j]["close"] / adj, 2)
                    deduped[j]["volume"] = deduped[j]["volume"] * adj

    return deduped

def load_benchmark():
    for f_name in ["NIFTY50.json", "NIFTY.json", "RELIANCE.json"]:
        path = os.path.join(DATA_DIR, f_name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                c = clean_data_fast(raw)
                if c:
                    return {r["time"]: r["close"] for r in c}
            except Exception:
                pass
    return {}

def fast_rolling_mean(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    result = np.empty_like(arr, dtype=float)
    result[:window-1] = np.nan
    result[window-1:] = ret[window-1:] / window
    # Forward fill the initial nan values for safe division
    first_val = result[window-1] if len(result) >= window else 1.0
    result[:window-1] = first_val
    return result

def run_backtest():
    print("🚀 Running High-Speed Stealth Accumulation Backtest...")
    benchmark_map = load_benchmark()
    if benchmark_map:
        print(f"✅ Loaded {len(benchmark_map)} benchmark dates for RS comparison.")

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
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "scanA_results.json",
            "scanB_backtest_report.json", "stealth_backtest_report.json"
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
        if len(clean) < 65:
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        opens = np.array([r["open"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # 1. Vectorized ATR Computation
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        atr10 = fast_rolling_mean(tr, 10)
        atr50 = fast_rolling_mean(tr, 50)
        natr10 = (atr10 / np.maximum(closes, 1e-4)) * 100.0
        natr50 = (atr50 / np.maximum(closes, 1e-4)) * 100.0

        # 2. Vectorized Volume SMAs
        vol_sma10 = fast_rolling_mean(volumes, 10)
        vol_sma20 = fast_rolling_mean(volumes, 20)
        vol_sma50 = fast_rolling_mean(volumes, 50)

        # 3. Vectorized Lower Wick Absorption
        candle_ranges = np.maximum(highs - lows, 1e-4)
        lower_wicks = np.minimum(opens, closes) - lows
        wick_absorption = ((lower_wicks / candle_ranges) >= 0.45) & (closes >= opens)
        wick_abs_int = wick_absorption.astype(float)
        rolling_wick_count = fast_rolling_mean(wick_abs_int, 15) * 15.0

        cooldown_idx = 0

        for i in range(50, N - 1):
            if i < cooldown_idx:
                continue

            # Criteria Checks
            vcp_ok = (natr50[i] > 0 and (natr10[i] / natr50[i]) <= 0.72) and \
                     (vol_sma50[i] > 0 and vol_sma10[i] <= (0.85 * vol_sma50[i]))
            
            wicks_ok = rolling_wick_count[i] >= 3.0

            d_now = times[i]
            d_past = times[i - 20]
            rs_ok = True
            if benchmark_map and d_now in benchmark_map and d_past in benchmark_map:
                stk_chg = (closes[i] - closes[i - 20]) / closes[i - 20]
                idx_chg = (benchmark_map[d_now] - benchmark_map[d_past]) / benchmark_map[d_past]
                rs_ok = stk_chg >= (idx_chg + 0.025)

            prior_20_low = np.min(lows[i - 20 : i])
            prior_20_high = np.max(highs[i - 20 : i])
            tightness = ((prior_20_high - prior_20_low) / max(prior_20_low, 1e-4)) * 100.0

            if vcp_ok and wicks_ok and rs_ok and (tightness <= 12.0):
                # Breakout Trigger
                if closes[i] > prior_20_high and closes[i - 1] <= prior_20_high:
                    if vol_sma20[i] > 0 and volumes[i] >= (1.40 * vol_sma20[i]):
                        entry_price = closes[i]
                        entry_date = times[i]

                        # Swing low anchor (prior 12 bars)
                        recent_swing_low = float(np.min(lows[i - 12 : i]))
                        sl_price = round(recent_swing_low * 0.995, 2)
                        risk_pct = ((entry_price - sl_price) / entry_price) * 100.0

                        # STRICT RISK FILTER: Skip if > 10%
                        if risk_pct > MAX_RISK_PCT or risk_pct <= 0:
                            continue

                        # 60-Day Forward Performance
                        fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                        fwd_highs = highs[i + 1 : fwd_end]
                        fwd_lows = lows[i + 1 : fwd_end]

                        max_gain = 0.0
                        hit_15 = False
                        hit_20 = False
                        hit_25 = False
                        hit_30 = False
                        stopped_out = False
                        partial_booked = False
                        active_sl = sl_price
                        days_held = len(fwd_highs)

                        for d_idx in range(len(fwd_highs)):
                            h_bar = fwd_highs[d_idx]
                            l_bar = fwd_lows[d_idx]

                            gain = ((h_bar - entry_price) / entry_price) * 100.0
                            if gain > max_gain:
                                max_gain = gain

                            if max_gain >= 15.0: hit_15 = True
                            if max_gain >= 20.0: hit_20 = True
                            if max_gain >= 25.0: hit_25 = True
                            if max_gain >= 30.0: hit_30 = True

                            # 50% partial book at +15% & SL to BE
                            if max_gain >= 15.0 and not partial_booked:
                                partial_booked = True
                                active_sl = entry_price

                            # Trailing stop on remainder
                            if partial_booked and d_idx >= 10:
                                trail = float(np.min(lows[i + 1 + d_idx - 10 : i + 1 + d_idx]))
                                if trail > active_sl:
                                    active_sl = trail

                            if l_bar <= active_sl:
                                stopped_out = True
                                days_held = d_idx + 1
                                break

                        all_trades.append({
                            "Symbol": sym,
                            "Entry Date": entry_date,
                            "Entry Price": round(entry_price, 2),
                            "Swing SL": sl_price,
                            "Risk %": round(risk_pct, 1),
                            "Base Tightness %": round(tightness, 1),
                            "Max Gain in 60D %": round(max_gain, 2),
                            "Hit +15%": hit_15,
                            "Hit +20%": hit_20,
                            "Hit +25%": hit_25,
                            "Hit +30%+": hit_30,
                            "Stopped Out": stopped_out,
                            "Evaluated Days": days_held
                        })
                        cooldown_idx = i + max(10, days_held)

    df_res = pd.DataFrame(all_trades)
    total = len(df_res)
    print(f"\n📊 Total Stealth Setups Found: {total}")

    if total == 0:
        print("⚠️ No trades found matching criteria.")
        return

    m15 = round((df_res["Hit +15%"].sum() / total) * 100, 2)
    m20 = round((df_res["Hit +20%"].sum() / total) * 100, 2)
    m25 = round((df_res["Hit +25%"].sum() / total) * 100, 2)
    m30 = round((df_res["Hit +30%+"].sum() / total) * 100, 2)
    stop_rate = round((df_res["Stopped Out"].sum() / total) * 100, 2)
    avg_gain = round(float(df_res["Max Gain in 60D %"].mean()), 2)
    avg_risk = round(float(df_res["Risk %"].mean()), 1)

    summary = {
        "Total Trades Executed": total,
        "Average Risk % (Swing SL)": f"{avg_risk}%",
        "+15% Target Hit Rate": f"{m15}% ({df_res['Hit +15%'].sum()}/{total})",
        "+20% Target Hit Rate": f"{m20}% ({df_res['Hit +20%'].sum()}/{total})",
        "+25% Target Hit Rate": f"{m25}% ({df_res['Hit +25%'].sum()}/{total})",
        "+30%+ Target Hit Rate": f"{m30}% ({df_res['Hit +30%+'].sum()}/{total})",
        "Stop Out Rate (Initial / Trail)": f"{stop_rate}%",
        "Average 60-Day Forward Expansion": f"+{avg_gain}%"
    }

    report_payload = {
        "Summary": summary,
        "Recent 25 Trades": all_trades[-25:] if len(all_trades) >= 25 else all_trades
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(report_payload, fp, indent=2)

    print("="*60)
    for k, v in summary.items():
        print(f"{k:<35}: {v}")
    print("="*60)
    print(f"📁 Report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_backtest()
