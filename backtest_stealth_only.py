import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "stealth_backtest_report.json")
MAX_RISK_PCT = 10.0
MAX_HOLD_DAYS = 60
MAX_PE = 35.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def clean_and_prepare_dataset(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()
        if not raw_t:
            continue
        try:
            dt = pd.to_datetime(raw_t)
            if dt.dayofweek >= 5:
                continue
            d_str = dt.strftime("%Y-%m-%d")
            c = float(r.get("close", 0))
            if c <= 0:
                continue
            
            entry = {
                "time": d_str,
                "open": float(r.get("open", c)),
                "high": float(r.get("high", c)),
                "low": float(r.get("low", c)),
                "close": c,
                "volume": float(r.get("volume", 0) or 0)
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]
    if len(clean) < 40:
        return []

    # Filter duplicate holiday candles
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

    # Corporate split multiplier adjustment
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
                    filtered[j]["volume"] = filtered[j]["volume"] * adj_factor

    return filtered

def load_benchmark():
    for f_name in ["NIFTY50.json", "NIFTY.json", "RELIANCE.json"]:
        path = os.path.join(DATA_DIR, f_name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                df = pd.DataFrame(clean_and_prepare_dataset(raw))
                if not df.empty and "close" in df.columns:
                    return df.set_index("time")["close"].to_dict()
            except Exception:
                pass
    return {}

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    return pd.Series(tr).rolling(period, min_periods=1).mean().values

def run_backtest():
    print("🚀 Running Pure Stealth Accumulation Backtest (Swing Low SL <= 10%)...")
    benchmark_map = load_benchmark()

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
        if processed % 300 == 0:
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

        clean = clean_and_prepare_dataset(raw)
        if len(clean) < 65:
            continue

        df = pd.DataFrame(clean)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        volumes = df["volume"].values
        times = df["time"].values
        N = len(closes)

        # 1. Volatility Contraction Indicators
        atr10 = compute_atr(highs, lows, closes, 10)
        atr50 = compute_atr(highs, lows, closes, 50)
        natr10 = (atr10 / np.maximum(closes, 1e-4)) * 100.0
        natr50 = (atr50 / np.maximum(closes, 1e-4)) * 100.0
        vol_sma10 = pd.Series(volumes).rolling(10, min_periods=1).mean().values
        vol_sma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        vol_sma50 = pd.Series(volumes).rolling(50, min_periods=1).mean().values

        # 2. Lower Wick Absorption Candles
        candle_ranges = np.maximum(highs - lows, 1e-4)
        lower_wicks = np.minimum(opens, closes) - lows
        wick_absorption = ((lower_wicks / candle_ranges) >= 0.45) & (closes >= opens)
        rolling_wick_count = pd.Series(wick_absorption.astype(int)).rolling(15, min_periods=1).sum().values

        cooldown_idx = 0

        for i in range(50, N - 1):
            if i < cooldown_idx:
                continue

            # Stealth Criteria over 20 sessions
            # A. VCP contraction & supply dry up
            vcp_ok = (natr50[i] > 0 and (natr10[i] / natr50[i]) <= 0.72) and \
                     (vol_sma50[i] > 0 and vol_sma10[i] <= (0.85 * vol_sma50[i]))

            # B. Passive Lower Shadow Absorption
            wicks_ok = rolling_wick_count[i] >= 3

            # C. Relative Strength Divergence vs Nifty 50
            d_now = times[i]
            d_past = times[i - 20]
            rs_ok = True
            if benchmark_map and d_now in benchmark_map and d_past in benchmark_map:
                stk_chg = (closes[i] - closes[i - 20]) / closes[i - 20]
                idx_chg = (benchmark_map[d_now] - benchmark_map[d_past]) / benchmark_map[d_past]
                rs_ok = stk_chg >= (idx_chg + 0.025)

            # D. Range Tightness <= 12%
            prior_20_low = np.min(lows[i - 20 : i])
            prior_20_high = np.max(highs[i - 20 : i])
            tightness = ((prior_20_high - prior_20_low) / max(prior_20_low, 1e-4)) * 100.0
            tight_ok = tightness <= 12.0

            if vcp_ok and wicks_ok and rs_ok and tight_ok:
                # Breakout Trigger: Price crosses prior 20D high with volume expansion
                if closes[i] > prior_20_high and closes[i - 1] <= prior_20_high:
                    if vol_sma20[i] > 0 and volumes[i] >= (1.40 * vol_sma20[i]):
                        entry_price = closes[i]
                        entry_date = times[i]

                        # Swing low stop: lowest low of pre-break base (lookback 12 bars)
                        recent_swing_low = float(np.min(lows[i - 12 : i]))
                        sl_price = round(recent_swing_low * 0.995, 2)
                        risk_pct = ((entry_price - sl_price) / entry_price) * 100.0

                        # STRICT RISK FILTER: Avoid entry if SL > 10%
                        if risk_pct > MAX_RISK_PCT or risk_pct <= 0:
                            continue

                        # Forward 60-day performance tracking
                        fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                        fwd_highs = highs[i + 1 : fwd_end]
                        fwd_lows = lows[i + 1 : fwd_end]
                        fwd_closes = closes[i + 1 : fwd_end]

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

                            # 50% Profit Booking & Move SL to Breakeven
                            if max_gain >= 15.0 and not partial_booked:
                                partial_booked = True
                                active_sl = entry_price

                            # Dynamic 15D Trailing Stop after +15%
                            if partial_booked and d_idx >= 10:
                                trail_15 = float(np.min(lows[i + 1 + d_idx - 10 : i + 1 + d_idx]))
                                if trail_15 > active_sl:
                                    active_sl = trail_15

                            # Stop hit
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
                            "Partial Booked": partial_booked,
                            "Evaluated Days": days_held
                        })
                        cooldown_idx = i + max(10, days_held)

    df_res = pd.DataFrame(all_trades)
    total = len(df_res)
    print(f"\n📊 Backtest Complete! Total Qualified Stealth Trades: {total}")

    if total == 0:
        print("⚠️ No trades matched.")
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
        "Stop Out Rate (Initial or Trail)": f"{stop_rate}%",
        "Average 60-Day Forward Expansion": f"+{avg_gain}%"
    }

    report_payload = {
        "Summary": summary,
        "Recent 25 Trades": all_trades[-25:] if len(all_trades) >= 25 else all_trades
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(report_payload, fp, indent=2)

    print("\n" + "="*65)
    print("🎯 PURE STEALTH ACCUMULATION BACKTEST RESULTS")
    print("="*65)
    for k, v in summary.items():
        print(f"{k:<35}: {v}")
    print("="*65)
    print(f"📁 Report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_backtest()
