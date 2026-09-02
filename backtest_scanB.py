import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "scanB_backtest_report.json")
MAX_PE = 35.0
MAX_HOLD_DAYS = 60
STOP_LOSS_PCT = 10.0

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
            with urllib.request.urlopen(req, timeout=6) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Universe fetch timeout/skip {u}: {e}")

    sorted_list = sorted(list(symbols))
    if sorted_list:
        try:
            with open(local_path, "w") as fp:
                json.dump(sorted_list, fp, indent=2)
        except Exception:
            pass
    return set(sorted_list)

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
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", 0) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]
    if len(clean) < 30:
        return []

    # Dedup identical holiday bars
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

    # Corporate split adjuster
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
                if not adj_factor and 1.70 <= ratio <= 2.30:
                    adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50:
                    adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50:
                    adj_factor = 10.0
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

def run_scanB_backtest():
    print("🚀 Running Deterministic Scan B Backtest Engine...")
    nifty_750_set = get_nifty_750_universe()

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
            "scanB_backtest_report.json", "wyckoff_screener_results.json",
            "active_trade_plan.json", "scanA_results.json",
            "nifty750.json", "NIFTY50.json", "NIFTY.json"
        ]
    ]

    all_trades = []
    processed_count = 0

    for f_name in stock_files:
        processed_count += 1
        if processed_count % 300 == 0:
            print(f"⏳ Processed {processed_count}/{len(stock_files)} stocks...")

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

        clean_history = clean_and_prepare_dataset(raw)
        if len(clean_history) < 60:
            continue

        df = pd.DataFrame(clean_history)
        df["deliv_sma20"] = df["delivery_vol"].rolling(window=20, min_periods=1).mean()
        df["deliv_pct_sma20"] = df["deliv_pct"].rolling(window=20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean()

        cur_obv = 0
        obvs = []
        for row in clean_history:
            dv = row["delivery_vol"]
            if obvs:
                pc = clean_history[len(obvs) - 1]["close"]
                cc = row["close"]
                if cc > pc: cur_obv += dv
                elif cc < pc: cur_obv -= dv
            else:
                cur_obv = dv
            obvs.append(cur_obv)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        pcts = df["deliv_pct"].values
        pct_sma = df["deliv_pct_sma20"].values
        d_vols = df["delivery_vol"].values
        deliv_sma = df["deliv_sma20"].values
        to_50 = df["turnover_50d"].values
        times = df["time"].values
        obvs = np.array(obvs)
        N = len(closes)

        is_n750 = sym in nifty_750_set
        cooldown_idx = 0

        # Deterministic for-loop: impossible to enter an infinite loop
        for i in range(20, N - 1):
            if i < cooldown_idx:
                continue

            curr_to = to_50[i] if not np.isnan(to_50[i]) else 0.0
            if is_n750:
                if curr_to >= 30.0: category = "Category A (>30 Cr)"
                elif curr_to >= 5.0: category = "Category B (5-30 Cr)"
                else: category = "Category C (<5 Cr)"
            else:
                category = "Category D (Non-Universe)"

            base_start = i - 20
            qualifying = (d_vols[base_start:i] >= (1.20 * deliv_sma[base_start:i])) & \
                         (pcts[base_start:i] >= (1.20 * pct_sma[base_start:i]))

            if np.sum(qualifying) >= 3:
                base_highs = highs[base_start:i]
                sw_idx = int(np.argmax(base_highs))
                sw_high = base_highs[sw_idx]
                sw_obv = obvs[base_start + sw_idx]

                if closes[i] > sw_high and closes[i - 1] <= sw_high:
                    obv_gain_pct = ((obvs[i] - sw_obv) / abs(sw_obv) * 100) if sw_obv != 0 else 100.0
                    
                    if obv_gain_pct >= 5.0:
                        entry_price = closes[i]
                        entry_date = times[i]
                        sl_price = round(entry_price * (1.0 - (STOP_LOSS_PCT / 100.0)), 2)

                        fwd_end = min(N, i + 1 + MAX_HOLD_DAYS)
                        fwd_highs = highs[i + 1 : fwd_end]
                        fwd_lows = lows[i + 1 : fwd_end]

                        max_gain = 0.0
                        stopped_out = False
                        days_held = len(fwd_highs)

                        for d_idx in range(len(fwd_highs)):
                            gain = ((fwd_highs[d_idx] - entry_price) / entry_price) * 100.0
                            if gain > max_gain:
                                max_gain = gain
                            if fwd_lows[d_idx] <= sl_price:
                                stopped_out = True
                                days_held = d_idx + 1
                                break

                        all_trades.append({
                            "Symbol": sym,
                            "Category": category,
                            "Entry Date": entry_date,
                            "Entry Price": round(entry_price, 2),
                            "Swing High": round(sw_high, 2),
                            "OBV Breakout Gain %": round(obv_gain_pct, 2),
                            "Stop Loss (₹)": sl_price,
                            "Max Gain in 60D %": round(max_gain, 2),
                            "Hit +15%": bool(max_gain >= 15.0),
                            "Hit +20%": bool(max_gain >= 20.0),
                            "Hit +25%": bool(max_gain >= 25.0),
                            "Hit +30%+": bool(max_gain >= 30.0),
                            "Stopped Out (-10%)": stopped_out,
                            "Evaluated Days": days_held
                        })
                        cooldown_idx = i + max(10, days_held)

    df_res = pd.DataFrame(all_trades)
    print(f"📊 Completed! Total Trades Evaluated: {len(df_res)}")

    if df_res.empty:
        print("⚠️ No trades matched.")
        return

    m1_trades = df_res
    m1_rate = round((m1_trades["Hit +15%"].sum() / len(m1_trades)) * 100, 2) if len(m1_trades) else 0.0

    m2_trades = df_res[df_res["Category"] != "Category A (>30 Cr)"]
    m2_rate = round((m2_trades["Hit +20%"].sum() / len(m2_trades)) * 100, 2) if len(m2_trades) else 0.0

    m3_trades = df_res[~df_res["Category"].isin(["Category A (>30 Cr)", "Category B (5-30 Cr)"])]
    m3_rate = round((m3_trades["Hit +25%"].sum() / len(m3_trades)) * 100, 2) if len(m3_trades) else 0.0

    m4_trades = df_res[df_res["Category"] == "Category D (Non-Universe)"]
    m4_rate = round((m4_trades["Hit +30%+"].sum() / len(m4_trades)) * 100, 2) if len(m4_trades) else 0.0

    category_summary = {}
    for cat in ["Category A (>30 Cr)", "Category B (5-30 Cr)", "Category C (<5 Cr)", "Category D (Non-Universe)"]:
        sub = df_res[df_res["Category"] == cat]
        t_count = len(sub)
        if t_count > 0:
            category_summary[cat] = {
                "Total Trades": t_count,
                "+15% Hit Rate %": f"{round((sub['Hit +15%'].sum() / t_count) * 100, 1)}%",
                "+20% Hit Rate %": f"{round((sub['Hit +20%'].sum() / t_count) * 100, 1)}%",
                "+25% Hit Rate %": f"{round((sub['Hit +25%'].sum() / t_count) * 100, 1)}%",
                "+30%+ Hit Rate %": f"{round((sub['Hit +30%+'].sum() / t_count) * 100, 1)}%",
                "Stopped Out (-10%) %": f"{round((sub['Stopped Out (-10%)'].sum() / t_count) * 100, 1)}%",
                "Avg Max Gain in 60D %": f"+{round(float(sub['Max Gain in 60D %'].mean()), 2)}%"
            }

    final_report = {
        "Strategy Name": "Scan B: 20D Base (1.2x SMA Vol & %) + 5% OBV Breakout",
        "Primary Target Metrics": {
            "1. % Trades Reaching +15% (All Categories)": f"{m1_rate}% ({m1_trades['Hit +15%'].sum()}/{len(m1_trades)})",
            "2. % Trades Reaching +20% (Excluding Category A)": f"{m2_rate}% ({m2_trades['Hit +20%'].sum()}/{len(m2_trades)})",
            "3. % Trades Reaching +25% (Excluding Category A & B)": f"{m3_rate}% ({m3_trades['Hit +25%'].sum()}/{len(m3_trades)})",
            "4. % Trades Reaching +30%+ (Excluding Category A, B & C)": f"{m4_rate}% ({m4_trades['Hit +30%+'].sum()}/{len(m4_trades)})"
        },
        "Category Breakdown": category_summary,
        "Recent Trade Sample (Last 25)": all_trades[-25:] if len(all_trades) >= 25 else all_trades
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(final_report, fp, indent=2)

    print(f"🎉 Success! Report generated and saved to '{OUTPUT_REPORT}'.")

if __name__ == "__main__":
    run_scanB_backtest()
