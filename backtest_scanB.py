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
STOP_LOSS_PCT = 10.0  # Random / Flat 10% SL

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
            with urllib.request.urlopen(req, timeout=12) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Warning fetching universe from {u}: {e}")

    sorted_list = sorted(list(symbols))
    if sorted_list:
        with open(local_path, "w") as fp:
            json.dump(sorted_list, fp, indent=2)
    return set(sorted_list)

def clean_and_prepare_dataset(raw_data):
    if not raw_data:
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
            if dt.dayofweek >= 5:  # Skip weekends
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

    sorted_records = [date_map[k] for k in sorted(date_map.keys())]

    clean = []
    for r in sorted_records:
        if clean:
            prev = clean[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    clean.pop()
        clean.append(r)

    # Corporate actions split adjustment
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

def run_scanB_backtest():
    print("🚀 Running Scan B Backtest Engine (60-Day Forward Target Hit Analysis)...")
    
    nifty_750_set = get_nifty_750_universe()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
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

    for f_name in stock_files:
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

        # True Demat Delivery OBV
        cur_obv = 0
        obvs = []
        for i, row in df.iterrows():
            dv = float(row["delivery_vol"])
            if i > 0:
                pc = float(df.at[i - 1, "close"])
                cc = float(row["close"])
                if cc > pc: cur_obv += dv
                elif cc < pc: cur_obv -= dv
            else:
                cur_obv = dv
            obvs.append(cur_obv)
        df["deliv_obv"] = obvs

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        pcts = df["deliv_pct"].values
        pct_sma = df["deliv_pct_sma20"].values
        d_vols = df["delivery_vol"].values
        deliv_sma = df["deliv_sma20"].values
        to_50 = df["turnover_50d"].values
        times = df["time"].values
        N = len(closes)

        is_n750 = sym in nifty_750_set

        i = 20
        while i < N - 1:
            curr_to = to_50[i] if not np.isnan(to_50[i]) else 0.0

            # Categorization
            if is_n750:
                if curr_to >= 30.0:
                    category = "Category A (>30 Cr)"
                elif curr_to >= 5.0:
                    category = "Category B (5-30 Cr)"
                else:
                    category = "Category C (<5 Cr)"
            else:
                category = "Category D (Non-Universe)"

            base_start = i - 20
            # Condition B: In any 3 days of prev 20 days: delivery_vol >= 1.2*sma20 AND deliv_pct >= 1.2*sma20
            qualifying = (d_vols[base_start:i] >= (1.20 * deliv_sma[base_start:i])) & \
                         (pcts[base_start:i] >= (1.20 * pct_sma[base_start:i]))
            
            if np.sum(qualifying) >= 3:
                base_highs = highs[base_start:i]
                sw_idx = int(np.argmax(base_highs))
                sw_high = base_highs[sw_idx]
                sw_obv = obvs[base_start + sw_idx]

                # Condition A: Price breaks previous swing high
                if closes[i] > sw_high and closes[i - 1] <= sw_high:
                    # Condition C: OBV at breakout is >= swing high OBV by at least 5%
                    # Check handles positive & relative baseline
                    obv_gain_pct = ((obvs[i] - sw_obv) / abs(sw_obv) * 100) if sw_obv != 0 else 100.0
                    
                    if obv_gain_pct >= 5.0:
                        entry_price = closes[i]
                        entry_date = times[i]
                        sl_price = round(entry_price * (1.0 - (STOP_LOSS_PCT / 100.0)), 2)

                        # Track 60 Trading Days Forward Performance
                        forward_limit = min(N, i + 1 + MAX_HOLD_DAYS)
                        forward_highs = highs[i + 1 : forward_limit]
                        forward_lows = lows[i + 1 : forward_limit]
                        forward_closes = closes[i + 1 : forward_limit]
                        
                        hit_15 = False
                        hit_20 = False
                        hit_25 = False
                        hit_30 = False
                        stopped_out = False
                        max_gain_pct = 0.0
                        days_to_exit = len(forward_highs)

                        for f_idx in range(len(forward_highs)):
                            bar_high = forward_highs[f_idx]
                            bar_low = forward_lows[f_idx]

                            current_gain = ((bar_high - entry_price) / entry_price) * 100
                            if current_gain > max_gain_pct:
                                max_gain_pct = current_gain

                            if max_gain_pct >= 15.0:
                                hit_15 = True
                            if max_gain_pct >= 20.0:
                                hit_20 = True
                            if max_gain_pct >= 25.0:
                                hit_25 = True
                            if max_gain_pct >= 30.0:
                                hit_30 = True

                            # Check 10% SL hit
                            if bar_low <= sl_price:
                                stopped_out = True
                                days_to_exit = f_idx + 1
                                break

                        all_trades.append({
                            "Symbol": sym,
                            "Category": category,
                            "Entry Date": entry_date,
                            "Entry Price": round(entry_price, 2),
                            "Swing High": round(sw_high, 2),
                            "OBV Breakout Gain %": round(obv_gain_pct, 2),
                            "Stop Loss (₹)": sl_price,
                            "Max Gain in 60D %": round(max_gain_pct, 2),
                            "Hit +15%": hit_15,
                            "Hit +20%": hit_20,
                            "Hit +25%": hit_25,
                            "Hit +30%+": hit_30,
                            "Stopped Out (-10%)": stopped_out,
                            "Evaluated Days": days_to_exit
                        })

                        # Advance pointer past trade evaluation to prevent overlap bias
                        i += max(10, days_to_exit)
                        continue

            i += 1

    df_res = pd.DataFrame(all_trades)
    total_trades = len(df_res)
    print(f"📊 Total Qualified Scan B Trades: {total_trades}")

    if df_res.empty:
        print("⚠️ No trades matched the criteria.")
        return

    # 1. Metric 1: % of trades getting 15% within 60 trading days (ALL trades)
    m1_trades = df_res
    m1_hit_rate = round((m1_trades["Hit +15%"].sum() / len(m1_trades)) * 100, 2) if len(m1_trades) > 0 else 0.0

    # 2. Metric 2: % of trades getting 20% within 60 trading days (EXCLUDING Category A)
    m2_trades = df_res[df_res["Category"] != "Category A (>30 Cr)"]
    m2_hit_rate = round((m2_trades["Hit +20%"].sum() / len(m2_trades)) * 100, 2) if len(m2_trades) > 0 else 0.0

    # 3. Metric 3: % of trades getting 25% (EXCLUDING Category A and B)
    m3_trades = df_res[~df_res["Category"].isin(["Category A (>30 Cr)", "Category B (5-30 Cr)"])]
    m3_hit_rate = round((m3_trades["Hit +25%"].sum() / len(m3_trades)) * 100, 2) if len(m3_trades) > 0 else 0.0

    # 4. Metric 4: % of trades getting 30%+ (EXCLUDING Category A, B, and C)
    m4_trades = df_res[df_res["Category"] == "Category D (Non-Universe)"]
    m4_hit_rate = round((m4_trades["Hit +30%+"].sum() / len(m4_trades)) * 100, 2) if len(m4_trades) > 0 else 0.0

    # Breakdown by Category
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
        "Strategy Name": "Scan B: 20D Cluster (1.2x SMA Vol & %) + 5% OBV Surge Breakout",
        "Parameters": {
            "Lookback Base": "20 Days",
            "Delivery Cluster Requirement": ">= 3 Days (Delivery Vol >= 1.2x SMA20 AND Deliv % >= 1.2x SMA20)",
            "OBV Breakout Surge Requirement": ">= 5% above Swing High OBV",
            "Stop Loss": "10% Random / Flat",
            "Max Forward Holding Period": "60 Trading Days"
        },
        "Primary Target Metrics": {
            "1. % Trades Reaching +15% (All Categories)": f"{m1_hit_rate}% ({m1_trades['Hit +15%'].sum()}/{len(m1_trades)})",
            "2. % Trades Reaching +20% (Excluding Category A)": f"{m2_hit_rate}% ({m2_trades['Hit +20%'].sum()}/{len(m2_trades)})",
            "3. % Trades Reaching +25% (Excluding Category A & B)": f"{m3_hit_rate}% ({m3_trades['Hit +25%'].sum()}/{len(m3_trades)})",
            "4. % Trades Reaching +30%+ (Excluding Category A, B & C)": f"{m4_hit_rate}% ({m4_trades['Hit +30%+'].sum()}/{len(m4_trades)})"
        },
        "Category Breakdown": category_summary,
        "Recent Trade Sample (Last 25)": all_trades[-25:] if len(all_trades) >= 25 else all_trades
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(final_report, fp, indent=2)

    print("\n" + "="*70)
    print("🎯 SCAN B BACKTEST RESULTS (60-DAY FORWARD WINDOW)")
    print("="*70)
    print(f"1. +15% Target Hit Rate (All Trades):                      {m1_hit_rate}%")
    print(f"2. +20% Target Hit Rate (Excluding Category A):             {m2_hit_rate}%")
    print(f"3. +25% Target Hit Rate (Excluding Category A & B):         {m3_hit_rate}%")
    print(f"4. +30%+ Target Hit Rate (Excluding Category A, B & C):     {m4_hit_rate}%")
    print("="*70)
    print(f"📁 Full report saved to '{OUTPUT_REPORT}'.")

if __name__ == "__main__":
    run_scanB_backtest()
