import os
import io
import json
import itertools
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

# ==========================================
# 1. LOAD NIFTY 750 UNIVERSE
# ==========================================
def get_nifty_750_universe():
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fp:
                return set(json.load(fp))
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
            with urllib.request.urlopen(req, timeout=15) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Notice fetching {u}: {e}")

    if symbols:
        with open(local_path, "w") as fp:
            json.dump(sorted(list(symbols)), fp, indent=2)
            
    return symbols

# ==========================================
# 2. SPLIT & CORPORATE ACTION CLEANER
# ==========================================
def full_corporate_action_adjustment(raw_data):
    if not raw_data or len(raw_data) < 2:
        return raw_data
    clean = []
    for r in raw_data:
        try:
            c = float(r.get("close", 0))
            if c <= 0:
                continue
            clean.append({
                "time": str(r.get("time", "")),
                "open": float(r.get("open", c)),
                "high": float(r.get("high", c)),
                "low": float(r.get("low", c)),
                "close": c,
                "delivery_vol": float(r.get("delivery_vol", 0) or 0),
                "volume": float(r.get("volume", r.get("delivery_vol", 0)) or 0),
                "deliv_pct": float(r.get("deliv_pct", 0) or 0)
            })
        except Exception:
            continue

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
    return clean

# ==========================================
# 3. PREPARE STOCK SERIES
# ==========================================
def prepare_stock_series(nifty_750_set):
    stock_map = {}
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: '{DATA_DIR}' directory does not exist.")
        return stock_map

    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "nifty750.json", "backtest_report.json",
            "segmented_backtest_report.json", "wyckoff_screener_results.json"
        ]
    ]
    if nifty_750_set:
        files = [f for f in files if f.replace(".json", "").strip().upper() in nifty_750_set]

    print(f"📂 Processing {len(files)} Nifty 750 stock files...")

    for f in files:
        sym = f.replace(".json", "").strip().upper()
        try:
            with open(os.path.join(DATA_DIR, f), "r") as fp:
                raw = json.load(fp)
            clean = full_corporate_action_adjustment(raw)
            if len(clean) < 80:
                continue

            df = pd.DataFrame(clean)
            
            # Turnover in Crores & 50D average
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d_avg"] = df["turnover_cr"].rolling(50, min_periods=10).mean()

            # 50D Mean Delivery %
            df["deliv_pct_50d_avg"] = df["deliv_pct"].rolling(50, min_periods=10).mean()

            # 20D Delivery Volume SMA
            df["deliv_vol_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()

            # True Delivery OBV (EOD calculated)
            closes = df["close"].values
            vols = df["delivery_vol"].values
            t_vols = df["volume"].values
            N = len(closes)
            obvs = np.zeros(N)
            cur_obv = 0
            for idx in range(N):
                d_v = min(vols[idx], t_vols[idx]) if t_vols[idx] > 0 else vols[idx]
                if idx > 0:
                    if closes[idx] > closes[idx - 1]:
                        cur_obv += d_v
                    elif closes[idx] < closes[idx - 1]:
                        cur_obv -= d_v
                else:
                    cur_obv = d_v
                obvs[idx] = cur_obv
            df["deliv_obv"] = obvs

            stock_map[sym] = df
        except Exception:
            continue

    print(f"✅ Prepared {len(stock_map)} stocks for liquidity-segmented testing.\n")
    return stock_map

# ==========================================
# 4. GRID SEARCH ENGINE (EOD BREAKOUT & OBV > SWING HIGH OBV)
# ==========================================
def run_segmented_backtest(stock_map):
    pct_spike_grid = [1.2, 1.4, 1.6, 1.8]
    vol_sma_grid = [1.0, 1.2, 1.5, 1.8]
    cluster_cnt_grid = [2, 3, 4]
    window_grid = [15, 20, 30]

    grid = list(itertools.product(pct_spike_grid, vol_sma_grid, cluster_cnt_grid, window_grid))

    buckets = {
        "Bucket A (High Liquidity: > 30 Cr)": {"min_to": 30.0, "max_to": 1e9, "results": []},
        "Bucket B (Medium Liquidity: 5 - 30 Cr)": {"min_to": 5.0, "max_to": 30.0, "results": []},
        "Bucket C (Low Liquidity: < 5 Cr)": {"min_to": 0.0, "max_to": 5.0, "results": []}
    }

    print(f"🔬 Testing {len(grid)} Parameter Setups (EOD Breakout OBV > Swing High OBV | 60-Day Horizon)...")

    for b_name, b_info in buckets.items():
        min_to = b_info["min_to"]
        max_to = b_info["max_to"]

        for pct_mult, vol_mult, min_cluster, window in grid:
            trades = []

            for sym, df in stock_map.items():
                opens = df["open"].values
                closes = df["close"].values
                highs = df["high"].values
                lows = df["low"].values
                pcts = df["deliv_pct"].values
                pct_50_avg = df["deliv_pct_50d_avg"].values
                d_vols = df["delivery_vol"].values
                vol_sma20 = df["deliv_vol_sma20"].values
                to_50_avg = df["turnover_50d_avg"].values
                obvs = df["deliv_obv"].values
                N = len(closes)

                qualifying_days = (pct_50_avg > 0) & (pcts >= (pct_mult * pct_50_avg)) & (d_vols >= (vol_mult * vol_sma20))

                last_exit = -1

                # Evaluate up to N - 61 so we have a full 60 trading-day holding window forward
                for i in range(max(window + 20, 30), N - 61):
                    if i <= last_exit:
                        continue

                    curr_to = to_50_avg[i]
                    if np.isnan(curr_to) or not (min_to <= curr_to < max_to):
                        continue

                    # 1. Cluster Count Condition in Base Window
                    cluster_count = np.sum(qualifying_days[i - window : i])
                    if cluster_count < min_cluster:
                        continue

                    # 2. Identify the Swing High price and its exact candle index in the base
                    base_highs = highs[i - window : i]
                    swing_high_rel_idx = np.argmax(base_highs)
                    swing_high_abs_idx = (i - window) + swing_high_rel_idx
                    swing_high = base_highs[swing_high_rel_idx]
                    
                    base_low = np.min(lows[i - window : i])

                    # 3. Price Breakout: EOD Close on Day i crosses previous Swing High
                    if closes[i] > swing_high and closes[i - 1] <= swing_high:
                        
                        # 4. OBV Confirmation: Finalized EOD OBV on Day i > OBV at Swing High Day
                        obv_at_swing_high = obvs[swing_high_abs_idx]
                        if obvs[i] <= obv_at_swing_high:
                            continue

                        # Realistic Execution: Entry at Day i+1 Market Open
                        entry_price = opens[i + 1] if opens[i + 1] > 0 else closes[i]
                        stop_loss = round(base_low * 0.995, 2)
                        risk = entry_price - stop_loss

                        if risk <= 0 or (risk / entry_price) > 0.15:  # Max 15% Risk Limit
                            continue

                        target_2r = entry_price + (2.0 * risk)

                        # 60 Trading Days (~3 Months) Forward Evaluation starting from i+1
                        hit_20pct_rally = False
                        hit_2r_target = False
                        hit_sl = False
                        max_gain = 0.0

                        for fwd in range(i + 1, min(N, i + 62)):
                            c_high = highs[fwd]
                            c_low = lows[fwd]

                            gain = ((c_high - entry_price) / entry_price) * 100
                            if gain > max_gain:
                                max_gain = gain

                            # Check Stop Loss first
                            if c_low <= stop_loss:
                                hit_sl = True
                                break

                            if c_high >= target_2r:
                                hit_2r_target = True

                            if gain >= 20.0:
                                hit_20pct_rally = True
                                break

                        trades.append({
                            "hit_20pct": 1 if hit_20pct_rally else 0,
                            "hit_2r": 1 if hit_2r_target else 0,
                            "hit_sl": 1 if hit_sl else 0,
                            "max_gain": max_gain
                        })
                        last_exit = i + 10  # Prevent consecutive duplicate triggers

            tot = len(trades)
            if tot >= 10:
                win_rate_20pct = round((sum(t["hit_20pct"] for t in trades) / tot) * 100, 1)
                win_rate_2r = round((sum(t["hit_2r"] for t in trades) / tot) * 100, 1)
                sl_rate = round((sum(t["hit_sl"] for t in trades) / tot) * 100, 1)
                avg_rally = round(float(np.mean([t["max_gain"] for t in trades])), 1)

                b_info["results"].append({
                    "Deliv % Spike": f"{pct_mult}x 50D Mean",
                    "Vol vs SMA20": f"{vol_mult}x SMA20",
                    "Cluster Count": f"{min_cluster} days",
                    "Base Window": f"{window}d",
                    "Total Trades": tot,
                    "Hit 20% Rally %": win_rate_20pct,
                    "Hit 1:2 RR %": win_rate_2r,
                    "Hit SL %": sl_rate,
                    "Avg Max Rally %": avg_rally,
                    "Score": round(win_rate_20pct * np.log10(tot) * (avg_rally / 10), 1)
                })

    return buckets

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    nifty750 = get_nifty_750_universe()
    stock_map = prepare_stock_series(nifty750)

    if stock_map:
        bucket_results = run_segmented_backtest(stock_map)

        for b_name, b_info in bucket_results.items():
            print("\n" + "=" * 95)
            print(f"🏆 TOP 5 OPTIMAL SETUPS FOR: {b_name.upper()} (EOD BREAKOUT OBV | 60-DAY HORIZON)")
            print("=" * 95)
            df_res = pd.DataFrame(b_info["results"])
            if not df_res.empty:
                df_top = df_res.sort_values(by="Score", ascending=False).head(5)
                print(df_top.to_string(index=False))
            else:
                print("No configurations met minimum trade sample threshold.")

        out_file = os.path.join(DATA_DIR, "segmented_backtest_report.json")
        with open(out_file, "w") as fp:
            json.dump({k: v["results"] for k, v in bucket_results.items()}, fp, indent=2)
        print(f"\n💾 Full backtest grid saved to '{out_file}'")
    else:
        print("⚠️ No stock data available to run backtest.")

if __name__ == "__main__":
    main()
