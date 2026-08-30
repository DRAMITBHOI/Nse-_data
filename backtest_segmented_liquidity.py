import os
import io
import json
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
# 3. PREPARE STOCK DATA & SERIES
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
            
            # Turnover & Baseline Metrics
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d_avg"] = df["turnover_cr"].rolling(50, min_periods=10).mean()
            df["deliv_pct_50d_avg"] = df["deliv_pct"].rolling(50, min_periods=10).mean()
            df["deliv_vol_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
            df["deliv_vol_sma50"] = df["delivery_vol"].rolling(50, min_periods=10).mean()
            df["deliv_vol_sma5"] = df["delivery_vol"].rolling(5, min_periods=1).mean()
            df["vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()

            # True Delivery OBV
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

    print(f"✅ Prepared {len(stock_map)} stocks for unconstrained exit evaluation.\n")
    return stock_map

# ==========================================
# 4. UNCONSTRAINED TRADE SIMULATOR
# ==========================================
def simulate_trade_exit(df, entry_idx, entry_price, initial_stop_loss, exit_model, max_unconstrained_hold=180):
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    vols = df["volume"].values
    d_vols = df["delivery_vol"].values
    pcts = df["deliv_pct"].values
    pct_50_avg = df["deliv_pct_50d_avg"].values
    d_vol_sma50 = df["deliv_vol_sma50"].values
    d_vol_sma5 = df["deliv_vol_sma5"].values
    vol_sma20 = df["vol_sma20"].values
    ema10 = df["ema10"].values
    obvs = df["deliv_obv"].values
    N = len(closes)

    end_idx = min(N - 1, entry_idx + max_unconstrained_hold)
    hit_20pct = False
    max_gain = 0.0
    current_sl = initial_stop_loss

    for curr in range(entry_idx, end_idx + 1):
        c_open = opens[curr]
        c_high = highs[curr]
        c_low = lows[curr]
        c_close = closes[curr]

        gain = ((c_high - entry_price) / entry_price) * 100
        if gain > max_gain:
            max_gain = gain
        if gain >= 20.0:
            hit_20pct = True

        # Model 3: Dynamic Trailing Stop Adjustment
        if exit_model == "Model 3: Dynamic Trailing Stop":
            # Stage 1: Move SL to Breakeven once gain >= +10%
            if max_gain >= 10.0 and current_sl < entry_price:
                current_sl = entry_price
            
            # Stage 2: Trail 10-Day Swing Low once gain >= +15%
            if max_gain >= 15.0 and curr >= entry_idx + 10:
                trail_10d_low = float(np.min(lows[curr - 10 : curr]))
                if trail_10d_low > current_sl:
                    current_sl = trail_10d_low

        # Stop Loss Trigger (Evaluated Intraday)
        if c_low <= current_sl:
            exit_price = min(c_open, current_sl)
            pnl = ((exit_price - entry_price) / entry_price) * 100
            return pnl, (curr - entry_idx + 1), hit_20pct, max_gain, "Stop Triggered"

        # Signal-Based Exits (Evaluated at Close of Day 'curr', Exited at Day 'curr + 1' Open)
        trigger_exit = False
        exit_reason = ""

        if curr > entry_idx + 2 and curr < end_idx:
            # Model 1: Bearish True Delivery OBV Divergence
            if exit_model == "Model 1: Bearish OBV Divergence":
                for k in range(5, min(21, curr - entry_idx + 1)):
                    prev_idx = curr - k
                    if c_high >= highs[prev_idx] and obvs[curr] < obvs[prev_idx]:
                        recent_5d_low = np.min(lows[curr - 4 : curr])
                        if c_close < recent_5d_low:
                            trigger_exit = True
                            exit_reason = "OBV Divergence"
                            break

            # Model 2A: Buyer Vacuum / Delivery Volume Shrinkage
            elif exit_model == "Model 2A: Volume Vacuum (Shrinkage)":
                if d_vol_sma50[curr] > 0 and d_vol_sma5[curr] <= (0.60 * d_vol_sma50[curr]):
                    if c_close < ema10[curr]:
                        trigger_exit = True
                        exit_reason = "Delivery Vacuum"

            # Model 2B: High-Volume Intraday Churn Day
            elif exit_model == "Model 2B: High-Vol Churn Day":
                if vol_sma20[curr] > 0 and pct_50_avg[curr] > 0:
                    is_heavy_vol = vols[curr] >= (1.5 * vol_sma20[curr])
                    is_low_deliv = pcts[curr] <= (0.70 * pct_50_avg[curr])
                    is_red_or_doji = c_close <= (c_open * 1.002)
                    if is_heavy_vol and is_low_deliv and is_red_or_doji:
                        trigger_exit = True
                        exit_reason = "Heavy Churn Climax"

            # Model 2C: Distribution Cluster
            elif exit_model == "Model 2C: Distribution Cluster":
                sub_par_days = 0
                for look in range(curr - 4, curr + 1):
                    if pct_50_avg[look] > 0 and pcts[look] < (0.80 * pct_50_avg[look]):
                        sub_par_days += 1
                if sub_par_days >= 3 and c_close < ema10[curr]:
                    trigger_exit = True
                    exit_reason = "Distribution Cluster"

        if trigger_exit and (curr + 1) <= end_idx:
            next_open = opens[curr + 1]
            pnl = ((next_open - entry_price) / entry_price) * 100
            return pnl, (curr - entry_idx + 2), hit_20pct, max_gain, exit_reason

    final_close = closes[end_idx]
    pnl = ((final_close - entry_price) / entry_price) * 100
    return pnl, (end_idx - entry_idx + 1), hit_20pct, max_gain, "End of History"

# ==========================================
# 5. EXECUTE COMPARATIVE GRID
# ==========================================
def run_exit_comparison_backtest(stock_map):
    exit_models = [
        "Baseline: Base SL Only (Unconstrained)",
        "Model 1: Bearish OBV Divergence",
        "Model 2A: Volume Vacuum (Shrinkage)",
        "Model 2B: High-Vol Churn Day",
        "Model 2C: Distribution Cluster",
        "Model 3: Dynamic Trailing Stop"
    ]

    tier_config = {
        "Bucket A (>30 Cr)": {"min_to": 30.0, "max_to": 1e9, "pct_mult": 1.4, "vol_mult": 1.0, "cluster": 3, "window": 15},
        "Bucket B (5-30 Cr)": {"min_to": 5.0, "max_to": 30.0, "pct_mult": 1.2, "vol_mult": 1.0, "cluster": 2, "window": 15},
        "Bucket C (<5 Cr)": {"min_to": 0.0, "max_to": 5.0, "pct_mult": 1.4, "vol_mult": 1.0, "cluster": 4, "window": 20}
    }

    report_data = {}

    for b_name, b_cfg in tier_config.items():
        print(f"📊 Testing 6 Exit Models on {b_name}...")
        min_to = b_cfg["min_to"]
        max_to = b_cfg["max_to"]
        pct_mult = b_cfg["pct_mult"]
        vol_mult = b_cfg["vol_mult"]
        min_cluster = b_cfg["cluster"]
        window = b_cfg["window"]

        entry_signals = []
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

            for i in range(max(window + 20, 30), N - 30):
                if i <= last_exit:
                    continue

                curr_to = to_50_avg[i]
                if np.isnan(curr_to) or not (min_to <= curr_to < max_to):
                    continue

                if np.sum(qualifying_days[i - window : i]) < min_cluster:
                    continue

                base_highs = highs[i - window : i]
                swing_high_rel_idx = np.argmax(base_highs)
                swing_high_abs_idx = (i - window) + swing_high_rel_idx
                swing_high = base_highs[swing_high_rel_idx]
                base_low = np.min(lows[i - window : i])

                if closes[i] > swing_high and closes[i - 1] <= swing_high:
                    if obvs[i] <= obvs[swing_high_abs_idx]:
                        continue

                    entry_price = opens[i + 1] if opens[i + 1] > 0 else closes[i]
                    stop_loss = round(base_low * 0.995, 2)
                    risk = entry_price - stop_loss

                    if risk <= 0 or (risk / entry_price) > 0.15:
                        continue

                    entry_signals.append({
                        "sym": sym,
                        "entry_idx": i + 1,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss
                    })
                    last_exit = i + 10

        tier_results = []
        for model in exit_models:
            trades_pnl = []
            trades_hold = []
            trades_hit_20 = []
            trades_win = []

            for sig in entry_signals:
                df = stock_map[sig["sym"]]
                pnl, hold_days, hit_20, max_g, reason = simulate_trade_exit(
                    df, sig["entry_idx"], sig["entry_price"], sig["stop_loss"], model, max_unconstrained_hold=180
                )
                trades_pnl.append(pnl)
                trades_hold.append(hold_days)
                trades_hit_20.append(1 if hit_20 else 0)
                trades_win.append(1 if pnl > 0 else 0)

            tot = len(trades_pnl)
            if tot > 0:
                win_rate = round((sum(trades_win) / tot) * 100, 1)
                win_20pct = round((sum(trades_hit_20) / tot) * 100, 1)
                avg_pnl = round(float(np.mean(trades_pnl)), 2)
                avg_days = round(float(np.mean(trades_hold)), 1)
                
                neg_sum = abs(float(np.sum([p for p in trades_pnl if p < 0])))
                pos_sum = float(np.sum([p for p in trades_pnl if p > 0]))
                profit_factor = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0

                tier_results.append({
                    "Exit Model": model,
                    "Total Trades": tot,
                    "Win Rate %": win_rate,
                    "Avg Return %": f"{avg_pnl:+0.2f}%",
                    "Profit Factor": profit_factor,
                    "Avg Hold (Days)": avg_days,
                    "Score": round(win_rate * profit_factor, 1)
                })

        report_data[b_name] = tier_results

    return report_data

# ==========================================
# 6. MAIN
# ==========================================
def main():
    nifty750 = get_nifty_750_universe()
    stock_map = prepare_stock_series(nifty750)

    if stock_map:
        report_data = run_exit_comparison_backtest(stock_map)

        for b_name, res in report_data.items():
            print("\n" + "=" * 95)
            print(f"🏆 EXIT STRATEGY COMPARISON FOR: {b_name.upper()}")
            print("=" * 95)
            df_res = pd.DataFrame(res)
            if not df_res.empty:
                print(df_res.to_string(index=False))

        out_file = os.path.join(DATA_DIR, "segmented_backtest_report.json")
        with open(out_file, "w") as fp:
            json.dump(report_data, fp, indent=2)
        print(f"\n💾 Exit comparison backtest saved to '{out_file}'")
    else:
        print("⚠️ No stock data available.")

if __name__ == "__main__":
    main()
