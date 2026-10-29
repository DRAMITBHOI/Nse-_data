import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
START_DATE = "2021-01-01"
END_DATE = "2026-10-28"
MIN_TURNOVER_CR = 10.0

SUMMARY_JSON_FILE = os.path.join(DATA_DIR, "nifty100_combined_summary.json")
TRADES_CSV_FILE = os.path.join(DATA_DIR, "nifty100_combined_trades.csv")

NIFTY100_SYMBOLS = {
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", 
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "ATGL", "AUBANK", "AUROPHARMA", 
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BANKBARODA", 
    "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BOSCHLTD", "BPCL", 
    "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL", "CUMMINSIND", 
    "DABUR", "DIVISLAB", "DLF", "DMART", "DRREDDY", "EICHERMOT", "GAIL", "GODREJCP", 
    "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", 
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "HINDZINC", "ICICIBANK", "ICICIGI", 
    "ICICIPRULI", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", 
    "INFY", "INTERGLOBE", "IOC", "IRCTC", "IRFC", "ITC", "JINDALSTEL", "JIOFIN", 
    "JSWENERGY", "JSWSTEEL", "KOTAKBANK", "LICI", "LODHA", "LT", "LTIM", "LUPIN", 
    "M&M", "MARICO", "MARUTI", "MAXHEALTH", "MOTHERSON", "NAUKRI", "NESTLEIND", 
    "NTPC", "ONGC", "PFC", "PIDILITIND", "PNB", "POWERGRID", "RECLTD", "RELIANCE", 
    "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SRF", 
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", 
    "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNITDSPR", 
    "VBL", "VEDL", "VOLTAS", "WIPRO", "ZOMATO", "ZYDUSLIFE"
}

def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list): return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict): continue
        t = str(r.get("time", ""))[:10]
        c = float(r.get("close", 0) or 0)
        if not t or c <= 0: continue
        v = float(r.get("volume", 0) or 0)
        dv = float(r.get("delivery_vol", 0) or 0)
        dp = float(r.get("deliv_pct", 0) or 0)
        if dp > 0 and dv == 0: dv = v * (dp / 100.0)
        if v > 0 and dv > v: dv = v
        date_map[t] = {
            "time": t, "open": float(r.get("open", c) or c),
            "high": float(r.get("high", c) or c), "low": float(r.get("low", c) or c),
            "close": c, "volume": v, "delivery_vol": dv,
            "deliv_pct": (dv / v * 100.0) if v > 0 else 0.0
        }
    clean = [date_map[k] for k in sorted(date_map.keys())]

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15: adj = k; break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
            if adj:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj
                    clean[j]["volume"] = clean[j]["volume"] * adj
    return clean

def load_nifty_benchmark():
    for p in glob.glob("**/*.json", recursive=True):
        if os.path.basename(p).lower() in ["nifty.json", "nifty50.json"]:
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    clean = clean_and_prepare(json.load(fp))
                if len(clean) > 40:
                    df_bm = pd.DataFrame(clean)
                    df_bm["p60"] = df_bm["close"].pct_change(60).fillna(0)
                    return {str(r["time"]): float(r["p60"]) for _, r in df_bm.iterrows()}
            except Exception:
                pass
    return {}

def run_nifty100_backtest():
    print(f"🚀 Running Nifty 100 Combined Backtest ({START_DATE} to {END_DATE})...")
    nifty_map = load_nifty_benchmark()
    
    all_json = glob.glob("**/*.json", recursive=True)
    stock_files = {}
    for p in all_json:
        bname = os.path.basename(p)
        sym = os.path.splitext(bname)[0].upper()
        if any(k in bname.lower() for k in ["nifty", "audit", "result", "trade", "secret", "package", "checkpoint", "live"]):
            continue
        if sym in NIFTY100_SYMBOLS:
            if sym not in stock_files or os.path.getsize(p) > os.path.getsize(stock_files[sym]):
                stock_files[sym] = p

    print(f"📦 Filtered {len(stock_files)} Verified Nifty 100 Datasets.")

    all_trades = []

    for sym, fpath in sorted(stock_files.items()):
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                clean = clean_and_prepare(json.load(fp))
        except Exception:
            continue

        if len(clean) < 80: continue

        df = pd.DataFrame(clean)
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(200, min_periods=40).mean()
        df["gross_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
        df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
        df["perf_60d"] = df["close"].pct_change(60).fillna(0)
        
        prev_c = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()

        diff = df["close"].diff()
        direction = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
        df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
        df["dobv_20max"] = df["deliv_obv"].rolling(20, min_periods=5).max()
        df["range_closeness"] = np.where(df["high"] > df["low"], (df["close"] - df["low"]) / (df["high"] - df["low"]), 0.5)

        dp_avg = df["deliv_pct"].rolling(50, min_periods=5).mean().fillna(35.0)
        df["is_dist"] = (
            (df["close"] < df["close"].shift(1)) & 
            ((df["delivery_vol"] >= 1.30 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * dp_avg)) & 
            (df["range_closeness"] <= 0.45)
        ).astype(int)
        df["dist_10d"] = df["is_dist"].rolling(10, min_periods=1).sum()

        n = len(df)
        active_in_trade = False
        trade_exit_idx = 0

        for i in range(60, n - 2):
            current_date = df["time"].iloc[i]
            if current_date < START_DATE or current_date > END_DATE:
                continue

            if active_in_trade:
                if i <= trade_exit_idx:
                    continue
                else:
                    active_in_trade = False

            c0 = df["close"].iloc[i]
            h0 = df["high"].iloc[i]
            l0 = df["low"].iloc[i]
            dv0 = df["delivery_vol"].iloc[i]

            # 1. Turnover, 200 SMA, and Benchmark Relative Strength
            if df["turnover_50d"].iloc[i] < MIN_TURNOVER_CR:
                continue
            if pd.notnull(df["sma_200"].iloc[i]) and c0 < df["sma_200"].iloc[i]:
                continue
            n_perf = nifty_map.get(current_date, 0.001)
            if df["perf_60d"].iloc[i] < n_perf:
                continue

            # 2. Optimal Base Shelf Search (10-22 bars, 5.0% - 11.0% Spread)
            highs = df["high"].values
            lows = df["low"].values
            delivs = df["delivery_vol"].values
            closes = df["close"].values

            shelf_found = False
            lp_h, lp_l, best_len, best_spread = 0.0, 0.0, 15, 999.0
            for s_len in range(10, 23):
                s_h = highs[max(0, i - s_len):i].max()
                s_l = lows[max(0, i - s_len):i].min()
                if s_l > 0:
                    spread = ((s_h - s_l) / s_l) * 100.0
                    if 5.0 <= spread <= 11.0 and spread < best_spread:
                        best_spread = spread
                        lp_h = s_h
                        lp_l = s_l
                        best_len = s_len
                        shelf_found = True

            if not shelf_found:
                continue

            # 3. Accumulation Ratio Gate (>= 1.50)
            base_up = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] >= closes[k - 1])
            base_dn = sum(delivs[k] for k in range(max(0, i - best_len), i) if closes[k] < closes[k - 1])
            acc_ratio = round((base_up / base_dn), 2) if base_dn > 0 else 2.50
            if acc_ratio < 1.50:
                continue

            # 4. Confirmation Gates: Surge >= 1.60x, RC >= 0.65, OBV Highs
            prev_deliv_sma = df["deliv_sma20"].iloc[i - 1]
            dv_surge = dv0 / prev_deliv_sma if prev_deliv_sma > 0 else 1.0
            rc = df["range_closeness"].iloc[i]
            dobv_prev = df["dobv_20max"].iloc[i - 1]
            obv_ok = df["deliv_obv"].iloc[i] >= (dobv_prev * 0.98)
            cleared_pivot = h0 >= (lp_h * 1.0015) and c0 >= lp_h

            if not (cleared_pivot and dv_surge >= 1.60 and rc >= 0.65 and obv_ok):
                continue

            # 5. Position Entry & Exit 2.2 Execution
            entry_price = round(lp_h * 1.0015, 2)
            hard_sl = min(l0, lp_l)
            peak_price = max(entry_price, h0)
            peak_gain_pct = ((peak_price - entry_price) / entry_price) * 100.0
            vdu_under_count = 0
            exit_reason = "End of Data"
            exit_price = c0
            exit_idx = n - 1

            for j in range(i + 1, n):
                curr_c = df["close"].iloc[j]
                curr_prev_c = df["close"].iloc[j - 1]
                curr_h = df["high"].iloc[j]
                curr_v = df["volume"].iloc[j]
                curr_ema20 = df["ema_20"].iloc[j]
                curr_atr14 = df["atr_14"].iloc[j]
                curr_sma20_v = df["gross_sma20"].iloc[j]
                curr_dist10 = df["dist_10d"].iloc[j]
                curr_rc = df["range_closeness"].iloc[j]

                peak_price = max(peak_price, curr_h)
                peak_gain_pct = max(peak_gain_pct, ((peak_price - entry_price) / entry_price) * 100.0)
                vol_pace = curr_v / max(1.0, curr_sma20_v)

                # Rule 1: Catastrophe Stop
                if curr_c < hard_sl:
                    exit_reason = "1. Catastrophe Stop"
                    exit_price = curr_c
                    exit_idx = j
                    break

                # Rule 2: Climactic Churn Harvest
                elif peak_gain_pct >= 20.0 and vol_pace >= 2.0 and curr_rc <= 0.35:
                    exit_reason = "2. Climactic Churn Harvest"
                    exit_price = curr_c
                    exit_idx = j
                    break

                # Rule 3: Distribution Cluster Breakdown
                elif curr_dist10 >= 3 and curr_c < curr_ema20:
                    exit_reason = "3. Distribution Cluster"
                    exit_price = curr_c
                    exit_idx = j
                    break

                # Rule 4: Mega-Runner Cushion Lock
                elif peak_gain_pct >= 50.0:
                    cushion = max(curr_ema20, peak_price * 0.85)
                    if curr_c < cushion:
                        exit_reason = "4. Mega-Runner Cushion Lock"
                        exit_price = curr_c
                        exit_idx = j
                        break

                # Rule 5: 20 EMA Dynamic Trailing
                elif curr_c < curr_ema20:
                    is_heavy_down_vol = (vol_pace >= 1.30) and (curr_c < curr_prev_c)
                    if is_heavy_down_vol:
                        exit_reason = "5A. Heavy Vol 20 EMA Breach"
                        exit_price = curr_c
                        exit_idx = j
                        break
                    else:
                        vdu_under_count += 1
                        atr_floor = curr_ema20 - (1.5 * curr_atr14)
                        pct_floor = curr_ema20 * 0.95
                        dynamic_leeway = min(pct_floor, atr_floor)
                        if peak_gain_pct >= 6.0:
                            active_stop = max(entry_price * 1.01, dynamic_leeway)
                        else:
                            active_stop = dynamic_leeway

                        if curr_c < active_stop:
                            exit_reason = "5B. Dynamic VDU Floor Breach"
                            exit_price = curr_c
                            exit_idx = j
                            break
                        elif vdu_under_count > 5:
                            exit_reason = "5B. 5-Session Reclaim Failure"
                            exit_price = curr_c
                            exit_idx = j
                            break
                else:
                    vdu_under_count = 0

            trade_return = ((exit_price - entry_price) / entry_price) * 100.0
            holding_bars = exit_idx - i

            all_trades.append({
                "symbol": sym,
                "entry_date": current_date,
                "exit_date": df["time"].iloc[exit_idx],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_%": round(trade_return, 2),
                "peak_gain_%": round(peak_gain_pct, 2),
                "holding_bars": int(holding_bars),
                "exit_reason": exit_reason,
                "shelf_spread_%": round(best_spread, 2),
                "accumulation_ratio": acc_ratio
            })

            active_in_trade = True
            trade_exit_idx = exit_idx

    tdf = pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)
    wins = tdf[tdf["return_%"] > 0]
    losses = tdf[tdf["return_%"] <= 0]

    win_rate = (len(wins) / len(tdf)) * 100.0 if len(tdf) > 0 else 0.0
    avg_return = tdf["return_%"].mean() if len(tdf) > 0 else 0.0
    avg_win = wins["return_%"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["return_%"].mean() if len(losses) > 0 else 0.0
    profit_factor = wins["return_%"].sum() / abs(losses["return_%"].sum()) if len(losses) > 0 and losses["return_%"].sum() != 0 else 99.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 99.0

    print("\n" + "=" * 80)
    print("🏆 NIFTY 100 COMBINED OPTIMIZED BACKTEST (2021 – 2026)")
    print("=" * 80)
    print(f"Total Completed Trades      : {len(tdf)}")
    print(f"Winning Trades              : {len(wins)} ({win_rate:.2f}%)")
    print(f"Losing Trades               : {len(losses)} ({100 - win_rate:.2f}%)")
    print(f"Profit Factor               : {profit_factor:.2f}")
    print(f"Net Cumulative Return       : +{tdf['return_%'].sum():,.2f}%")
    print(f"Average Return / Trade      : {avg_return:+.2f}%")
    print(f"Average Gain / Winning Trade: +{avg_win:.2f}%")
    print(f"Average Loss / Losing Trade : {avg_loss:.2f}%")
    print(f"Win/Loss Payoff Ratio       : {payoff_ratio:.2f}x")
    print(f"Average Holding Period      : {tdf['holding_bars'].mean():.1f} trading sessions")
    print("=" * 80)

    tdf.to_csv(TRADES_CSV_FILE, index=False)

    exit_summary = tdf.groupby("exit_reason").agg(
        Count=("return_%", "count"),
        Win_Rate=("return_%", lambda s: round(float((s > 0).mean() * 100), 1)),
        Avg_Return=("return_%", lambda s: round(float(s.mean()), 2))
    ).reset_index().to_dict(orient="records")

    top_compounders = tdf.sort_values("return_%", ascending=False).head(10).to_dict(orient="records")

    summary_payload = {
        "Audit_Period": f"{START_DATE} to {END_DATE}",
        "Audit_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "Total_Trades": int(len(tdf)),
        "Winning_Trades": int(len(wins)),
        "Losing_Trades": int(len(losses)),
        "Win_Rate_%": round(win_rate, 2),
        "Profit_Factor": round(profit_factor, 2),
        "Net_Cumulative_Return_%": round(tdf["return_%"].sum(), 2),
        "Average_Return_Per_Trade_%": round(avg_return, 2),
        "Average_Win_%": round(avg_win, 2),
        "Average_Loss_%": round(avg_loss, 2),
        "Payoff_Ratio": round(payoff_ratio, 2),
        "Average_Holding_Period": round(tdf["holding_bars"].mean(), 1),
        "Winners_Held_Avg": round(wins["holding_bars"].mean(), 1),
        "Losers_Cut_Avg": round(losses["holding_bars"].mean(), 1),
        "Max_Gain_Single_Trade": f"+{tdf['return_%'].max():.2f}% ({tdf.loc[tdf['return_%'].idxmax(), 'symbol']})" if len(tdf) > 0 else "N/A",
        "Max_Loss_Single_Trade": f"{tdf['return_%'].min():.2f}% ({tdf.loc[tdf['return_%'].idxmin(), 'symbol']})" if len(tdf) > 0 else "N/A",
        "Exit_Reason_Breakdown": exit_summary,
        "Top_10_Compounders": top_compounders
    }

    with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"✅ Saved summary payload to `{SUMMARY_JSON_FILE}`.")

if __name__ == "__main__":
    run_nifty100_backtest()
