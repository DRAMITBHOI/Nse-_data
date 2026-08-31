import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_REPORT = os.path.join(DATA_DIR, "segmented_backtest_report.json")
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
            with urllib.request.urlopen(req, timeout=12) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception as e:
            print(f"⚠️ Warning fetching {u}: {e}")

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
            d_str = pd.to_datetime(raw_t).strftime("%Y-%m-%d")
            c = float(r.get("close", 0))
            if c <= 0:
                continue
            
            v = float(r.get("volume", 0) or 0)
            dv = float(r.get("delivery_vol", 0) or 0)
            pct = float(r.get("deliv_pct", 0) or 0)

            entry = {
                "time": d_str,
                "open": float(r.get("open", c)),
                "high": float(r.get("high", c)),
                "low": float(r.get("low", c)),
                "close": c,
                "delivery_vol": dv,
                "volume": v,
                "deliv_pct": pct
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
                r["low"] == prev["low"] and r["close"] == prev["close"] and r["volume"] == 0):
                continue
        clean.append(r)

    # Corporate actions adjustment
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

    # Forward fill missing delivery volume
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

def build_nifty_regime_map():
    nifty_path = os.path.join(DATA_DIR, "NIFTY50.json")
    if not os.path.exists(nifty_path):
        nifty_path = os.path.join(DATA_DIR, "NIFTY.json")
    
    regime_map = {}
    if os.path.exists(nifty_path):
        try:
            with open(nifty_path, "r") as f:
                raw = json.load(f)
            df = pd.DataFrame(clean_and_prepare_dataset(raw))
            if not df.empty and "close" in df.columns:
                df["sma50"] = df["close"].rolling(50, min_periods=20).mean()
                for _, r in df.iterrows():
                    d = str(r["time"]).split(" ")[0]
                    c = float(r["close"])
                    sma = float(r["sma50"]) if not np.isnan(r["sma50"]) else c
                    regime_map[d] = "Favourable" if c >= sma else "Unfavourable"
                return regime_map
        except Exception:
            pass

    proxy_path = os.path.join(DATA_DIR, "RELIANCE.json")
    if os.path.exists(proxy_path):
        try:
            with open(proxy_path, "r") as f:
                raw = json.load(f)
            df = pd.DataFrame(clean_and_prepare_dataset(raw))
            df["sma50"] = df["close"].rolling(50, min_periods=20).mean()
            for _, r in df.iterrows():
                d = str(r["time"]).split(" ")[0]
                c = float(r["close"])
                sma = float(r["sma50"]) if not np.isnan(r["sma50"]) else c
                regime_map[d] = "Favourable" if c >= sma else "Unfavourable"
        except Exception:
            pass
    return regime_map

def run_segmented_backtest():
    print("🚀 Running High-Win-Rate Backtest (Bucket C 10% TP + Macro Weekly EMA Filter)...")
    
    nifty_750_set = get_nifty_750_universe()
    nifty_regime = build_nifty_regime_map()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "backtest_report.json", "segmented_backtest_report.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "scanA_results.json", "nifty750.json", "NIFTY50.json", "NIFTY.json"
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
        df["deliv_sma"] = df["delivery_vol"].rolling(window=20, min_periods=1).mean()
        df["gross_vol_sma20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
        df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean()
        df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean()
        df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()  # Weekly 20 EMA proxy

        # Continuous True Demat Delivery OBV
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
        opens = df["open"].values
        pcts = df["deliv_pct"].values
        pct_50 = df["deliv_pct_50d"].values
        d_vols = df["delivery_vol"].values
        deliv_sma = df["deliv_sma"].values
        gross_vols = df["volume"].values
        gross_sma = df["gross_vol_sma20"].values
        to_50 = df["turnover_50d"].values
        ema100 = df["ema100"].values
        times = df["time"].values
        N = len(closes)

        is_n750 = sym in nifty_750_set

        in_trade = False
        entry_price = 0.0
        entry_idx = 0
        entry_date = ""
        active_sl = 0.0
        active_bucket = ""
        entry_nifty_regime = "Favourable"
        max_run_gain = 0.0
        partial_booked = False
        partial_target_pct = 15.0
        cooldown_until = 0

        for i in range(50, N):
            curr_to = to_50[i] if not np.isnan(to_50[i]) else 0.0

            if is_n750:
                if curr_to >= 30.0:
                    tier = "Bucket A (>30 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 3, 45
                    target_tp = 15.0
                elif curr_to >= 5.0:
                    tier = "Bucket B (5-30 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.2, 1.0, 2, 15
                    target_tp = 15.0
                else:
                    tier = "Bucket C (<5 Cr)"
                    pct_m, vol_m, min_c, base_w = 1.4, 1.0, 5, 25  # 5 dots in 25d
                    target_tp = 10.0  # High-hit 10% TP
            else:
                tier = "Bucket C (<5 Cr)"
                pct_m, vol_m, min_c, base_w = 1.4, 1.0, 5, 25
                target_tp = 10.0

            # 1. Trade Management & Exits
            if in_trade:
                gain = ((highs[i] - entry_price) / entry_price) * 100
                if gain > max_run_gain:
                    max_run_gain = gain

                # 50% Profit Booking & Move SL to Breakeven
                if max_run_gain >= partial_target_pct and not partial_booked:
                    partial_booked = True
                    active_sl = entry_price  # Breakeven for remainder

                # Trailing logic on remaining 50% position
                if "Bucket C" in active_bucket:
                    if max_run_gain >= 12.0 and i >= entry_idx + 8:
                        trail_15 = float(np.min(lows[i - 15 : i]))
                        if trail_15 > active_sl:
                            active_sl = trail_15
                elif "Bucket B" in active_bucket:
                    if max_run_gain >= 20.0 and i >= entry_idx + 20:
                        trail_20 = float(np.min(lows[i - 20 : i]))
                        if trail_20 > active_sl:
                            active_sl = trail_20
                else:
                    # Bucket A
                    if max_run_gain >= 25.0 and i >= entry_idx + 30:
                        trail_30 = float(np.min(lows[i - 30 : i]))
                        if trail_30 > active_sl:
                            active_sl = trail_30

                exit_triggered = False
                exit_reason = ""
                exit_price = closes[i]

                # Stop Loss / Trailing Hit
                if lows[i] <= active_sl:
                    exit_triggered = True
                    exit_reason = "Trailing / Breakeven SL Hit" if partial_booked else "Initial Swing SL Hit"
                    exit_price = min(closes[i], active_sl)
                
                # Bucket C Model 2B Climax Exit
                elif "Bucket C" in active_bucket and i > entry_idx + 2:
                    if gross_sma[i] > 0 and pct_50[i] > 0:
                        if gross_vols[i] >= (1.5 * gross_sma[i]) and pcts[i] <= (0.70 * pct_50[i]) and closes[i] <= (opens[i] * 1.002):
                            exit_triggered = True
                            exit_reason = "Climax Churn Exit"
                            exit_price = closes[i]

                # Bucket A Stagnation Exit
                elif "Bucket A" in active_bucket and (i - entry_idx) >= 30 and max_run_gain < 8.0:
                    exit_triggered = True
                    exit_reason = "Stagnation Time Exit (30D < 8%)"
                    exit_price = closes[i]

                if exit_triggered:
                    base_ret = ((exit_price - entry_price) / entry_price) * 100
                    
                    if partial_booked:
                        final_ret = round((partial_target_pct * 0.50) + (base_ret * 0.50), 2)
                    else:
                        final_ret = round(base_ret, 2)

                    holding_days = i - entry_idx
                    all_trades.append({
                        "Symbol": sym,
                        "Tier": active_bucket,
                        "Nifty Regime": entry_nifty_regime,
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Partial Booked": partial_booked,
                        "Exit Date": times[i],
                        "Exit Price": round(exit_price, 2),
                        "Return %": final_ret,
                        "Max Run Gain %": round(max_run_gain, 2),
                        "Rally 20%": bool(max_run_gain >= 20.0),
                        "Holding Days": holding_days,
                        "Exit Reason": exit_reason,
                        "Is Win": bool(final_ret > 0)
                    })
                    in_trade = False
                    cooldown_until = i + 5
                    continue

            # 2. Breakout Entry Logic
            if not in_trade and i > cooldown_until and i >= base_w:
                regime_now = nifty_regime.get(times[i], "Favourable")
                
                # Hard Gate: Bucket C only allowed in Favourable regimes
                if "Bucket C" in tier and regime_now == "Unfavourable":
                    continue

                # Hard Gate: Bucket C must be above Weekly 20 EMA proxy
                if "Bucket C" in tier and closes[i] < ema100[i]:
                    continue

                base_start = i - base_w
                qualifying = (pcts[base_start:i] >= (pct_m * pct_50[base_start:i])) & (d_vols[base_start:i] >= (vol_m * deliv_sma[base_start:i]))
                if np.sum(qualifying) >= min_c:
                    base_highs = highs[base_start:i]
                    sw_idx = int(np.argmax(base_highs))
                    sw_high = base_highs[sw_idx]
                    sw_obv = obvs[base_start + sw_idx]

                    if closes[i] > sw_high and closes[i - 1] <= sw_high and obvs[i] > sw_obv:
                        entry_price = closes[i]
                        pre_break_lookback = min(12, i - base_start)
                        recent_swing_low = float(np.min(lows[i - pre_break_lookback : i]))
                        active_sl = round(recent_swing_low * 0.995, 2)
                        
                        risk_pct = ((entry_price - active_sl) / entry_price) * 100

                        if risk_pct <= MAX_RISK_PCT:
                            if "Bucket C" in tier and entry_price < 20.0:
                                continue

                            active_bucket = tier
                            in_trade = True
                            entry_idx = i
                            entry_date = times[i]
                            entry_nifty_regime = regime_now
                            max_run_gain = 0.0
                            partial_booked = False
                            partial_target_pct = target_tp

    print(f"📊 Total Trades Processed: {len(all_trades)}")

    df_trades = pd.DataFrame(all_trades)
    segmented_results = {}

    def compute_stats(df_sub):
        if df_sub.empty:
            return {"Trades": 0, "Win Rate %": "0%", "+20% Rally %": "0%", "Avg Return %": "0%", "Profit Factor": 0.0, "Avg Hold": "0 d", "Score": 0.0}
        total_t = len(df_sub)
        wins = len(df_sub[df_sub["Is Win"] == True])
        win_rate = round((wins / total_t) * 100, 1)
        r20 = round((len(df_sub[df_sub["Rally 20%"] == True]) / total_t) * 100, 1)
        avg_ret = round(float(df_sub["Return %"].mean()), 2)
        
        gross_win = df_sub[df_sub["Is Win"] == True]["Return %"].sum()
        gross_loss = abs(df_sub[df_sub["Is Win"] == False]["Return %"].sum())
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
        avg_h = round(float(df_sub["Holding Days"].mean()), 1)
        score = round(win_rate * pf, 1)

        return {
            "Trades": total_t,
            "Win Rate %": f"{win_rate}%",
            "+20% Rally %": f"{r20}%",
            "Avg Return %": f"{'+' if avg_ret >= 0 else ''}{avg_ret}%",
            "Profit Factor": pf,
            "Avg Hold": f"{avg_h} d",
            "Score": score
        }

    for bucket in ["Bucket A (>30 Cr)", "Bucket B (5-30 Cr)", "Bucket C (<5 Cr)"]:
        b_df = df_trades[df_trades["Tier"] == bucket] if not df_trades.empty else pd.DataFrame()
        segmented_results[bucket] = {
            "Favourable (Nifty >= 50 SMA)": compute_stats(b_df[b_df["Nifty Regime"] == "Favourable"]) if not b_df.empty else compute_stats(pd.DataFrame()),
            "Unfavourable (Nifty < 50 SMA)": compute_stats(b_df[b_df["Nifty Regime"] == "Unfavourable"]) if not b_df.empty else compute_stats(pd.DataFrame()),
            "Combined (All Market Regimes)": compute_stats(b_df)
        }

    final_payload = {
        "Segmented Regime Summary": segmented_results,
        "Trade Log": all_trades
    }

    with open(OUTPUT_REPORT, "w") as fp:
        json.dump(final_payload, fp, indent=2)

    print(f"🎉 Updated High-Expectancy Backtest Report Generated! Saved to '{OUTPUT_REPORT}'.")

if __name__ == "__main__":
    run_segmented_backtest()
