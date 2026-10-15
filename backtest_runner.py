import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# --- 1. Dynamic NSE Universe Fetcher ---
def get_universes():
    print("📥 Fetching official NSE universes...")
    headers = {"User-Agent": "Mozilla/5.0"}
    n750_symbols = set()
    all_nse_symbols = set()

    # 1. Fetch Nifty 500 + MidCap 150 + SmallCap 250 (Nifty 750 Equivalent)
    for index_csv in ["ind_nifty500list.csv", "ind_niftysmallcap250list.csv", "ind_niftymicrocap250_list.csv"]:
        try:
            url = f"https://archives.nseindia.com/content/indices/{index_csv}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                for line in r.text.split("\n")[1:]:
                    p = line.split(",")
                    if len(p) > 2:
                        sym = p[2].strip().replace('"', "")
                        if sym:
                            n750_symbols.add(sym)
                            all_nse_symbols.add(sym)
        except Exception:
            pass

    # 2. Fetch full NSE equity list for Non-Nifty 750
    try:
        url_all = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        r = requests.get(url_all, headers=headers, timeout=10)
        if r.status_code == 200:
            for line in r.text.split("\n")[1:]:
                p = line.split(",")
                if len(p) > 1:
                    sym = p[0].strip().replace('"', "")
                    series = p[1].strip().replace('"', "")
                    if series == "EQ" and sym:
                        all_nse_symbols.add(sym)
    except Exception:
        pass

    if not all_nse_symbols:
        # Fallback to local repo list if offline
        for p in ["./data/nifty750.json", "./nifty750.json"]:
            if os.path.exists(p):
                with open(p, "r") as f:
                    n750_symbols = set(json.load(f))
                    all_nse_symbols = set(n750_symbols)
                break

    non_n750_symbols = all_nse_symbols - n750_symbols
    print(f"✅ Nifty 750 Universe: {len(n750_symbols)} stocks")
    print(f"✅ Non-Nifty 750 Universe: {len(non_n750_symbols)} stocks")
    return n750_symbols, non_n750_symbols

# --- 2. Simulation Logic (Aligned to Original Dashboards) ---
def simulate_symbol(df, is_swing3=False):
    if len(df) < 80:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values
    
    v_avgs = pd.Series(volumes).rolling(20).mean().values
    ema20s = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values

    # Swing 3 requires Stage 2 alignment (Close > SMA50 > SMA200)
    is_stage2 = (closes > sma50) & (sma50 > sma200)

    trades = []
    in_trade = False
    entry_price = 0.0
    current_stop = 0.0
    entry_idx = 0
    setup_type = ""
    consecutive_under_ema = 0

    for i in range(50, len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        v = volumes[i]
        v_avg = v_avgs[i] if not np.isnan(v_avgs[i]) and v_avgs[i] > 0 else 1.0
        ema = ema20s[i]

        if in_trade:
            # Trailing Exit: Stop loss breach OR confirmed 2-day close below 20-EMA
            if c < ema:
                consecutive_under_ema += 1
            else:
                consecutive_under_ema = 0

            # Trail stop upward: higher of initial stop or 5-day trough * 0.96
            recent_trough = lows[max(0, i - 5):i].min()
            current_stop = max(current_stop, recent_trough * 0.96)

            if l <= current_stop or consecutive_under_ema >= 2:
                exit_price = current_stop if l <= current_stop else c
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "setup": setup_type,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                    "is_win": pnl > 0
                })
                in_trade = False
                consecutive_under_ema = 0
            continue

        trend_ok = is_stage2[i] if is_swing3 else True
        if not trend_ok:
            continue

        # SETUP 1: LAUNCHPAD (Primary Engine - Evaluated First)
        max_spread = 0.08 if is_swing3 else 0.13
        shelf_high = highs[max(0, i - 15):i].max()
        shelf_low = lows[max(0, i - 15):i].min()
        shelf_spread = (shelf_high - shelf_low) / shelf_low if shelf_low > 0 else 1.0

        c_bo_lp = (c >= shelf_high) and (shelf_spread <= max_spread)
        c_vol_lp = (v >= 1.30 * v_avg) and (v >= 50000)

        if c_bo_lp and c_vol_lp:
            in_trade = True
            entry_price = shelf_high
            current_stop = shelf_low * 0.96
            entry_idx = i
            setup_type = "LAUNCHPAD"
            continue

        # SETUP 2: V-REVERSAL (Secondary Engine - Requires Deep Flush >= 18%)
        recent_20_high = highs[max(0, i - 20):i].max()
        flush_depth = (recent_20_high - l) / recent_20_high if recent_20_high > 0 else 0.0

        if flush_depth >= 0.18:
            was_under = any(lows[k] <= ema20s[k] * 1.005 for k in range(max(0, i - 4), i))
            prior_base_high = closes[max(0, i - 4):i].max()
            c_rec_vr = was_under and (c >= ema) and (c >= prior_base_high)
            c_vol_vr = (v >= 1.25 * v_avg) or (v >= 150000)

            if c_rec_vr and c_vol_vr:
                in_trade = True
                entry_price = c
                current_stop = lows[max(0, i - 5):i].min() * 0.955
                entry_idx = i
                setup_type = "V-REVERSAL"

    return trades

# --- 3. Main Multi-Universe Pipeline ---
def main():
    print("🚀 Initializing Restored Strategy Backtest (2021 – Present)...")
    n750_set, non_n750_set = get_universes()
    
    all_symbols = list(n750_set) + list(non_n750_set)
    print(f"Total Combined Universe: {len(all_symbols)} tickers.")

    buckets = {
        "Swing 3 - N750 Launchpad": [],
        "Swing 3 - N750 V-Reversal": [],
        "Run 2 - N750 Launchpad": [],
        "Run 2 - N750 V-Reversal": [],
        "Run 2 - Non-N750 Launchpad": [],
        "Run 2 - Non-N750 V-Reversal": []
    }

    # Batch download data in chunks of 60 tickers
    batch_size = 60
    total_batches = (len(all_symbols) + batch_size - 1) // batch_size

    for b in range(0, len(all_symbols), batch_size):
        chunk = all_symbols[b : b + batch_size]
        tickers = [f"{s}.NS" for s in chunk]
        print(f"Processing Batch {b // batch_size + 1}/{total_batches}...")

        try:
            data = yf.download(tickers, start="2021-01-01", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue

        for sym in chunk:
            try:
                t_str = f"{sym}.NS"
                if len(chunk) == 1:
                    df = data.copy()
                else:
                    if t_str not in data or data[t_str].empty:
                        continue
                    df = data[t_str].dropna()

                if len(df) < 80:
                    continue

                is_n750 = sym in n750_set

                # 1. Swing 3 (Strictly N750)
                if is_n750:
                    sw_trades = simulate_symbol(df, is_swing3=True)
                    for t in sw_trades:
                        if t["setup"] == "LAUNCHPAD":
                            buckets["Swing 3 - N750 Launchpad"].append(t)
                        else:
                            buckets["Swing 3 - N750 V-Reversal"].append(t)

                # 2. Run 2 (All Stocks)
                r2_trades = simulate_symbol(df, is_swing3=False)
                for t in r2_trades:
                    prefix = "Run 2 - N750" if is_n750 else "Run 2 - Non-N750"
                    tag = f"{prefix} {t['setup'].title()}"
                    if tag in buckets:
                        buckets[tag].append(t)
            except Exception:
                continue

    # --- 4. Report Generation ---
    rows = []
    for k, t_list in buckets.items():
        if not t_list:
            rows.append({
                "System & Universe": k, "Trades": 0, "Win Rate": "0.0%", "Profit Factor": 0.0,
                "Avg Gain": "0.0%", "Avg Days": 0.0, "Max Gain": "0.0%"
            })
            continue
        df_res = pd.DataFrame(t_list)
        wins = df_res[df_res["pnl"] > 0]
        losses = df_res[df_res["pnl"] <= 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0.0)

        rows.append({
            "System & Universe": k,
            "Trades": len(df_res),
            "Win Rate": f"{(len(wins) / len(df_res)) * 100:.1f}%",
            "Profit Factor": pf,
            "Avg Gain": f"{df_res['pnl'].mean() * 100:+.2f}%",
            "Avg Days": round(df_res["holding_days"].mean(), 1),
            "Max Gain": f"{df_res['pnl'].max() * 100:+.1f}%"
        })

    report_df = pd.DataFrame(rows)
    table_str = report_df.to_string(index=False)
    print("\n" + "="*85)
    print(table_str)
    print("="*85)

    os.makedirs("./backtest_results", exist_ok=True)
    report_df.to_csv("./backtest_results/latest_backtest.csv", index=False)
    with open("./backtest_results/latest_backtest.md", "w") as f:
        f.write("# Backtest Performance Summary (2021 – Present)\n\n")
        f.write(report_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
