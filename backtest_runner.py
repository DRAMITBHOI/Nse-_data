import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# --- 1. Universe Definition ---
def get_universes():
    # Attempt loading local Nifty 750 list if in repo
    n750 = []
    for p in ["./data/nifty750.json", "./nifty750.json", "./Codes/nifty750.json"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    n750 = json.load(f)
                break
            except Exception:
                pass
    
    # If not present, download the official Nifty 500 / Microcap list or define core fallbacks
    if not n750:
        try:
            url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                lines = resp.text.split("\n")
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) > 2:
                        n750.append(parts[2].strip().replace('"', ""))
        except Exception:
            pass

    if not n750:
        # Standard representative core symbols
        n750 = ["TRENT", "HAL", "ZOMATO", "TATAELXSI", "BEL", "DIXON", "POLYCAB", "BAJAJHLDNG"]

    return set(n750)

# --- 2. Simulation Logic (2021 - Present) ---
def simulate_symbol(df, is_stage_filtered=False):
    if len(df) < 60:
        return []

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values
    
    # Precompute indicators via vectorized numpy/pandas
    v_avgs = pd.Series(volumes).rolling(20).mean().values
    ema20s = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values

    is_stage2 = (closes > sma50) & (sma50 > sma200)

    trades = []
    in_trade = False
    entry_price = 0.0
    current_stop = 0.0
    entry_idx = 0
    setup_type = ""

    for i in range(50, len(df)):
        c = closes[i]
        l = lows[i]
        v = volumes[i]
        v_avg = v_avgs[i] if not np.isnan(v_avgs[i]) and v_avgs[i] > 0 else 1.0
        ema = ema20s[i]

        if in_trade:
            # Trailing stop or 20-EMA breakdown
            if l <= current_stop or c < ema:
                exit_price = current_stop if l <= current_stop else c
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    "setup": setup_type,
                    "pnl": pnl,
                    "holding_days": i - entry_idx,
                    "is_win": pnl > 0
                })
                in_trade = False
            else:
                recent_trough = lows[max(0, i - 5):i].min()
                current_stop = max(current_stop, recent_trough * 0.955)
            continue

        # Condition: Launchpad
        coil_high = highs[max(0, i - 15):i].max()
        coil_low = lows[max(0, i - 15):i].min()
        coil_w = (coil_high - coil_low) / coil_low if coil_low > 0 else 1.0
        c_bo_lp = (c >= coil_high) and (coil_w <= 0.12)
        c_vol_lp = (v >= 1.30 * v_avg)

        # Condition: V-Reversal
        was_under = any(lows[k] <= ema20s[k] * 1.005 for k in range(max(0, i - 4), i))
        prior_base_high = closes[max(0, i - 4):i].max()
        c_rec_vr = was_under and (c >= ema) and (c >= prior_base_high)
        c_vol_vr = (v >= 1.25 * v_avg) or (v >= 150000)

        # Stage gate
        trend_allowed = is_stage2[i] if is_stage_filtered else True

        if trend_allowed:
            if c_rec_vr and c_vol_vr:
                in_trade = True
                entry_price = c
                current_stop = lows[max(0, i - 5):i].min() * 0.955
                entry_idx = i
                setup_type = "V-REVERSAL"
            elif c_bo_lp and c_vol_lp:
                in_trade = True
                entry_price = coil_high
                current_stop = coil_low * 0.955
                entry_idx = i
                setup_type = "LAUNCHPAD"

    return trades

# --- 3. Main Runner ---
def main():
    print("🚀 Initializing Cloud Backtest Engine (2021 – Present)...")
    n750_set = get_universes()
    
    # Load all symbol targets
    symbols_file = "./all_symbols.json"
    if os.path.exists(symbols_file):
        with open(symbols_file, "r") as f:
            all_symbols = json.load(f)
    else:
        # Fallback list if all_symbols.json is not explicitly committed
        all_symbols = list(n750_set) + ["SIRCA", "RAIN", "NITINSPIN", "TANLA", "STOVEKRAFT", "BEPL", "CEIGALL"]

    # Filter distinct
    all_symbols = sorted(list(set(all_symbols)))
    print(f"Total Universe: {len(all_symbols)} tickers.")

    buckets = {
        "Swing 3 - N750 V-Reversal": [],
        "Swing 3 - N750 Launchpad": [],
        "Run 2 - N750 V-Reversal": [],
        "Run 2 - N750 Launchpad": [],
        "Run 2 - Non-N750 V-Reversal": [],
        "Run 2 - Non-N750 Launchpad": []
    }

    # Batch download data via yfinance
    batch_size = 50
    for b in range(0, len(all_symbols), batch_size):
        chunk = all_symbols[b : b + batch_size]
        tickers = [f"{s}.NS" for s in chunk]
        print(f"Fetching batch {b // batch_size + 1}/{(len(all_symbols) + batch_size - 1) // batch_size}...")

        try:
            data = yf.download(tickers, start="2021-01-01", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print(f"Batch fetch error: {e}")
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

                # 1. Swing 3 (Only Nifty 750)
                if is_n750:
                    sw_trades = simulate_symbol(df, is_stage_filtered=True)
                    for t in sw_trades:
                        if t["setup"] == "V-REVERSAL":
                            buckets["Swing 3 - N750 V-Reversal"].append(t)
                        else:
                            buckets["Swing 3 - N750 Launchpad"].append(t)

                # 2. Run 2 (Global across all stocks)
                r2_trades = simulate_symbol(df, is_stage_filtered=False)
                for t in r2_trades:
                    prefix = "Run 2 - N750" if is_n750 else "Run 2 - Non-N750"
                    tag = f"{prefix} {t['setup'].title()}"
                    if tag in buckets:
                        buckets[tag].append(t)
            except Exception:
                continue

    # --- 4. Metric Compilation ---
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

    # Save to disk
    os.makedirs("./backtest_results", exist_ok=True)
    report_df.to_csv("./backtest_results/latest_backtest.csv", index=False)
    with open("./backtest_results/latest_backtest.md", "w") as f:
        f.write("# Backtest Performance Summary (2021 – Present)\n\n")
        f.write(report_df.to_markdown(index=False))

    # Optional Telegram Notification
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        tg_text = f"📊 *Backtest Summary (2021–Present)*\n```\n{table_str}\n```"
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": tg_text, "parse_mode": "Markdown"},
                timeout=10
            )
        except Exception:
            pass

if __name__ == "__main__":
    main()
