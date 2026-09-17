import os
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_all_data_files():
    """Recursively search for all data files, ignoring hidden/git/results folders."""
    data_files = []
    ignore_dirs = {'.git', '.github', OUTPUT_DIR, '__pycache__'}
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if file.endswith(('.csv', '.parquet')) and not file.startswith('.'):
                # Avoid processing generated trade result files
                if "smc_trades" not in file:
                    data_files.append(os.path.join(root, file))
    return data_files

def load_stock_data(file_path):
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        df.columns = [c.strip().capitalize() for c in df.columns]
        
        date_col = next((c for c in df.columns if c.lower() in ['date', 'timestamp', 'time']), None)
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df = df[df['Date'] >= '2021-01-01']
        else:
            return None
        
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(set(df.columns)):
            return None
            
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=list(required_cols))

        return df
    except Exception:
        return None

def run_smc_backtest(df, ticker):
    if len(df) < 40:
        return []

    df = df.copy().reset_index(drop=True)
    trades = []
    
    df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['is_green'] = df['Close'] > df['Open']

    i = 15
    while i < len(df) - 10:
        prior_lows = df.loc[:i-2].loc[df['swing_low'], 'Low']
        if prior_lows.empty:
            i += 1
            continue
        recent_swing_low = prior_lows.iloc[-1]
        
        swept = (df.loc[i-1, 'Low'] < recent_swing_low and df.loc[i-1, 'Close'] > recent_swing_low) or \
                (df.loc[i, 'Low'] < recent_swing_low and df.loc[i, 'Close'] > recent_swing_low)
        if not swept:
            i += 1
            continue

        prior_highs = df.loc[:i-1].loc[df['swing_high'], 'High']
        if prior_highs.empty:
            i += 1
            continue
        prev_swing_high = prior_highs.iloc[-1]

        j = i + 1
        green_count = 0
        choch_index = -1
        while j < len(df) and df.loc[j, 'is_green']:
            green_count += 1
            if df.loc[j, 'Close'] > prev_swing_high and choch_index == -1:
                choch_index = j
            j += 1

        if choch_index == -1 or green_count < 2:
            i = j + 1
            continue

        displacement_high = df.loc[i+1:j-1, 'High'].max()
        high_idx = df.loc[i+1:j-1, 'High'].idxmax()

        fvg_found = False
        fvg_entry = None
        fvg_sl = None

        for k in range(i + 1, min(j - 1, len(df) - 1)):
            c1_high = df.loc[k - 1, 'High']
            c3_low = df.loc[k + 1, 'Low']
            c1_low = df.loc[k - 1, 'Low']

            if c3_low > c1_high:
                fvg_top = c3_low
                fvg_bottom = c1_high
                fvg_mid = (fvg_top + fvg_bottom) / 2.0
                
                target_entry = (fvg_bottom + fvg_mid) / 2.0
                stop_loss = c1_low

                target_pct = (displacement_high - target_entry) / target_entry
                sl_pct = (target_entry - stop_loss) / target_entry

                if target_pct >= 0.15 and sl_pct <= 0.05:
                    fvg_found = True
                    fvg_entry = target_entry
                    fvg_sl = stop_loss
                    break

        if not fvg_found:
            i = j + 1
            continue

        entry_idx = None
        for r in range(high_idx + 1, min(high_idx + 25, len(df))):
            if df.loc[r, 'Low'] <= fvg_entry:
                entry_idx = r
                break

        if entry_idx is None:
            i = j + 1
            continue

        target_price = displacement_high
        exit_price = None
        exit_date = None
        outcome = "OPEN"

        for f in range(entry_idx + 1, min(entry_idx + 60, len(df))):
            curr_low = df.loc[f, 'Low']
            curr_high = df.loc[f, 'High']

            if curr_low <= fvg_sl:
                outcome = "LOSS"
                exit_price = fvg_sl
                exit_date = str(df.loc[f, 'Date'])[:10]
                break
            elif curr_high >= target_price:
                outcome = "WIN"
                exit_price = target_price
                exit_date = str(df.loc[f, 'Date'])[:10]
                break

        pnl_pct = ((exit_price - fvg_entry) / fvg_entry * 100) if exit_price else 0.0

        trades.append({
            "ticker": ticker,
            "entry_date": str(df.loc[entry_idx, 'Date'])[:10],
            "entry_price": round(fvg_entry, 2),
            "sl_price": round(fvg_sl, 2),
            "tp_price": round(target_price, 2),
            "exit_date": exit_date,
            "exit_price": round(exit_price, 2) if exit_price else None,
            "target_pct": round(target_pct * 100, 2),
            "sl_pct": round(sl_pct * 100, 2),
            "outcome": outcome,
            "pnl_pct": round(pnl_pct, 2)
        })

        i = entry_idx + 5

    return trades

def main():
    files = find_all_data_files()
    print(f"Discovered {len(files)} potential data files across repository.")
    
    if not files:
        print("No stock data files found. Generating empty placeholder outputs.")
        pd.DataFrame().to_csv(f"{OUTPUT_DIR}/smc_trades.csv", index=False)
        with open(f"{OUTPUT_DIR}/smc_metrics.json", "w") as f:
            json.dump({"error": "No data files found"}, f)
        return

    all_trades = []
    processed = 0
    for file_path in files:
        ticker = os.path.basename(file_path).replace('.parquet', '').replace('.csv', '')
        df = load_stock_data(file_path)
        if df is None or len(df) < 40:
            continue
        
        trades = run_smc_backtest(df, ticker)
        all_trades.extend(trades)
        processed += 1

    print(f"Successfully processed {processed} stocks.")

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(f"{OUTPUT_DIR}/smc_trades.csv", index=False)

    metrics = {}
    if not trades_df.empty and 'outcome' in trades_df.columns:
        closed = trades_df[trades_df['outcome'].isin(['WIN', 'LOSS'])]
        wins = closed[closed['outcome'] == 'WIN']
        losses = closed[closed['outcome'] == 'LOSS']

        metrics = {
            "total_signals": int(len(trades_df)),
            "closed_trades": int(len(closed)),
            "win_rate": round(len(wins) / len(closed) * 100, 2) if len(closed) else 0,
            "avg_win_pct": round(wins['pnl_pct'].mean(), 2) if not wins.empty else 0,
            "avg_loss_pct": round(losses['pnl_pct'].mean(), 2) if not losses.empty else 0,
            "profit_factor": round(abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()), 2) if (not losses.empty and losses['pnl_pct'].sum() != 0) else 0,
            "net_cumulative_pnl": round(closed['pnl_pct'].sum(), 2)
        }

    with open(f"{OUTPUT_DIR}/smc_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Backtest Complete. Summary:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
