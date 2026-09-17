import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADE_COLUMNS = [
    "ticker", "entry_date", "entry_price", "sl_price", 
    "tp_price", "exit_date", "exit_price", "target_pct", 
    "sl_pct", "outcome", "pnl_pct"
]

def load_universe():
    json_path = "data/nifty750.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            symbols = json.load(f)
    else:
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS"]

    tickers = []
    for s in symbols:
        clean = str(s).strip().replace("&", "%26")
        if not clean.endswith(".NS"):
            clean += ".NS"
        tickers.append(clean)
    return tickers

def run_smc_backtest(df, ticker):
    if df is None or len(df) < 40:
        return []

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c).capitalize() for c in df.columns]
    df = df.reset_index()

    date_col = next((c for c in df.columns if c.lower() in ['date', 'timestamp', 'index']), None)
    if not date_col:
        return []
    
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close']).reset_index(drop=True)
    if len(df) < 40:
        return []

    # 3-bar swing pivots
    df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['is_green'] = df['Close'] >= df['Open']

    trades = []
    n = len(df)
    i = 15

    while i < n - 15:
        prior_lows = df.loc[:i-2].loc[df['swing_low'], 'Low']
        if prior_lows.empty:
            i += 1
            continue
        key_low = prior_lows.iloc[-1]

        # Fake breakdown sweep breached and reclaimed
        sweep_window = df.loc[max(0, i-3):i]
        breached = (sweep_window['Low'] < key_low).any()
        reclaimed = df.loc[i, 'Close'] > key_low
        if not (breached and reclaimed):
            i += 1
            continue

        prior_highs = df.loc[:i-1].loc[df['swing_high'], 'High']
        if prior_highs.empty:
            i += 1
            continue
        major_high = prior_highs.iloc[-1]

        # CHoCH Expansion rally breaking swing high
        choch_idx = None
        for j in range(i + 1, min(i + 12, n)):
            if df.loc[j, 'Close'] > major_high:
                choch_idx = j
                break

        if choch_idx is None:
            i += 1
            continue

        high_idx = df.loc[i+1:min(choch_idx + 6, n - 1), 'High'].idxmax()
        displacement_high = df.loc[high_idx, 'High']

        # Bullish FVG detection in displacement leg
        fvg_entry, fvg_sl = None, None
        for k in range(i + 1, high_idx):
            if k + 1 >= n:
                break
            c1_high = df.loc[k - 1, 'High']
            c3_low = df.loc[k + 1, 'Low']
            c1_low = df.loc[k - 1, 'Low']

            if c3_low > c1_high:
                fvg_mid = (c3_low + c1_high) / 2.0
                target_entry = (c1_high + fvg_mid) / 2.0
                stop_loss = c1_low

                # If candle 1 low exceeds 5%, set stop to FVG bottom buffer
                if (target_entry - stop_loss) / target_entry > 0.05:
                    stop_loss = c1_high * 0.99

                target_pct = (displacement_high - target_entry) / target_entry
                sl_pct = (target_entry - stop_loss) / target_entry

                # Rule 8: Target >= 10% and SL <= 6%
                if target_pct >= 0.10 and sl_pct <= 0.06:
                    fvg_entry = target_entry
                    fvg_sl = stop_loss
                    break

        if fvg_entry is None:
            i = choch_idx + 1
            continue

        # Retracement into entry zone
        entry_idx = None
        for r in range(high_idx + 1, min(high_idx + 35, n)):
            if df.loc[r, 'Low'] <= fvg_entry:
                entry_idx = r
                break

        if entry_idx is None:
            i = high_idx + 1
            continue

        # Forward simulate outcome
        exit_price, exit_date, outcome = None, None, "OPEN"
        for f in range(entry_idx + 1, min(entry_idx + 65, n)):
            curr_low = df.loc[f, 'Low']
            curr_high = df.loc[f, 'High']

            if curr_low <= fvg_sl:
                outcome = "LOSS"
                exit_price = fvg_sl
                exit_date = str(df.loc[f, 'Date'])[:10]
                break
            elif curr_high >= displacement_high:
                outcome = "WIN"
                exit_price = displacement_high
                exit_date = str(df.loc[f, 'Date'])[:10]
                break

        pnl_pct = ((exit_price - fvg_entry) / fvg_entry * 100) if exit_price else 0.0

        trades.append({
            "ticker": ticker.replace(".NS", ""),
            "entry_date": str(df.loc[entry_idx, 'Date'])[:10],
            "entry_price": round(float(fvg_entry), 2),
            "sl_price": round(float(fvg_sl), 2),
            "tp_price": round(float(displacement_high), 2),
            "exit_date": exit_date,
            "exit_price": round(float(exit_price), 2) if exit_price else None,
            "target_pct": round(float(target_pct) * 100, 2),
            "sl_pct": round(float(sl_pct) * 100, 2),
            "outcome": outcome,
            "pnl_pct": round(float(pnl_pct), 2)
        })

        i = entry_idx + 6

    return trades

def main():
    tickers = load_universe()
    print(f"Loaded {len(tickers)} tickers. Starting download and SMC backtest...")

    all_trades = []
    # Test across universe in batches
    batch_size = 40
    for b in range(0, len(tickers), batch_size):
        batch = tickers[b:b+batch_size]
        try:
            data = yf.download(
                batch, 
                start="2021-01-01", 
                interval="1d", 
                auto_adjust=True, 
                progress=False
            )
            if data.empty:
                continue

            for sym in batch:
                try:
                    if len(batch) == 1:
                        df_stock = data.copy()
                    else:
                        df_stock = data.xs(sym, level='Ticker', axis=1) if 'Ticker' in data.columns.names else data[sym]
                    
                    df_stock = df_stock.dropna()
                    if len(df_stock) >= 40:
                        t = run_smc_backtest(df_stock, sym)
                        all_trades.extend(t)
                except Exception:
                    continue
        except Exception as e:
            print(f"Batch {b} failed: {e}")
            continue

    trades_df = pd.DataFrame(all_trades, columns=TRADE_COLUMNS)
    trades_df.to_csv(f"{OUTPUT_DIR}/smc_trades.csv", index=False)

    metrics = {
        "total_signals": int(len(trades_df)),
        "closed_trades": 0,
        "win_rate": 0,
        "net_cumulative_pnl": 0
    }

    if not trades_df.empty:
        closed = trades_df[trades_df['outcome'].isin(['WIN', 'LOSS'])]
        wins = closed[closed['outcome'] == 'WIN']
        losses = closed[closed['outcome'] == 'LOSS']
        if len(closed) > 0:
            metrics = {
                "total_signals": int(len(trades_df)),
                "closed_trades": int(len(closed)),
                "win_rate": round(len(wins) / len(closed) * 100, 2),
                "avg_win_pct": round(wins['pnl_pct'].mean(), 2) if not wins.empty else 0,
                "avg_loss_pct": round(losses['pnl_pct'].mean(), 2) if not losses.empty else 0,
                "profit_factor": round(abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()), 2) if (not losses.empty and losses['pnl_pct'].sum() != 0) else 0,
                "net_cumulative_pnl": round(closed['pnl_pct'].sum(), 2)
            }

    with open(f"{OUTPUT_DIR}/smc_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Backtest complete. Generated {len(all_trades)} trades.")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
