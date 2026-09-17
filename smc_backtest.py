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
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

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

    # Reset index and clean column casing
    df = df.copy().reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower() in ['date', 'timestamp', 'index']), None)
    if not date_col:
        return []
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close']).reset_index(drop=True)

    trades = []
    df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    df['is_green'] = df['Close'] >= df['Open']

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
        choch_index = -1
        while j < len(df) and (df.loc[i+1:j, 'is_green'].mean() >= 0.70):
            if df.loc[j, 'Close'] > prev_swing_high and choch_index == -1:
                choch_index = j
            if j - i > 12:
                break
            j += 1

        if choch_index == -1:
            i += 1
            continue

        displacement_high = df.loc[i+1:j, 'High'].max()
        high_idx = df.loc[i+1:j, 'High'].idxmax()

        fvg_entry, fvg_sl = None, None
        for k in range(i + 1, min(j - 1, len(df) - 1)):
            c1_high = df.loc[k - 1, 'High']
            c3_low = df.loc[k + 1, 'Low']
            c1_low = df.loc[k - 1, 'Low']

            if c3_low > c1_high:
                fvg_mid = (c3_low + c1_high) / 2.0
                target_entry = (c1_high + fvg_mid) / 2.0
                stop_loss = c1_low

                target_pct = (displacement_high - target_entry) / target_entry
                sl_pct = (target_entry - stop_loss) / target_entry

                if target_pct >= 0.15 and sl_pct <= 0.05:
                    fvg_entry = target_entry
                    fvg_sl = stop_loss
                    break

        if fvg_entry is None:
            i = j + 1
            continue

        entry_idx = None
        for r in range(high_idx + 1, min(high_idx + 30, len(df))):
            if df.loc[r, 'Low'] <= fvg_entry:
                entry_idx = r
                break

        if entry_idx is None:
            i = j + 1
            continue

        exit_price, exit_date, outcome = None, None, "OPEN"
        for f in range(entry_idx + 1, min(entry_idx + 60, len(df))):
            if df.loc[f, 'Low'] <= fvg_sl:
                outcome = "LOSS"
                exit_price = fvg_sl
                exit_date = str(df.loc[f, 'Date'])[:10]
                break
            elif df.loc[f, 'High'] >= displacement_high:
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
        i = entry_idx + 5

    return trades

def main():
    tickers = load_universe()
    print(f"Loaded {len(tickers)} tickers from data/nifty750.json.")

    all_trades = []
    batch_size = 50
    for b in range(0, len(tickers), batch_size):
        batch = tickers[b:b+batch_size]
        try:
            data = yf.download(batch, start="2021-01-01", interval="1d", group_by='ticker', progress=False, threads=True)
            for sym in batch:
                try:
                    df = data[sym].dropna() if len(batch) > 1 else data.dropna()
                    if df.empty or len(df) < 40:
                        continue
                    trades = run_smc_backtest(df, sym)
                    all_trades.extend(trades)
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

    print(f"Completed backtest. Found {len(all_trades)} trades across universe.")

if __name__ == "__main__":
    main()
