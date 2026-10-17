import os
import requests
import io
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_all_nse_symbols():
    """Fetch active equity list directly from NSE's official archive."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        symbols = df['SYMBOL'].dropna().unique().tolist()
        # Add .NS suffix for Yahoo Finance
        return [f"{sym}.NS" for sym in symbols if sym and not any(c in sym for c in ['&', ' '])]
    except Exception as e:
        print(f"Error fetching symbols from NSE: {e}. Falling back to default liquid list.")
        return ["HFCL.NS", "DIXON.NS", "KAYNES.NS", "TRENT.NS", "BEL.NS", "HAL.NS", "BSE.NS", "MAZDOCK.NS"]

def download_ohlcv_batch(symbols, start_date="2020-01-01"):
    """Download historical data in bulk to optimize execution speed."""
    print(f"Downloading historical data for {len(symbols)} symbols from {start_date}...")
    data = yf.download(
        tickers=symbols,
        start=start_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )
    return data

def run_backtest(raw_data, symbols, min_turnover_cr=5.0):
    all_trades = []
    processed_count = 0

    for sym in symbols:
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if sym not in raw_data.columns.levels[0]:
                    continue
                df = raw_data[sym].dropna().copy()
            else:
                df = raw_data.dropna().copy()

            if len(df) < 220:
                continue

            processed_count += 1

            # 1. Turnovers & Moving Averages
            df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7
            df['Turnover_SMA50'] = df['Turnover_Cr'].rolling(50).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['Vol_SMA50'] = df['Volume'].rolling(50).mean()
            df['Vol_SMA5'] = df['Volume'].rolling(5).mean()

            # 2. Multibagger Base Conditions (20 to 60 trading days)
            base_window = 35
            df['Base_High'] = df['High'].shift(1).rolling(base_window).max()
            df['Base_Low'] = df['Low'].shift(1).rolling(base_window).min()
            df['Base_Depth'] = (df['Base_High'] - df['Base_Low']) / df['Base_High']

            # Filters:
            # - Liquidity: Turnover SMA50 >= min_turnover_cr
            # - Trend: Close > EMA50 > EMA200 & EMA200 rising over 20 sessions
            # - Base Depth: 10% to 20% (HFCL-type wide consolidation)
            # - Supply Dry-up: Prior 5-day volume < 60% of 50-day average
            # - Breakout: Close > Base_High with Volume >= 2.5x 50-day average
            c_liquidity = df['Turnover_SMA50'] >= min_turnover_cr
            c_trend = (df['Close'] > df['EMA50']) & (df['EMA50'] > df['EMA200']) & (df['EMA200'] > df['EMA200'].shift(20))
            c_base = (df['Base_Depth'] >= 0.10) & (df['Base_Depth'] <= 0.20)
            c_dryup = df['Vol_SMA5'].shift(1) < (0.60 * df['Vol_SMA50'].shift(1))
            c_breakout = (df['Close'] > df['Base_High']) & (df['Volume'] >= 2.5 * df['Vol_SMA50'])

            df['Entry_Signal'] = c_liquidity & c_trend & c_base & c_dryup & c_breakout

            # 3. Position Execution Loop
            in_pos = False
            entry_idx = 0
            entry_price = 0.0
            stop_loss = 0.0
            entry_date = None

            for i in range(len(df)):
                current_date = df.index[i]
                c = df['Close'].iloc[i]
                l = df['Low'].iloc[i]
                ema50 = df['EMA50'].iloc[i]

                if not in_pos:
                    if df['Entry_Signal'].iloc[i]:
                        in_pos = True
                        entry_idx = i
                        entry_price = c
                        entry_date = current_date
                        
                        # Stop set at base low; capped at 16% max risk
                        base_low = df['Base_Low'].iloc[i]
                        stop_loss = max(base_low, entry_price * 0.84)
                else:
                    exit_triggered = False
                    exit_price = c
                    exit_reason = ""

                    # Stop-loss check
                    if l <= stop_loss:
                        exit_triggered = True
                        exit_price = min(df['Open'].iloc[i], stop_loss)
                        exit_reason = "Base Stop Loss"
                    # Trailing trend check: Daily close below 50 EMA
                    elif c < ema50 and i > (entry_idx + 2):
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "Trailing 50 EMA Exit"

                    if exit_triggered:
                        pct_gain = ((exit_price - entry_price) / entry_price) * 100
                        holding_days = (current_date - entry_date).days

                        all_trades.append({
                            'Symbol': sym.replace('.NS', ''),
                            'Entry_Date': entry_date.strftime('%Y-%m-%d'),
                            'Exit_Date': current_date.strftime('%Y-%m-%d'),
                            'Entry_Price': round(entry_price, 2),
                            'Exit_Price': round(exit_price, 2),
                            'Gain_Pct': round(pct_gain, 2),
                            'Holding_Days': holding_days,
                            'Exit_Reason': exit_reason
                        })
                        in_pos = False

        except Exception as e:
            continue

    print(f"Processed {processed_count} qualified symbols.")
    return pd.DataFrame(all_trades)

def generate_report(trades_df):
    if trades_df.empty:
        summary_text = "### Multibagger Strategy Backtest\nNo trades were generated matching all parameters."
        print(summary_text)
        return summary_text

    total_trades = len(trades_df)
    winners = trades_df[trades_df['Gain_Pct'] > 0]
    losers = trades_df[trades_df['Gain_Pct'] <= 0]

    win_rate = (len(winners) / total_trades) * 100
    avg_gain = trades_df['Gain_Pct'].mean()
    avg_win = winners['Gain_Pct'].mean() if not winners.empty else 0
    avg_loss = losers['Gain_Pct'].mean() if not losers.empty else 0

    gross_profit = winners['Gain_Pct'].sum()
    gross_loss = abs(losers['Gain_Pct'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    max_gain = trades_df['Gain_Pct'].max()
    avg_holding = trades_df['Holding_Days'].mean()
    avg_holding_win = winners['Holding_Days'].mean() if not winners.empty else 0

    # Classify according to exact gain tiers
    bins = [-np.inf, 0, 30, 60, 100, 200, 300, np.inf]
    labels = ['Loss (<0%)', '0% - 30%', '30% - 60%', '60% - 100%', '100% - 200%', '200% - 300%', '300%+']
    trades_df['Tier'] = pd.cut(trades_df['Gain_Pct'], bins=bins, labels=labels)
    
    tier_counts = trades_df['Tier'].value_counts(sort=False)
    tier_pcts = (tier_counts / total_trades) * 100

    tier_table = pd.DataFrame({
        'Tier': tier_counts.index,
        'Count': tier_counts.values,
        'Percentage': [f"{p:.2f}%" for p in tier_pcts.values]
    })

    markdown_report = f"""
## Multibagger Momentum Strategy: Backtest Performance Report
* **Universe:** All NSE (Turnover >= ₹5-10 Cr)
* **Date Range:** 2020 to Present
* **Total Completed Trades:** {total_trades}

### Key Metrics
| Metric | Value |
| :--- | :--- |
| **Win Rate** | **{win_rate:.2f}%** |
| **Profit Factor (PF)** | **{profit_factor:.2f}** |
| **Average Return / Stock** | **{avg_gain:.2f}%** |
| **Average Win** | **+{avg_win:.2f}%** |
| **Average Loss** | **{avg_loss:.2f}%** |
| **Maximum Single Gain** | **+{max_gain:.2f}%** |
| **Average Holding Period (All)** | **{avg_holding:.1f} Calendar Days** |
| **Average Holding Period (Winners)** | **{avg_holding_win:.1f} Calendar Days** |

### Gain Classification Distribution
| Return Tier | Trade Count | % of Total Trades |
| :--- | :--- | :--- |
"""
    for _, row in tier_table.iterrows():
        markdown_report += f"| {row['Tier']} | {row['Count']} | {row['Percentage']} |\n"

    print(markdown_report)

    # Save outputs for GitHub artifacts and summary
    trades_df.to_csv("multibagger_trades.csv", index=False)
    with open("backtest_summary.md", "w") as f:
        f.write(markdown_report)

    return markdown_report

if __name__ == "__main__":
    symbols = get_all_nse_symbols()
    data = download_ohlcv_batch(symbols, start_date="2020-01-01")
    trades_df = run_backtest(data, symbols, min_turnover_cr=5.0)
    generate_report(trades_df)
