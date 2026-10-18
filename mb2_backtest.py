import os
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf

def get_nse_symbols():
    sources = [
        "https://raw.githubusercontent.com/anirudha/nse-listed-companies/master/NSE-Symbols.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    ]
    for url in sources:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                sym_col = [c for c in df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()][0]
                symbols = df[sym_col].dropna().unique().tolist()
                clean = [f"{s.strip()}.NS" for s in symbols if s and not any(x in str(s) for x in ['&', ' ', '^'])]
                if len(clean) > 500:
                    print(f"Loaded {len(clean)} symbols from master list.")
                    return clean
        except Exception:
            continue

    print("Falling back to expanded liquid list...")
    return [
        "HFCL.NS", "DIXON.NS", "KAYNES.NS", "TRENT.NS", "BEL.NS", "HAL.NS", "BSE.NS", "MAZDOCK.NS",
        "RAILTEL.NS", "ATGL.NS", "TATAINVEST.NS", "ELECTCAST.NS", "PTC.NS", "TATACHEM.NS", "IIFLCAPS.NS",
        "GICRE.NS", "POLYPLEX.NS", "TORNTPOWER.NS", "STOVEKRAFT.NS", "RATEGAIN.NS", "JBMA.NS", "LUPIN.NS",
        "HERITGFOOD.NS", "RVNL.NS", "IRFC.NS", "COCHINSHIP.NS", "HUDCO.NS", "SUZLON.NS", "CDSL.NS", "MOTILALOFS.NS"
    ]

def download_ohlcv_batch(symbols, start_date="2020-01-01"):
    fetch_list = list(set(symbols + ["^NSEI"]))
    print(f"Downloading historical data for {len(fetch_list)} tickers from {start_date}...")
    data = yf.download(
        fetch_list,
        start=start_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )
    return data

def prepare_data_and_signals(raw_data, symbols, min_turnover_cr=10.0):
    clean_dfs = {}

    # Nifty 50 60-day Return Benchmark
    nifty_ret_60 = None
    try:
        if isinstance(raw_data.columns, pd.MultiIndex) and "^NSEI" in raw_data.columns.levels[0]:
            nifty_df = raw_data["^NSEI"].dropna().copy()
            nifty_ret_60 = nifty_df['Close'].pct_change(60)
    except Exception:
        nifty_ret_60 = None

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

            # Core Indicators
            df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7
            df['Turnover_SMA50'] = df['Turnover_Cr'].rolling(50).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['Vol_SMA20'] = df['Volume'].rolling(20).mean()

            # Baseline 1: 50-Day Avg Turnover >= 10 Crore
            c_turnover = df['Turnover_SMA50'] >= min_turnover_cr

            # Baseline 2: Stage 2 Trend (Close >= 200 SMA)
            c_stage2 = df['Close'] >= df['SMA200']

            # Baseline 3: 60-Day Relative Strength (Stock Return > Nifty Return)
            stock_ret_60 = df['Close'].pct_change(60)
            if nifty_ret_60 is not None:
                c_rs = stock_ret_60 > nifty_ret_60.reindex(df.index)
            else:
                c_rs = stock_ret_60 > 0.0

            # Base Delivery Accumulation (CLV Demat Proxy)
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
            clv = clv.fillna(0.0)
            demat_vol = np.maximum(0, clv * df['Volume'])
            up_demat = np.where(df['Close'] > df['Close'].shift(1), demat_vol, 0)
            down_demat = np.where(df['Close'] < df['Close'].shift(1), demat_vol, 0)
            up_15 = pd.Series(up_demat, index=df.index).rolling(15).sum()
            down_15 = pd.Series(down_demat, index=df.index).rolling(15).sum().replace(0, 1.0)
            df['Base_Delivery_Ratio'] = up_15 / down_15

            # Restored Swing Ultra Shelf (18-day window)
            shelf_len = 18
            df['Shelf_High'] = df['High'].shift(1).rolling(shelf_len).max()
            df['Shelf_Low'] = df['Low'].shift(1).rolling(shelf_len).min()
            df['Shelf_Spread'] = (df['Shelf_High'] - df['Shelf_Low']) / df['Shelf_Low']

            # Entry Criteria
            c_shelf_spread = df['Shelf_Spread'] <= 0.18
            c_base_accum = df['Base_Delivery_Ratio'] >= 1.15
            c_clear_shelf = (df['Close'] > df['Shelf_High']) & (df['High'] > df['Shelf_High'])
            c_vol_surge = df['Volume'] >= (1.40 * df['Vol_SMA20'])

            candle_range = (df['High'] - df['Low']).replace(0, np.nan)
            df['Range_Closeness'] = ((df['Close'] - df['Low']) / candle_range).fillna(0.0)
            c_range_closeness = df['Range_Closeness'] >= 0.65

            df['MB2_Signal'] = (
                c_turnover & c_stage2 & c_rs &
                c_shelf_spread & c_base_accum &
                c_clear_shelf & c_vol_surge & c_range_closeness
            )

            df['DayOfWeek'] = df.index.dayofweek
            clean_dfs[sym] = df
        except Exception:
            continue

    print(f"Qualified {len(clean_dfs)} liquid equities for MB2.")
    return clean_dfs

def run_mb2_backtest(clean_dfs):
    trades = []

    for sym, df in clean_dfs.items():
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        entry_date = None
        scaled_out = False
        highest_price = 0.0

        for i in range(len(df)):
            current_date = df.index[i]
            c = df['Close'].iloc[i]
            h = df['High'].iloc[i]
            l = df['Low'].iloc[i]
            o = df['Open'].iloc[i]
            ema50 = df['EMA50'].iloc[i]
            dow = df['DayOfWeek'].iloc[i]

            if not in_pos:
                if df['MB2_Signal'].iloc[i]:
                    base_low = df['Shelf_Low'].iloc[i]
                    calc_stop = min(l, base_low)
                    
                    # Capped stop: Base Low clamped to max 14% risk
                    max_stop_risk = c * 0.86
                    final_stop = max(calc_stop, max_stop_risk)
                    
                    if (c - final_stop) / c <= 0.14:
                        in_pos = True
                        entry_idx = i
                        entry_price = c
                        entry_date = current_date
                        stop_loss = final_stop
                        scaled_out = False
                        highest_price = c
            else:
                highest_price = max(highest_price, h)
                unrealized_gain = (highest_price - entry_price) / entry_price

                # Move stop to Breakeven at +15%
                if unrealized_gain >= 0.15 and stop_loss < entry_price:
                    stop_loss = entry_price

                # Take 25% profit at +25%
                if not scaled_out and unrealized_gain >= 0.25:
                    scaled_out = True

                exit_triggered = False
                exit_price = c
                exit_reason = ""

                # Stop loss / Breakeven hit
                if l <= stop_loss:
                    exit_triggered = True
                    exit_price = min(o, stop_loss)
                    exit_reason = "Stop Loss / BE"
                elif i > (entry_idx + 1):
                    # Exit on weekly close below 50 EMA
                    is_week_end = (dow == 4) or (i < len(df) - 1 and df['DayOfWeek'].iloc[i+1] <= dow)
                    if is_week_end and c < ema50:
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "Weekly Close < 50 EMA"

                if exit_triggered:
                    raw_gain = ((exit_price - entry_price) / entry_price) * 100
                    final_gain = (0.25 * 25.0 + 0.75 * raw_gain) if scaled_out else raw_gain

                    trades.append({
                        'Symbol': sym.replace('.NS', ''),
                        'Entry_Date': entry_date.strftime('%Y-%m-%d'),
                        'Exit_Date': current_date.strftime('%Y-%m-%d'),
                        'Entry_Price': round(entry_price, 2),
                        'Exit_Price': round(exit_price, 2),
                        'Gain_Pct': round(final_gain, 2),
                        'Holding_Days': (current_date - entry_date).days,
                        'Scaled_Out': scaled_out,
                        'Exit_Reason': exit_reason
                    })
                    in_pos = False

    return pd.DataFrame(trades)

def generate_report(trades_df):
    if trades_df.empty:
        summary_stats = pd.DataFrame([{
            'Strategy': 'MB2', 'Total Trades': 0, 'Win Rate %': 0.0,
            'Profit Factor': 0.0, 'Avg Return / Trade %': 0.0, 'Avg Winner %': 0.0,
            'Avg Loser %': 0.0, 'Max Gain %': 0.0, 'Avg Holding Days (All)': 0.0,
            'Avg Holding Days (Winners)': 0.0
        }])
        tier_df = pd.DataFrame({'Return Tier': [], 'Trade Count': [], 'Percentage': []})
        summary_stats.to_csv("mb2_summary.csv", index=False)
        tier_df.to_csv("mb2_tiers.csv", index=False)
        trades_df.to_csv("mb2_trades.csv", index=False)
        return

    total_trades = len(trades_df)
    winners = trades_df[trades_df['Gain_Pct'] > 0]
    losers = trades_df[trades_df['Gain_Pct'] <= 0]

    win_rate = (len(winners) / total_trades) * 100
    gross_p = winners['Gain_Pct'].sum()
    gross_l = abs(losers['Gain_Pct'].sum())
    profit_factor = (gross_p / gross_l) if gross_l > 0 else np.inf

    avg_return = trades_df['Gain_Pct'].mean()
    avg_winner = winners['Gain_Pct'].mean() if not winners.empty else 0
    avg_loser = losers['Gain_Pct'].mean() if not losers.empty else 0
    max_gain = trades_df['Gain_Pct'].max()
    avg_holding = trades_df['Holding_Days'].mean()
    avg_holding_win = winners['Holding_Days'].mean() if not winners.empty else 0

    # Profit Tier Classification
    bins = [-np.inf, 0, 20, 40, 60, 80, 100, 200, np.inf]
    labels = ['Loss (<0%)', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%', '100-200%', '>200%']
    trades_df['Tier'] = pd.cut(trades_df['Gain_Pct'], bins=bins, labels=labels)

    tier_counts = trades_df['Tier'].value_counts(sort=False)
    tier_pcts = (tier_counts / total_trades) * 100

    tier_df = pd.DataFrame({
        'Return Tier': tier_counts.index,
        'Trade Count': tier_counts.values,
        'Percentage': [f"{p:.2f}%" for p in tier_pcts.values]
    })

    summary_stats = pd.DataFrame([{
        'Strategy': 'MB2 (Swing Ultra Entry + 50 EMA Multibagger Engine)',
        'Total Trades': total_trades,
        'Win Rate %': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 2),
        'Avg Return / Trade %': round(avg_return, 2),
        'Avg Winner %': round(avg_winner, 2),
        'Avg Loser %': round(avg_loser, 2),
        'Max Gain %': round(max_gain, 2),
        'Avg Holding Days (All)': round(avg_holding, 1),
        'Avg Holding Days (Winners)': round(avg_holding_win, 1)
    }])

    summary_stats.to_csv("mb2_summary.csv", index=False)
    tier_df.to_csv("mb2_tiers.csv", index=False)
    trades_df.to_csv("mb2_trades.csv", index=False)

    print("\n" + "=" * 80)
    print("                  MB2 STRATEGY: CORE BACKTEST METRICS                  ")
    print("=" * 80)
    print(summary_stats.to_string(index=False))
    print("\n" + "-" * 80)
    print("                       GAIN TIER CLASSIFICATION                       ")
    print("-" * 80)
    print(tier_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    symbols = get_nse_symbols()
    raw_data = download_ohlcv_batch(symbols, "2020-01-01")
    dfs = prepare_data_and_signals(raw_data, symbols, min_turnover_cr=10.0)
    trades_df = run_mb2_backtest(dfs)
    generate_report(trades_df)
