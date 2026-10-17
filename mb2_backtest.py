import os
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf

def get_all_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(r.text))
        symbols = df['SYMBOL'].dropna().unique().tolist()
        return [f"{s}.NS" for s in symbols if s and not any(c in s for c in ['&', ' '])]
    except Exception:
        return ["HFCL.NS", "DIXON.NS", "KAYNES.NS", "TRENT.NS", "BEL.NS", "HAL.NS", "BSE.NS", "MAZDOCK.NS"]

def download_ohlcv_batch(symbols, start_date="2020-01-01"):
    print(f"Downloading historical data for {len(symbols)} symbols...")
    data = yf.download(symbols, start=start_date, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
    return data

def prepare_market_data(raw_data, symbols, min_turnover_cr=5.0):
    clean_dfs = {}
    returns_3m = {}

    for sym in symbols:
        try:
            df = raw_data[sym].dropna().copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.dropna().copy()
            if len(df) < 220:
                continue

            # Core Technicals
            df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7
            df['Turnover_SMA50'] = df['Turnover_Cr'].rolling(50).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['Vol_SMA50'] = df['Volume'].rolling(50).mean()
            df['Vol_SMA5'] = df['Volume'].rolling(5).mean()

            # ATR (14) Calculation
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR14'] = tr.rolling(14).mean()

            # Base Setup
            base_window = 35
            df['Base_High'] = df['High'].shift(1).rolling(base_window).max()
            df['Base_Low'] = df['Low'].shift(1).rolling(base_window).min()
            df['Base_Depth'] = (df['Base_High'] - df['Base_Low']) / df['Base_High']

            # Base Triggers
            c_liq = df['Turnover_SMA50'] >= min_turnover_cr
            c_trend = (df['Close'] > df['EMA50']) & (df['EMA50'] > df['EMA200']) & (df['EMA200'] > df['EMA200'].shift(20))
            c_base = (df['Base_Depth'] >= 0.10) & (df['Base_Depth'] <= 0.20)
            c_dryup = df['Vol_SMA5'].shift(1) < (0.60 * df['Vol_SMA50'].shift(1))
            c_breakout = (df['Close'] > df['Base_High']) & (df['Volume'] >= 2.5 * df['Vol_SMA50'])

            df['Base_Signal'] = c_liq & c_trend & c_base & c_dryup & c_breakout
            df['Ret_3M'] = df['Close'].pct_change(63)

            clean_dfs[sym] = df
            returns_3m[sym] = df['Ret_3M']
        except Exception:
            continue

    # Compute Daily Cross-Sectional RS Percentile Rank (0 to 100)
    ret_matrix = pd.DataFrame(returns_3m)
    rs_percentiles = ret_matrix.rank(axis=1, pct=True) * 100

    for sym, df in clean_dfs.items():
        if sym in rs_percentiles.columns:
            df['RS_Rank'] = rs_percentiles[sym]

    return clean_dfs

def simulate_engine(clean_dfs, use_rs=False, trailing_type="21_EMA"):
    all_trades = []

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
            ema21 = df['EMA21'].iloc[i]
            ema50 = df['EMA50'].iloc[i]
            atr = df['ATR14'].iloc[i]

            if not in_pos:
                sig = df['Base_Signal'].iloc[i]
                if use_rs:
                    sig = sig and (df['RS_Rank'].iloc[i] >= 90.0)

                if sig:
                    in_pos = True
                    entry_idx = i
                    entry_price = c
                    entry_date = current_date
                    base_low = df['Base_Low'].iloc[i]
                    stop_loss = max(base_low, entry_price * 0.84)
                    scaled_out = False
                    highest_price = c
            else:
                highest_price = max(highest_price, h)
                unrealized_gain = (highest_price - entry_price) / entry_price

                # 1. Partial Scale-Out: 30% trim at +25%, move stop to Breakeven
                if not scaled_out and unrealized_gain >= 0.25:
                    scaled_out = True
                    stop_loss = max(stop_loss, entry_price)

                # 2. Dual-Speed Trailing Architecture
                exit_triggered = False
                exit_price = c
                exit_reason = ""

                # Stop loss hit
                if l <= stop_loss:
                    exit_triggered = True
                    exit_price = min(o, stop_loss)
                    exit_reason = "Stop Loss / BE"
                elif i > (entry_idx + 2):
                    if unrealized_gain < 0.50:
                        # Standard base trail: 50 EMA
                        if c < ema50:
                            exit_triggered = True
                            exit_price = c
                            exit_reason = "Trailing 50 EMA"
                    else:
                        # High-Speed Trail (> +50% gain)
                        if trailing_type == "21_EMA":
                            if c < ema21:
                                exit_triggered = True
                                exit_price = c
                                exit_reason = "Parabolic 21 EMA"
                        elif trailing_type == "2_ATR":
                            atr_stop = highest_price - (2.0 * atr)
                            if l <= atr_stop:
                                exit_triggered = True
                                exit_price = min(o, atr_stop)
                                exit_reason = "Parabolic 2 ATR"

                if exit_triggered:
                    # Blended realized gain (accounting for 30% scaled out at +25% if achieved)
                    raw_exit_gain = ((exit_price - entry_price) / entry_price) * 100
                    if scaled_out:
                        final_gain = (0.30 * 25.0) + (0.70 * raw_exit_gain)
                    else:
                        final_gain = raw_exit_gain

                    all_trades.append({
                        'Symbol': sym.replace('.NS', ''),
                        'Entry_Date': entry_date.strftime('%Y-%m-%d'),
                        'Exit_Date': current_date.strftime('%Y-%m-%d'),
                        'Gain_Pct': round(final_gain, 2),
                        'Holding_Days': (current_date - entry_date).days,
                        'Scaled_Out': scaled_out,
                        'Exit_Reason': exit_reason
                    })
                    in_pos = False

    return pd.DataFrame(all_trades)

def get_metrics(trades_df, name):
    if trades_df.empty:
        return {'Strategy': name, 'Trades': 0, 'Win Rate': '0%', 'PF': '0', 'Avg Gain': '0%', 'Max Gain': '0%', 'Avg Days': 0}
    winners = trades_df[trades_df['Gain_Pct'] > 0]
    losers = trades_df[trades_df['Gain_Pct'] <= 0]
    wr = (len(winners) / len(trades_df)) * 100
    gross_p = winners['Gain_Pct'].sum()
    gross_l = abs(losers['Gain_Pct'].sum())
    pf = (gross_p / gross_l) if gross_l > 0 else np.inf
    
    return {
        'Strategy': name,
        'Trades': len(trades_df),
        'Win Rate': f"{wr:.2f}%",
        'PF': f"{pf:.2f}",
        'Avg Gain': f"{trades_df['Gain_Pct'].mean():.2f}%",
        'Max Gain': f"+{trades_df['Gain_Pct'].max():.2f}%",
        'Avg Days': round(trades_df['Holding_Days'].mean(), 1)
    }

if __name__ == "__main__":
    symbols = get_all_nse_symbols()
    raw_data = download_ohlcv_batch(symbols, "2020-01-01")
    dfs = prepare_market_data(raw_data, symbols, min_turnover_cr=5.0)

    # Run Matrix Comparisons
    t1 = simulate_engine(dfs, use_rs=False, trailing_type="21_EMA")
    t2 = simulate_engine(dfs, use_rs=False, trailing_type="2_ATR")
    t3 = simulate_engine(dfs, use_rs=True, trailing_type="21_EMA")
    t4 = simulate_engine(dfs, use_rs=True, trailing_type="2_ATR")

    m = [
        get_metrics(t1, "No RS | 50 EMA -> 21 EMA"),
        get_metrics(t2, "No RS | 50 EMA -> 2 ATR"),
        get_metrics(t3, "RS >= 90 | 50 EMA -> 21 EMA"),
        get_metrics(t4, "RS >= 90 | 50 EMA -> 2 ATR")
    ]
    summary_df = pd.DataFrame(m)
    print(summary_df.to_string(index=False))

    # Save best candidate trades for viewing
    best_df = t3 if not t3.empty else t1
    best_df.to_csv("multibagger_trades.csv", index=False)
    summary_df.to_markdown("backtest_summary.md", index=False)
