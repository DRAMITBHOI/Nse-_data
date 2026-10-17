import os
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf

def get_nse_symbols():
    """Fetch active NSE symbols from multiple redundant sources."""
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
                    print(f"Loaded {len(clean)} NSE symbols successfully.")
                    return clean
        except Exception:
            continue

    # Comprehensive fallback of top 150 liquid momentum/swing mid & small caps
    print("Falling back to expanded universe...")
    return [
        "HFCL.NS", "DIXON.NS", "KAYNES.NS", "TRENT.NS", "BEL.NS", "HAL.NS", "BSE.NS", "MAZDOCK.NS",
        "RAILTEL.NS", "ATGL.NS", "TATAINVEST.NS", "ELECTCAST.NS", "PTC.NS", "TATACHEM.NS", "IIFLCAPS.NS",
        "GICRE.NS", "POLYPLEX.NS", "TORNTPOWER.NS", "STOVEKRAFT.NS", "RATEGAIN.NS", "JBMA.NS", "LUPIN.NS",
        "HERITGFOOD.NS", "RVNL.NS", "IRFC.NS", "COCHINSHIP.NS", "HUDCO.NS", "SUZLON.NS", "CDSL.NS", "MOTILALOFS.NS"
    ]

def download_ohlcv_batch(symbols, start_date="2020-01-01"):
    # Always include NIFTY benchmark for proper RS calculations
    fetch_list = list(set(symbols + ["^NSEI"]))
    print(f"Downloading historical data for {len(fetch_list)} tickers from {start_date}...")
    data = yf.download(
        tickers=fetch_list,
        start=start_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )
    return data

def prepare_market_data(raw_data, symbols, min_turnover_cr=5.0):
    clean_dfs = {}
    
    # Extract NIFTY 3M Returns
    try:
        nifty_df = raw_data["^NSEI"].dropna().copy()
        nifty_ret_3m = nifty_df['Close'].pct_change(63)
    except Exception:
        nifty_ret_3m = None

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

            df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7
            df['Turnover_SMA50'] = df['Turnover_Cr'].rolling(50).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['Vol_SMA50'] = df['Volume'].rolling(50).mean()
            df['Vol_SMA5'] = df['Volume'].rolling(5).mean()

            # ATR 14
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift()).abs(),
                (df['Low'] - df['Close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['ATR14'] = tr.rolling(14).mean()

            # Base parameters (35 days, 10% - 20% depth)
            base_window = 35
            df['Base_High'] = df['High'].shift(1).rolling(base_window).max()
            df['Base_Low'] = df['Low'].shift(1).rolling(base_window).min()
            df['Base_Depth'] = (df['Base_High'] - df['Base_Low']) / df['Base_High']

            # Base Setup Conditions
            c_liq = df['Turnover_SMA50'] >= min_turnover_cr
            c_trend = (df['Close'] > df['EMA50']) & (df['EMA50'] > df['EMA200']) & (df['EMA200'] > df['EMA200'].shift(20))
            c_base = (df['Base_Depth'] >= 0.10) & (df['Base_Depth'] <= 0.20)
            c_dryup = df['Vol_SMA5'].shift(1) < (0.60 * df['Vol_SMA50'].shift(1))
            c_breakout = (df['Close'] > df['Base_High']) & (df['Volume'] >= 2.5 * df['Vol_SMA50'])

            df['Base_Signal'] = c_liq & c_trend & c_base & c_dryup & c_breakout

            # Relative Strength Filter: Outperforming benchmark by >= 15% over 3 months
            stock_ret_3m = df['Close'].pct_change(63)
            if nifty_ret_3m is not None:
                df['RS_Pass'] = (stock_ret_3m - nifty_ret_3m.reindex(df.index)) >= 0.15
            else:
                df['RS_Pass'] = stock_ret_3m >= 0.20

            clean_dfs[sym] = df
        except Exception:
            continue

    print(f"Qualified {len(clean_dfs)} liquid symbols for simulation.")
    return clean_dfs

def simulate_variant(clean_dfs, strategy_name, use_rs=False, trailing_type="21_EMA"):
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
                    sig = sig and bool(df['RS_Pass'].iloc[i])

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

                # Scale out 30% at +25%, trail stop to Breakeven
                if not scaled_out and unrealized_gain >= 0.25:
                    scaled_out = True
                    stop_loss = max(stop_loss, entry_price)

                exit_triggered = False
                exit_price = c
                exit_reason = ""

                if l <= stop_loss:
                    exit_triggered = True
                    exit_price = min(o, stop_loss)
                    exit_reason = "Stop Loss / BE"
                elif i > (entry_idx + 2):
                    if unrealized_gain < 0.50:
                        if c < ema50:
                            exit_triggered = True
                            exit_price = c
                            exit_reason = "Trailing 50 EMA"
                    else:
                        if trailing_type == "21_EMA":
                            if c < ema21:
                                exit_triggered = True
                                exit_price = c
                                exit_reason = "Dual Speed 21 EMA"
                        elif trailing_type == "2_ATR":
                            atr_stop = highest_price - (2.0 * atr)
                            if l <= atr_stop:
                                exit_triggered = True
                                exit_price = min(o, atr_stop)
                                exit_reason = "Dual Speed 2 ATR"

                if exit_triggered:
                    raw_exit_gain = ((exit_price - entry_price) / entry_price) * 100
                    final_gain = (0.30 * 25.0 + 0.70 * raw_exit_gain) if scaled_out else raw_exit_gain

                    all_trades.append({
                        'Strategy': strategy_name,
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

if __name__ == "__main__":
    symbols = get_nse_symbols()
    raw_data = download_ohlcv_batch(symbols, "2020-01-01")
    dfs = prepare_market_data(raw_data, symbols, min_turnover_cr=5.0)

    # Run the 4 Matrix Variants
    v1 = simulate_variant(dfs, "1. No RS | 50 -> 21 EMA", use_rs=False, trailing_type="21_EMA")
    v2 = simulate_variant(dfs, "2. No RS | 50 -> 2 ATR", use_rs=False, trailing_type="2_ATR")
    v3 = simulate_variant(dfs, "3. RS>=15% | 50 -> 21 EMA", use_rs=True, trailing_type="21_EMA")
    v4 = simulate_variant(dfs, "4. RS>=15% | 50 -> 2 ATR", use_rs=True, trailing_type="2_ATR")

    all_trades_df = pd.concat([v1, v2, v3, v4], ignore_index=True)
    all_trades_df.to_csv("mb2_all_trades.csv", index=False)

    summary_rows = []
    for name, t_df in [
        ("1. No RS | 50 -> 21 EMA", v1),
        ("2. No RS | 50 -> 2 ATR", v2),
        ("3. RS>=15% | 50 -> 21 EMA", v3),
        ("4. RS>=15% | 50 -> 2 ATR", v4)
    ]:
        if t_df.empty:
            continue
        winners = t_df[t_df['Gain_Pct'] > 0]
        losers = t_df[t_df['Gain_Pct'] <= 0]
        wr = (len(winners) / len(t_df)) * 100
        gross_p = winners['Gain_Pct'].sum()
        gross_l = abs(losers['Gain_Pct'].sum())
        pf = (gross_p / gross_l) if gross_l > 0 else np.inf
        mb_count = len(t_df[t_df['Gain_Pct'] >= 100.0])

        summary_rows.append({
            'Strategy': name,
            'Trades': len(t_df),
            'Win Rate %': round(wr, 2),
            'Profit Factor': round(pf, 2),
            'Avg Return %': round(t_df['Gain_Pct'].mean(), 2),
            'Avg Winner %': round(winners['Gain_Pct'].mean(), 2) if not winners.empty else 0,
            'Avg Loser %': round(losers['Gain_Pct'].mean(), 2) if not losers.empty else 0,
            'Multibaggers (>=100%)': mb_count,
            'Max Gain %': round(t_df['Gain_Pct'].max(), 2),
            'Avg Holding Days': round(t_df['Holding_Days'].mean(), 1)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("mb2_comparison_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
