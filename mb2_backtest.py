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
    data = yf.download(fetch_list, start=start_date, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
    return data

def prepare_data_and_signals(raw_data, symbols, min_turnover_cr=10.0):
    clean_dfs = {}

    # Nifty 50 60-day Return Benchmark
    try:
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
            c_
