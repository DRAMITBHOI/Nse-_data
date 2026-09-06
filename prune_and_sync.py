import os
import io
import json
import time
import requests
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

RESERVED_FILES = {
    "fundamentals.json", "screener_results.json", "nifty750.json",
    "NIFTY50.json", "NIFTY.json", "fno_history.json",
    "wyckoff_screener_results.json", "obv_backtest_report.json",
    "scana_vs_absorption_report.json", "scana_candidates.json",
    "optimal_strategies.json", "scana_sensitivity_report.json",
    "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
    "scan_hp1_results.json", "scan_hp2_results.json", "scan_hp3_results.json",
    "backtest_hp3_report.json", "scan_macro_results.json", "macro_combo_leaderboard.json",
    "bucket_optimization_leaderboard.json", "backtest_bucket_report.json",
    "screener_macro_buckets_results.json", "scan_ultra_results.json",
    "init_progress.json", "backfill_progress.json"
}

def get_active_nse_universe():
    symbols = set()
    
    # 1. Mainboard Equities
    url_mb = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        r = requests.get(url_mb, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip().str.upper()
            symbols.update(df["SYMBOL"].dropna().astype(str).str.strip().str.upper().tolist())
            print(f"Loaded {len(symbols)} Mainboard symbols from EQUITY_L.csv")
    except Exception as e:
        print(f"Error fetching mainboard list: {e}")

    # 2. SME / Emerge Equities
    url_sme = "https://nsearchives.nseindia.com/content/equities/SME_EQUITY_L.csv"
    try:
        r_sme = requests.get(url_sme, headers=HEADERS, timeout=15)
        if r_sme.status_code == 200:
            df_sme = pd.read_csv(io.StringIO(r_sme.text))
            df_sme.columns = df_sme.columns.str.strip().str.upper()
            sme_syms = df_sme["SYMBOL"].dropna().astype(str).str.strip().str.upper().tolist()
            symbols.update(sme_syms)
            print(f"Loaded {len(sme_syms)} SME symbols from SME_EQUITY_L.csv")
    except Exception as e:
        print(f"Error fetching SME list: {e}")

    # Explicitly ensure target tickers are retained
    symbols.add("SUPREMEPWR")
    return symbols

def fetch_supremepwr_history():
    print("🚀 Fetching complete trading history for SUPREMEPWR...")
    out_file = os.path.join(DATA_DIR, "SUPREMEPWR.json")
    
    try:
        # yfinance ticker for NSE SME equity
        df = yf.download("SUPREMEPWR.NS", start="2024-01-01", progress=False, auto_adjust=False)
        if df.empty:
            print("Trying alternative symbol format...")
            df = yf.download("SUPREMEPWR.BO", start="2024-01-01", progress=False, auto_adjust=False)

        if not df.empty:
            records = []
            for idx, row in df.iterrows():
                # Handle potential multi-index column structures in modern yfinance
                c = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
                o = float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"])
                h = float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"])
                l = float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"])
                v = float(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"])
                
                if c <= 0:
                    continue

                # SME delivery percentage defaults to 100% (mandatory delivery / Trade-for-Trade)
                records.append({
                    "time": idx.strftime("%Y-%m-%d"),
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "volume": round(v),
                    "delivery_vol": round(v),
                    "deliv_pct": 100.0
                })

            with open(out_file, "w", encoding="utf-8") as fp:
                json.dump(records, fp, indent=2)
            print(f"✅ Successfully created SUPREMEPWR.json with {len(records)} daily candles!")
        else:
            print("⚠️ Notice: yfinance returned empty set for SUPREMEPWR.")
    except Exception as e:
        print(f"Error downloading SUPREMEPWR: {e}")

def prune_obsolete_symbols(active_symbols):
    print("🧹 Scanning data/ directory for dead or delisted symbols...")
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in RESERVED_FILES]
    print(f"Total files currently on disk: {len(all_files)}")

    removed_count = 0
    kept_count = 0

    for f in all_files:
        sym = f.replace(".json", "").strip().upper()
        file_path = os.path.join(DATA_DIR, f)

        # Retain all active mainboard and SME symbols
        if sym in active_symbols:
            kept_count += 1
            continue

        # For inactive symbols, inspect whether they traded in 2026
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                content = json.load(fp)
            
            # Prune if empty or last traded date was prior to 2026
            if not content or not isinstance(content, list):
                os.remove(file_path)
                removed_count += 1
                continue
            
            last_date = str(content[-1].get("time", ""))[:10]
            if last_date < "2026-01-01":
                os.remove(file_path)
                removed_count += 1
            else:
                kept_count += 1
        except Exception:
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception:
                pass

    print(f"🗑️ Pruned {removed_count} obsolete/delisted symbols.")
    print(f"📦 Active tracked universe now aligned: {kept_count} stocks.")

if __name__ == "__main__":
    active_universe = get_active_nse_universe()
    fetch_supremepwr_history()
    prune_obsolete_symbols(active_universe)
