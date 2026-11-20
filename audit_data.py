import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")

EXCLUDED_FILES = {
    "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
    "fundamentals.json", "backtest_results.json", "swing3_results.json",
    "swing3_run2_results.json", "swing3_run2_global_results.json",
    "swing3_vs_run2_nifty750_results.json", "swing3_run2_non_nifty750_results.json",
    "swing3_scanner_results.json", "swing3_run2_non_nifty750_scanner_results.json",
    "swing_ultra750_results.json", "swing_ultraN750_results.json",
    "live_intraday_breakouts.json", "live_intraday_breakouts_v2.json",
    "accuracy_audit_log.json"
}

def audit_repository():
    # 1. Check Nifty 750 mapping
    n750_syms = set()
    if os.path.exists(NIFTY750_FILE):
        with open(NIFTY750_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
            n750_syms = {str(x).strip().upper() for x in (d.keys() if isinstance(d, dict) else d)}
    print(f"📋 Nifty 750 symbols loaded: {len(n750_syms)}")

    json_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if os.path.basename(f) not in EXCLUDED_FILES
    ]
    print(f"📁 Total symbol JSON files found: {len(json_files)}\n")

    records = []
    anomalies = []

    for path in sorted(json_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list) or len(raw) == 0:
                anomalies.append((sym, "Empty or non-list JSON"))
                continue

            df = pd.DataFrame(raw)
            required_cols = {"time", "open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                anomalies.append((sym, f"Missing columns: {required_cols - set(df.columns)}"))
                continue

            df["date"] = df["time"].astype(str).str[:10]
            df = df.sort_values("date").drop_duplicates("date")
            for col in ["open", "high", "low", "close", "volume", "delivery_vol", "deliv_pct"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                else:
                    df[col] = 0.0

            # Authenticity & Sanity Checks
            n_rows = len(df)
            min_date = df["date"].min()
            max_date = df["date"].max()
            
            # Check OHLC logical consistency (High >= max(Open, Close) and Low <= min(Open, Close))
            ohlc_valid_pct = (
                (df["high"] >= df[["open", "close"]].max(axis=1) - 0.05) &
                (df["low"] <= df[["open", "close"]].min(axis=1) + 0.05) &
                (df["close"] > 0)
            ).mean() * 100

            # Check Delivery Data Authenticity (real NSE delivery varies daily, not constant 0 or 50%)
            nonzero_deliv_pct = (df["delivery_vol"] > 0).mean() * 100
            unique_deliv_ratios = (df["delivery_vol"] / df["volume"].replace(0, np.nan)).dropna().round(4).nunique()
            deliv_exceeds_vol = (df["delivery_vol"] > df["volume"] * 1.001).sum()

            # Check if prices look synthetic (e.g., exact constant returns or zero variance)
            ret_std = df["close"].pct_change().dropna().std()

            records.append({
                "symbol": sym,
                "is_n750": sym in n750_syms,
                "rows": n_rows,
                "min_date": min_date,
                "max_date": max_date,
                "ohlc_valid_%": round(ohlc_valid_pct, 2),
                "has_deliv_%": round(nonzero_deliv_pct, 2),
                "unique_deliv_ratios": unique_deliv_ratios,
                "deliv_gt_vol": deliv_exceeds_vol,
                "daily_ret_std": round(ret_std, 4) if pd.notnull(ret_std) else 0.0
            })
        except Exception as e:
            anomalies.append((sym, f"Read error: {str(e)[:60]}"))

    audit_df = pd.DataFrame(records)
    audit_df.to_csv(os.path.join(DATA_DIR, "repo_data_audit.csv"), index=False)

    min_d = sorted(audit_df["min_date"].dropna())[len(audit_df) // 2] if len(
