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
    n750_syms = set()
    if os.path.exists(NIFTY750_FILE):
        with open(NIFTY750_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
            items = d.keys() if isinstance(d, dict) else d
            n750_syms = {str(x).strip().upper() for x in items}
    print(f"Nifty 750 symbols loaded: {len(n750_syms)}")

    json_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if os.path.basename(f) not in EXCLUDED_FILES
    ]
    print(f"Total symbol JSON files found: {len(json_files)}\n")

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
                missing = required_cols - set(df.columns)
                anomalies.append((sym, f"Missing columns: {missing}"))
                continue

            df["date"] = df["time"].astype(str).str[:10]
            df = df.sort_values("date").drop_duplicates("date")
            for col in ["open", "high", "low", "close", "volume", "delivery_vol", "deliv_pct"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                else:
                    df[col] = 0.0

            n_rows = len(df)
            min_date = df["date"].min()
            max_date = df["date"].max()

            ohlc_ok = (
                (df["high"] >= df[["open", "close"]].max(axis=1) - 0.05) &
                (df["low"] <= df[["open", "close"]].min(axis=1) + 0.05) &
                (df["close"] > 0)
            )
            ohlc_valid_pct = ohlc_ok.mean() * 100

            nonzero_deliv_pct = (df["delivery_vol"] > 0).mean() * 100
            ratios = (df["delivery_vol"] / df["volume"].replace(0, np.nan)).dropna()
            unique_deliv_ratios = ratios.round(4).nunique()
            deliv_exceeds_vol = (df["delivery_vol"] > df["volume"] * 1.001).sum()
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

    min_list = sorted(audit_df["min_date"].dropna().tolist())
    max_list = sorted(audit_df["max_date"].dropna().tolist())
    mid = len(audit_df) // 2
    min_d = min_list[mid] if min_list else "N/A"
    max_d = max_list[mid] if max_list else "N/A"

    print("=== REPOSITORY AUTHENTICITY & COVERAGE SUMMARY ===")
    print(f"Valid parsed symbols:          {len(audit_df)}")
    print(f"Nifty 750 matched in data/:    {audit_df['is_n750'].sum()} / {len(n750_syms)}")
    print(f"Median trading days / symbol:  {audit_df['rows'].median():.0f}")
    print(f"Date span (median min -> max): {min_d} -> {max_d}")
    print(f"Symbols with >= 1000 rows:     {(audit_df['rows'] >= 1000).sum()}")
    print(f"Avg OHLC logical validity:     {audit_df['ohlc_valid_%'].mean():.2f}%")
    print(f"Avg true delivery coverage:    {audit_df['has_deliv_%'].mean():.2f}%")
    print(f"Suspect flat/synthetic files:  {(audit_df['unique_deliv_ratios'] < 10).sum()}")
    print(f"Corrupt/Empty files skipped:   {len(anomalies)}")

    if anomalies[:10]:
        print("\nSample anomalies:", anomalies[:10])
    print("\nSaved full per-symbol report to data/repo_data_audit.csv")

if __name__ == "__main__":
    audit_repository()
