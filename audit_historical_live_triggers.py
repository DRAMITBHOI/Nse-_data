import os
import json
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "live_scanner_full_audit.json")

# Core universes from your master configuration
N750_FILE = os.path.join(DATA_DIR, "nifty750.json")

def load_tickers():
    n750 = []
    if os.path.exists(N750_FILE):
        try:
            with open(N750_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, list):
                    n750 = [str(x).strip().upper() for x in d]
                elif isinstance(d, dict):
                    n750 = [str(x).strip().upper() for x in d.get("symbols", d.get("data", []))]
        except Exception:
            pass

    # Filter out cash/liquid ETFs
    n750 = [s for s in n750 if not s.endswith("BEES") and s not in ["LIQUIDCASE", "LIQGRWBEES"]]
    if not n750:
        n750 = ["NAVINFLUOR", "IGL", "PETRONET", "DIXON", "TRENT", "POLYCAB", "KAYNES", "PERSISTENT", "HAL", "BEL"]

    # Top high-beta / non-N750 breakout universe
    non_n750 = [
        "MANORAMA", "PREMEXPLN", "PRUDENT", "SIGNPOST", "DOLPHIN", "PITTIENG",
        "AVANTIFEED", "GEOJITFSL", "KRN", "GRAVITA", "NEULANDLAB", "GANDHAR"
    ]
    return n750[:75], non_n750

def run_historical_audit():
    n750_list, non_n750_list = load_tickers()
    all_targets = [(s, "NIFTY 750") for s in n750_list] + [(s, "NON-N750") for s in non_n750_list]
    
    print(f"🚀 Starting 5-Year Historical Live Trigger Audit for {len(all_targets)} symbols...")
    ticker_str = " ".join([f"{s}.NS" for s, _ in all_targets])
    
    # Download 5 years of daily data via fast multithreaded cloud connection
    data = yf.download(ticker_str, period="5y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    
    events = []

    for sym, universe in all_targets:
        yf_sym = f"{sym}.NS"
        try:
            df = data[yf_sym].dropna(how="all").copy()
            if len(df) < 60:
                continue

            df.columns = [c.lower() for c in df.columns]
            df["sma20_vol"] = df["volume"].rolling(20).mean()
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["shelf_high_20"] = df["high"].shift(1).rolling(20).max()
            df["shelf_low_20"] = df["low"].shift(1).rolling(20).min()

            for i in range(30, len(df)):
                prev = df.iloc[i - 1]
                curr = df.iloc[i]

                shelf_height_pct = ((prev["shelf_high_20"] - prev["shelf_low_20"]) / prev["shelf_low_20"]) * 100
                is_launchpad = (shelf_height_pct <= 14.0) and (prev["close"] >= prev["ema20"])
                is_v_reversal = (not is_launchpad) and (prev["close"] > prev["ema20"]) and (df.iloc[i - 3]["low"] < prev["ema20"] * 0.96)

                if not (is_launchpad or is_v_reversal):
                    continue

                setup_type = "LAUNCHPAD" if is_launchpad else "V-REVERSAL"
                pivot = prev["shelf_high_20"]

                # Intraday touch trigger
                if curr["high"] >= pivot:
                    closed_above = bool(curr["close"] >= pivot)
                    vol_expanded = bool(curr["volume"] >= (1.5 * prev["sma20_vol"]))
                    true_breakout = bool(closed_above and vol_expanded)

                    fwd_5d_ret = None
                    if i + 5 < len(df):
                        fwd_5d_ret = round(float((df.iloc[i + 5]["close"] - pivot) / pivot) * 100.0, 2)

                    events.append({
                        "symbol": sym,
                        "date": str(curr.name)[:10] if hasattr(curr, "name") else "",
                        "universe": universe,
                        "setup": setup_type,
                        "pivot": float(pivot),
                        "closed_above": closed_above,
                        "vol_expanded": vol_expanded,
                        "true_breakout": true_breakout,
                        "fwd_5d_return_%": fwd_5d_ret
                    })
        except Exception:
            continue

    if not events:
        print("❌ No trigger events identified.")
        return

    ev_df = pd.DataFrame(events)
    summary_rows = []

    for (univ, stp), grp in ev_df.groupby(["universe", "setup"]):
        total_triggers = len(grp)
        closed_above_cnt = int(grp["closed_above"].sum())
        true_breakout_cnt = int(grp["true_breakout"].sum())

        wick_failure_rate = round(((total_triggers - closed_above_cnt) / total_triggers) * 100, 1)
        true_accuracy_rate = round((true_breakout_cnt / total_triggers) * 100, 1)
        avg_5d_win = grp[grp["true_breakout"]]["fwd_5d_return_%"].mean()
        avg_5d_loss = grp[~grp["closed_above"]]["fwd_5d_return_%"].mean()

        summary_rows.append({
            "Universe": univ,
            "Setup": stp,
            "Total_Triggers": total_triggers,
            "Closed_Above_Pivot": closed_above_cnt,
            "Closed_Above_Pct": round(100 - wick_failure_rate, 1),
            "True_Breakout_Count": true_breakout_cnt,
            "True_Breakout_Pct": true_accuracy_rate,
            "Wick_Rejection_Pct": wick_failure_rate,
            "Avg_5D_Win_Return_%": round(float(avg_5d_win), 2) if pd.notna(avg_5d_win) else 0.0,
            "Avg_5D_Wick_Loss_%": round(float(avg_5d_loss), 2) if pd.notna(avg_5d_loss) else 0.0,
        })

    payload = {
        "Audit_Date": str(pd.Timestamp.now()),
        "Total_Triggers_Analyzed": len(ev_df),
        "Summary": summary_rows,
        "Sample_Triggers": events[-50:]
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Audit completed! Analyzed {len(ev_df)} historical triggers. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_historical_audit()
