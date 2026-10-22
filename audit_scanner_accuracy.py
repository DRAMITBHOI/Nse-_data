import os
import glob
import json
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "accuracy_audit_log.json")
V1_LOG = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
V2_LOG = os.path.join(DATA_DIR, "live_intraday_breakouts_v2.json")

def run_comprehensive_audit():
    today_str = datetime.now().strftime("%Y-%m-%d")
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    
    print("=" * 105)
    print(f"🔍 COMPILING MULTI-DAY ACCURACY AUDIT ARCHIVE ({today_str})")
    print("=" * 105)

    # 1. Ground Truth EOD Setups Today
    ground_truth_positives = set()
    ground_truth_negatives = set()

    for f_name in ["swing_ultra750_results.json", "swing_ultraN750_results.json"]:
        p = os.path.join(DATA_DIR, f_name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                for item in d.get("Fresh_Triggers", []):
                    sym = str(item.get("symbol", "")).strip().upper()
                    if sym:
                        ground_truth_positives.add(sym)
                for item in d.get("Watchlist", []):
                    sym = str(item.get("symbol", "")).strip().upper()
                    if sym and sym not in ground_truth_positives:
                        ground_truth_negatives.add(sym)
            except Exception as e:
                print(f"⚠️ Error reading {f_name}: {e}")

    # 2. Scanner Signals Recorded Today
    v1_alerts = set()
    if os.path.exists(V1_LOG):
        try:
            with open(V1_LOG, "r", encoding="utf-8") as fp:
                d1 = json.load(fp)
            if d1.get("Date") == today_str:
                v1_alerts = set(d1.get("Alert1_Sent", []) + [x["symbol"] for x in d1.get("Breakouts_All_Day", []) if isinstance(x, dict)])
        except Exception:
            pass

    v2_alerts = set()
    if os.path.exists(V2_LOG):
        try:
            with open(V2_LOG, "r", encoding="utf-8") as fp:
                d2 = json.load(fp)
            if d2.get("Date") == today_str:
                v2_alerts = set([x["symbol"] for x in d2.get("Breakouts_All_Day", []) if isinstance(x, dict)])
        except Exception:
            pass

    def evaluate_engine(alerts):
        tp = len(alerts.intersection(ground_truth_positives))
        fn = len(ground_truth_positives - alerts)
        fp = len(alerts.intersection(ground_truth_negatives))
        tn = len(ground_truth_negatives - alerts)
        tpr = round((tp / (tp + fn) * 100.0), 1) if (tp + fn) > 0 else 100.0
        fnr = round((fn / (tp + fn) * 100.0), 1) if (tp + fn) > 0 else 0.0
        fpr = round((fp / (fp + tn) * 100.0), 1) if (fp + tn) > 0 else 0.0
        tnr = round((tn / (fp + tn) * 100.0), 1) if (fp + tn) > 0 else 100.0
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "tpr": tpr, "fnr": fnr, "fpr": fpr, "tnr": tnr}

    v1_stats = evaluate_engine(v1_alerts)
    v2_stats = evaluate_engine(v2_alerts)

    today_record = {
        "date": today_str,
        "ground_truth_count": len(ground_truth_positives),
        "ground_truth_symbols": list(ground_truth_positives),
        "v1": v1_stats,
        "v2": v2_stats
    }

    # 3. Load Existing Audit History
    history = []
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as fp:
                old_data = json.load(fp)
                history = old_data.get("Daily_History", [])
        except Exception:
            history = []

    # Replace entry if today was already audited, else append
    history = [rec for rec in history if rec.get("date") != today_str]
    history.append(today_record)
    history.sort(key=lambda x: x["date"])

    # 4. Compute Cumulative Totals across all archived days
    def aggregate_engine(engine_key):
        tot_tp = sum(r[engine_key]["tp"] for r in history)
        tot_fn = sum(r[engine_key]["fn"] for r in history)
        tot_fp = sum(r[engine_key]["fp"] for r in history)
        tot_tn = sum(r[engine_key]["tn"] for r in history)
        tot_tpr = round((tot_tp / (tot_tp + tot_fn) * 100.0), 1) if (tot_tp + tot_fn) > 0 else 100.0
        tot_fnr = round((tot_fn / (tot_tp + tot_fn) * 100.0), 1) if (tot_tp + tot_fn) > 0 else 0.0
        tot_fpr = round((tot_fp / (tot_fp + tot_tn) * 100.0), 1) if (tot_fp + tot_tn) > 0 else 0.0
        tot_tnr = round((tot_tn / (tot_fp + tot_tn) * 100.0), 1) if (tot_fp + tot_tn) > 0 else 100.0
        return {
            "Total_TP": tot_tp, "Total_FN": tot_fn, "Total_FP": tot_fp, "Total_TN": tot_tn,
            "Cumulative_TPR": f"{tot_tpr}%", "Cumulative_FNR": f"{tot_fnr}%",
            "Cumulative_FPR": f"{tot_fpr}%", "Cumulative_TNR": f"{tot_tnr}%"
        }

    cum_v1 = aggregate_engine("v1")
    cum_v2 = aggregate_engine("v2")

    today_summary = [
        {
            "Engine": "Scanner V1 (Live Alerts)",
            "TP (Caught)": v1_stats["tp"], "FN (Missed)": v1_stats["fn"],
            "FP (Fakeouts)": v1_stats["fp"], "TN (Quiet)": v1_stats["tn"],
            "TPR (Sensitivity)": f"{v1_stats['tpr']}%", "FNR (Miss Rate)": f"{v1_stats['fnr']}%",
            "FPR (Fallout)": f"{v1_stats['fpr']}%", "TNR (Specificity)": f"{v1_stats['tnr']}%"
        },
        {
            "Engine": "Scanner V2 (Optimized Silent)",
            "TP (Caught)": v2_stats["tp"], "FN (Missed)": v2_stats["fn"],
            "FP (Fakeouts)": v2_stats["fp"], "TN (Quiet)": v2_stats["tn"],
            "TPR (Sensitivity)": f"{v2_stats['tpr']}%", "FNR (Miss Rate)": f"{v2_stats['fnr']}%",
            "FPR (Fallout)": f"{v2_stats['fpr']}%", "TNR (Specificity)": f"{v2_stats['tnr']}%"
        }
    ]

    payload = {
        "Last_Updated": now_ts,
        "Total_Sessions_Audited": len(history),
        "Latest_Session_Date": today_str,
        "Latest_Ground_Truth": list(ground_truth_positives),
        "Latest_Day_Matrix": today_summary,
        "Cumulative_Metrics": {
            "Scanner_V1": cum_v1,
            "Scanner_V2": cum_v2
        },
        "Daily_History": history
    }

    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ Audit Log saved with {len(history)} total sessions compiled.")

if __name__ == "__main__":
    run_comprehensive_audit()
