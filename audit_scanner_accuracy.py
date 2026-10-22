from datetime import datetime
import glob
import json
import os
import pandas as pd

DATA_DIR = "data"
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "accuracy_audit_log.json")
V1_LOG = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
V2_LOG = os.path.join(DATA_DIR, "live_intraday_breakouts_v2.json")


def run_comprehensive_audit():
  print("=" * 100)
  print(
      "🔍 COMPREHENSIVE SCANNER ACCURACY AUDIT (CONFUSION MATRIX: TPR, FNR, FPR,"
      " TNR)"
  )
  print("=" * 100)

  # 1. Collect EOD Ground Truth from Swing Ultra production files
  ground_truth_positives = set()
  ground_truth_negatives = set()

  for f_name in [
      "swing_ultra750_results.json",
      "swing_ultraN750_results.json",
  ]:
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

  print(
      f"📊 Ground Truth Today: {len(ground_truth_positives)} Verified Breakouts"
      f" {list(ground_truth_positives)}, {len(ground_truth_negatives)} Coiling"
      " Negatives."
  )

  # 2. Audit V1 Scanner
  v1_alerts = set()
  if os.path.exists(V1_LOG):
    try:
      with open(V1_LOG, "r", encoding="utf-8") as fp:
        d1 = json.load(fp)
      v1_alerts = set(
          d1.get("Alert1_Sent", [])
          + [
              x["symbol"]
              for x in d1.get("Breakouts_All_Day", [])
              if isinstance(x, dict)
          ]
      )
    except Exception:
      pass

  # 3. Audit V2 Scanner
  v2_alerts = set()
  if os.path.exists(V2_LOG):
    try:
      with open(V2_LOG, "r", encoding="utf-8") as fp:
        d2 = json.load(fp)
      v2_alerts = set([
          x["symbol"]
          for x in d2.get("Breakouts_All_Day", [])
          if isinstance(x, dict)
      ])
    except Exception:
      pass

  def calculate_rates(alerts, name):
    tp = len(alerts.intersection(ground_truth_positives))
    fn = len(ground_truth_positives - alerts)
    fp = len(alerts.intersection(ground_truth_negatives))
    tn = len(ground_truth_negatives - alerts)

    tpr = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
    fnr = (fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    tnr = (tn / (fp + tn) * 100.0) if (fp + tn) > 0 else 100.0

    return {
        "Engine": name,
        "TP (Caught)": tp,
        "FN (Missed)": fn,
        "FP (Fakeouts)": fp,
        "TN (Quiet)": tn,
        "True Positive Rate (Sensitivity)": f"{tpr:.1f}%",
        "False Negative Rate (Miss Rate)": f"{fnr:.1f}%",
        "False Positive Rate (Fallout)": f"{fpr:.1f}%",
        "True Negative Rate (Specificity)": f"{tnr:.1f}%",
    }

  matrix = [
      calculate_rates(v1_alerts, "Scanner V1 (Live Alerts Engine)"),
      calculate_rates(v2_alerts, "Scanner V2 (Optimized Silent Engine)"),
  ]

  df_matrix = pd.DataFrame(matrix)
  print("\n" + df_matrix.to_string(index=False) + "\n")

  # Persist to disk
  out_audit = {
      "Date": datetime.now().strftime("%Y-%m-%d"),
      "Audit_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
      "Ground_Truth_Positives": list(ground_truth_positives),
      "Audit_Summary": matrix,
  }
  with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as fp:
    json.dump(out_audit, fp, indent=2)


if __name__ == "__main__":
  run_comprehensive_audit()
