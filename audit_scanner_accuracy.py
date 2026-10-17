import os
import sys
import json
import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_V1 = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
LIVE_BREAKOUTS_V2 = os.path.join(DATA_DIR, "live_intraday_breakouts_v2.json")
ULTRA_750_FILE = os.path.join(DATA_DIR, "swing_ultra750_results.json")
ULTRA_N750_FILE = os.path.join(DATA_DIR, "swing_ultraN750_results.json")
AUDIT_LOG_FILE = os.path.join(DATA_DIR, "accuracy_audit_log.json")

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def get_ist_now():
    return datetime.datetime.now(IST)

def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def extract_breakout_symbols(source_dict):
    signals = set()
    for item in source_dict.get("Breakouts_All_Day", []):
        sym = str(item.get("symbol", "")).strip().upper()
        act = str(item.get("action", "")).strip().upper()
        if sym and ("BREAKOUT" in act or item.get("clearance_%", 0) >= 0.25):
            signals.add(sym)
    return signals

def run_accuracy_audit():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")

    v1_data = load_json(LIVE_BREAKOUTS_V1)
    v2_data = load_json(LIVE_BREAKOUTS_V2)
    u750 = load_json(ULTRA_750_FILE)
    un750 = load_json(ULTRA_N750_FILE)

    # Collect EOD Swing Ultra Final Confirmed Triggers
    eod_verified = set()
    for src in [u750, un750]:
        for key in ["Fresh_Triggers", "Breakout_Triggers"]:
            items = src.get(key, [])
            if isinstance(items, list):
                for it in items:
                    sym = it.get("symbol", "") if isinstance(it, dict) else str(it)
                    if sym.strip():
                        eod_verified.add(sym.strip().upper())

    # Evaluate Scanner 1
    v1_sigs = extract_breakout_symbols(v1_data)
    v1_hits = sorted(list(v1_sigs.intersection(eod_verified)))
    v1_acc = round((len(v1_hits) / len(v1_sigs) * 100.0), 2) if v1_sigs else 100.0

    # Evaluate Scanner 2
    v2_sigs = extract_breakout_symbols(v2_data)
    v2_hits = sorted(list(v2_sigs.intersection(eod_verified)))
    v2_acc = round((len(v2_hits) / len(v2_sigs) * 100.0), 2) if v2_sigs else 100.0

    audit_history = load_json(AUDIT_LOG_FILE)
    records = [r for r in audit_history.get("Daily_Records", []) if r.get("date") != today_date]

    daily_entry = {
        "date": today_date,
        "eod_verified_count": len(eod_verified),
        "scanner_v1": {
            "signals": len(v1_sigs),
            "hits": len(v1_hits),
            "accuracy_%": v1_acc,
            "tickers": sorted(list(v1_sigs))
        },
        "scanner_v2": {
            "signals": len(v2_sigs),
            "hits": len(v2_hits),
            "accuracy_%": v2_acc,
            "tickers": sorted(list(v2_sigs))
        },
        "eod_verified": sorted(list(eod_verified))
    }
    records.append(daily_entry)

    # Cumulative Head-to-Head Totals
    v1_tot_sigs = sum(r.get("scanner_v1", {}).get("signals", 0) for r in records)
    v1_tot_hits = sum(r.get("scanner_v1", {}).get("hits", 0) for r in records)
    v1_cum_acc = round((v1_tot_hits / v1_tot_sigs * 100.0), 2) if v1_tot_sigs > 0 else 100.0

    v2_tot_sigs = sum(r.get("scanner_v2", {}).get("signals", 0) for r in records)
    v2_tot_hits = sum(r.get("scanner_v2", {}).get("hits", 0) for r in records)
    v2_cum_acc = round((v2_tot_hits / v2_tot_sigs * 100.0), 2) if v2_tot_sigs > 0 else 100.0

    output_payload = {
        "Last_Updated": now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST"),
        "Sessions_Audited": len(records),
        "Cumulative_Accuracy": {
            "Scanner_V1_%": v1_cum_acc,
            "Scanner_V2_%": v2_cum_acc
        },
        "Daily_Records": records
    }

    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"✅ Head-to-Head Audit Logged (Silent): V1={v1_acc}% (Cum: {v1_cum_acc}%) vs V2={v2_acc}% (Cum: {v2_cum_acc}%)")

if __name__ == "__main__":
    run_accuracy_audit()
