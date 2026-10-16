import os
import sys
import json
import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LIVE_BREAKOUTS_FILE = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
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
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
    return {}

def run_accuracy_audit():
    now_dt = get_ist_now()
    today_date = now_dt.strftime("%Y-%m-%d")

    live_data = load_json(LIVE_BREAKOUTS_FILE)
    u750_data = load_json(ULTRA_750_FILE)
    un750_data = load_json(ULTRA_N750_FILE)

    # 1. Collect live breakouts captured at 3:15 PM window
    live_signals = set()
    breakouts_all_day = live_data.get("Breakouts_All_Day", [])
    for item in breakouts_all_day:
        sym = str(item.get("symbol", "")).strip().upper()
        act = str(item.get("action", "")).strip().upper()
        if sym and ("BREAKOUT" in act or item.get("clearance_%", 0) >= 0.25):
            live_signals.add(sym)

    # 2. Collect confirmed breakouts verified by EOD Swing Ultra (750 + Non-N750)
    eod_verified = set()
    for source in [u750_data, un750_data]:
        fresh = source.get("Fresh_Triggers", [])
        if isinstance(fresh, list):
            for item in fresh:
                if isinstance(item, dict):
                    sym = str(item.get("symbol", "")).strip().upper()
                    if sym:
                        eod_verified.add(sym)
                elif isinstance(item, str):
                    eod_verified.add(item.strip().upper())

        triggers = source.get("Breakout_Triggers", [])
        if isinstance(triggers, list):
            for item in triggers:
                if isinstance(item, dict):
                    sym = str(item.get("symbol", "")).strip().upper()
                    if sym:
                        eod_verified.add(sym)

    # 3. Calculate Precision & Mismatches
    hits = sorted(list(live_signals.intersection(eod_verified)))
    false_positives = sorted(list(live_signals.difference(eod_verified)))
    missed_breakouts = sorted(list(eod_verified.difference(live_signals)))

    total_live = len(live_signals)
    daily_accuracy = round((len(hits) / total_live * 100.0), 2) if total_live > 0 else 100.0

    # 4. Update Cumulative Historical Log
    audit_history = load_json(AUDIT_LOG_FILE)
    daily_records = audit_history.get("Daily_Records", [])
    daily_records = [r for r in daily_records if r.get("date") != today_date]

    daily_entry = {
        "date": today_date,
        "timestamp": now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST"),
        "live_signals_count": total_live,
        "eod_verified_count": len(eod_verified),
        "hits_count": len(hits),
        "false_positives_count": len(false_positives),
        "missed_count": len(missed_breakouts),
        "daily_accuracy_pct": daily_accuracy,
        "hits": hits,
        "false_positives": false_positives,
        "missed": missed_breakouts
    }
    daily_records.append(daily_entry)

    cum_live = sum(r.get("live_signals_count", 0) for r in daily_records)
    cum_hits = sum(r.get("hits_count", 0) for r in daily_records)
    cum_accuracy = round((cum_hits / cum_live * 100.0), 2) if cum_live > 0 else 100.0

    compiled_audit = {
        "Last_Updated": now_dt.strftime("%Y-%m-%d %I:%M:%S %p IST"),
        "Total_Days_Audited": len(daily_records),
        "Cumulative_Live_Signals": cum_live,
        "Cumulative_Hits": cum_hits,
        "Cumulative_Accuracy_Pct": cum_accuracy,
        "Daily_Records": daily_records
    }

    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(compiled_audit, f, indent=2)

    print(f"✅ Accuracy Audit Logged (Silent): Daily={daily_accuracy}% | Cumulative={cum_accuracy}%")

if __name__ == "__main__":
    run_accuracy_audit()
