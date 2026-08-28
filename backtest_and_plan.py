import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.json")

def run_obv_divergence_backtest():
    print("🚀 Running True Delivery OBV Horizon Backtest...")
    if not os.path.exists(DATA_DIR):
        print("❌ Data folder not found.")
        return

    files = [
        f for f in os.listdir(DATA_DIR) 
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "wyckoff_screener_results.json", "active_trade_plan.json", 
            "backtest_report.json"
        ]
    ]

    spans = {
        "1W-2W Base (5-10d)": (5, 10),
        "3W-6W Base (15-30d)": (15, 30),
        "7W-10W Base (35-50d)": (35, 50),
        ">10W Base (50-70d)": (50, 70)
    }

    results = {k: {"signals": 0, "wins": 0, "losses": 0, "returns": []} for k in spans}

    for f in files:
        path = os.path.join(DATA_DIR, f)
        try:
            with open(path, "r") as fp:
                data = json.load(fp)
        except Exception:
            continue

        if not isinstance(data, list) or len(data) < 100:
            continue

        # Sort and deduplicate
        d_map = {}
        for r in data:
            if isinstance(r, dict) and "time" in r:
                t = str(r["time"]).split(" ")[0].split("T")[0]
                d_map[t] = r
        clean = [d_map[k] for k in sorted(d_map.keys())]

        # Calculate True Delivery OBV
        cur_obv = 0
        for i, r in enumerate(clean):
            dv = float(r.get("delivery_vol", 0))
            tv = float(r.get("volume", dv))
            if tv > 0 and dv > tv: dv = tv
            if i > 0:
                pc, cc = float(clean[i-1]["close"]), float(r["close"])
                if cc > pc: cur_obv += dv
                elif cc < pc: cur_obv -= dv
            r["deliv_obv"] = cur_obv

        closes = [float(x["close"]) for x in clean]
        obvs = [float(x["deliv_obv"]) for x in clean]

        # Evaluate across rolling historical windows (with 60-day forward validation)
        for idx in range(60, len(clean) - 60, 5):
            for label, (min_d, max_d) in spans.items():
                for s in range(min_d, max_d + 1):
                    p_start = closes[idx - s]
                    p_low = min(closes[idx - s : idx + 1])
                    p_drop = ((p_low - p_start) / p_start) * 100

                    o_start = obvs[idx - s]
                    o_end = obvs[idx]
                    o_gain = ((o_end - o_start) / abs(o_start)) * 100 if abs(o_start) > 0 else 0

                    # Criteria: Price Drop >= 7.5%, OBV Gain >= 8.0%
                    if p_drop <= -7.5 and o_gain >= 8.0:
                        entry_price = closes[idx]
                        stop_loss = min([float(x["low"]) for x in clean[idx - s : idx + 1]])
                        target_price = entry_price * 1.20 # 20% Target

                        forward_candles = clean[idx + 1 : idx + 61]
                        forward_closes = [float(x["close"]) for x in forward_candles]
                        forward_lows = [float(x["low"]) for x in forward_candles]

                        hit_target = any(c >= target_price for c in forward_closes)
                        hit_stop = any(l <= stop_loss for l in forward_lows)

                        results[label]["signals"] += 1
                        f_ret = ((forward_closes[-1] - entry_price) / entry_price) * 100
                        results[label]["returns"].append(f_ret)

                        if hit_target and not hit_stop:
                            results[label]["wins"] += 1
                        else:
                            results[label]["losses"] += 1
                        break

    summary = []
    for label, stat in results.items():
        total = stat["signals"]
        if total > 0:
            win_rate = (stat["wins"] / total) * 100
            avg_ret = float(np.mean(stat["returns"]))
            summary.append({
                "horizon": label,
                "signals": total,
                "win_rate": round(win_rate, 1),
                "avg_return": round(avg_ret, 2)
            })

    with open(REPORT_FILE, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"✅ Backtest completed! Saved summary to {REPORT_FILE}")

if __name__ == "__main__":
    run_obv_divergence_backtest()
