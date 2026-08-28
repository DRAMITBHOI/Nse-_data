import os
import json
import numpy as np

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.json")

def run_realistic_backtest():
    print("🚀 Running Wyckoff OBV Backtest with Breakout Confirmation...")
    if not os.path.exists(DATA_DIR):
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
        highs = [float(x["high"]) for x in clean]
        lows = [float(x["low"]) for x in clean]
        obvs = [float(x["deliv_obv"]) for x in clean]

        idx = 60
        while idx < len(clean) - 60:
            signal_found = False
            for label, (min_d, max_d) in spans.items():
                for s in range(min_d, max_d + 1):
                    p_start = closes[idx - s]
                    p_low = min(closes[idx - s : idx + 1])
                    p_drop = ((p_low - p_start) / p_start) * 100

                    o_start = obvs[idx - s]
                    o_end = obvs[idx]
                    o_gain = ((o_end - o_start) / abs(o_start)) * 100 if abs(o_start) > 0 else 0

                    # 1. Base Criteria
                    if p_drop <= -7.5 and o_gain >= 8.0:
                        swing_high = max(highs[idx - s : idx + 1])
                        stop_loss = min(lows[idx - s : idx + 1])

                        # 2. Look forward up to 15 days for a CONFIRMED BREAKOUT above Swing High
                        breakout_idx = None
                        for fwd in range(idx + 1, min(idx + 16, len(clean) - 45)):
                            if closes[fwd] > swing_high:
                                breakout_idx = fwd
                                break
                            if lows[fwd] < stop_loss:
                                break  # Invalidation before breakout

                        if breakout_idx is not None:
                            entry_price = closes[breakout_idx]
                            risk = entry_price - stop_loss
                            if risk <= 0: continue
                            
                            target_price = entry_price + (2.0 * risk) # 2:1 Reward-to-Risk Target

                            eval_window = clean[breakout_idx + 1 : breakout_idx + 61]
                            fwd_highs = [float(x["high"]) for x in eval_window]
                            fwd_lows = [float(x["low"]) for x in eval_window]
                            fwd_closes = [float(x["close"]) for x in eval_window]

                            if not fwd_closes: continue

                            hit_target = False
                            hit_stop = False
                            for h, l in zip(fwd_highs, fwd_lows):
                                if h >= target_price:
                                    hit_target = True
                                    break
                                if l <= stop_loss:
                                    hit_stop = True
                                    break

                            results[label]["signals"] += 1
                            f_ret = ((fwd_closes[-1] - entry_price) / entry_price) * 100
                            results[label]["returns"].append(f_ret)

                            if hit_target and not hit_stop:
                                results[label]["wins"] += 1
                            else:
                                results[label]["losses"] += 1

                            signal_found = True
                            idx = breakout_idx + 20  # Cooldown skip
                            break
                if signal_found:
                    break
            idx += 1

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
    print(f"✅ Clean backtest complete. Summary saved to {REPORT_FILE}")

if __name__ == "__main__":
    run_realistic_backtest()
