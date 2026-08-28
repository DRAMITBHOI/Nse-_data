import os
import json
import numpy as np

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.json")

def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        return float(str(val).replace(",", "").replace("%", ""))
    except Exception:
        return default

def run_realistic_backtest():
    print("🚀 Running Wyckoff OBV Backtest with Breakout Confirmation...")
    if not os.path.exists(DATA_DIR):
        print(f"❌ '{DATA_DIR}' directory not found.")
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
                raw = json.load(fp)
        except Exception:
            continue

        if not isinstance(raw, list) or len(raw) < 80:
            continue

        # Sort and deduplicate by date
        d_map = {}
        for r in raw:
            if isinstance(r, dict) and "time" in r:
                t = str(r["time"]).split(" ")[0].split("T")[0]
                d_map[t] = r
        clean = [d_map[k] for k in sorted(d_map.keys())]

        if len(clean) < 60:
            continue

        # Compute True Delivery OBV with safe fallbacks
        cur_obv = 0
        parsed_data = []
        for i, r in enumerate(clean):
            c = safe_float(r.get("close"))
            h = safe_float(r.get("high"), c)
            l = safe_float(r.get("low"), c)
            o = safe_float(r.get("open"), c)
            dv = safe_float(r.get("delivery_vol"))
            tv = safe_float(r.get("volume"), dv)
            
            if tv > 0 and dv > tv:
                dv = tv
                
            if i > 0:
                pc = parsed_data[i - 1]["close"]
                if c > pc:
                    cur_obv += dv
                elif c < pc:
                    cur_obv -= dv
            else:
                cur_obv = 0

            parsed_data.append({
                "open": o, "high": h, "low": l, "close": c,
                "obv": cur_obv
            })

        closes = [x["close"] for x in parsed_data]
        highs = [x["high"] for x in parsed_data]
        lows = [x["low"] for x in parsed_data]
        obvs = [x["obv"] for x in parsed_data]

        n_bars = len(parsed_data)
        idx = 50
        while idx < n_bars - 45:
            signal_found = False
            for label, (min_d, max_d) in spans.items():
                for s in range(min_d, max_d + 1):
                    if idx - s < 0:
                        continue
                        
                    p_start = closes[idx - s]
                    if p_start <= 0:
                        continue
                        
                    p_low = min(closes[idx - s : idx + 1])
                    p_drop = ((p_low - p_start) / p_start) * 100

                    o_start = obvs[idx - s]
                    o_end = obvs[idx]
                    o_gain = ((o_end - o_start) / abs(o_start)) * 100 if abs(o_start) > 0 else 0

                    # 1. Base Setup Criteria
                    if p_drop <= -7.5 and o_gain >= 8.0:
                        swing_high = max(highs[idx - s : idx + 1])
                        stop_loss = min(lows[idx - s : idx + 1])

                        # 2. Wait up to 15 trading days for confirmed breakout above Swing High
                        breakout_idx = None
                        for fwd in range(idx + 1, min(idx + 16, n_bars - 30)):
                            if closes[fwd] > swing_high:
                                breakout_idx = fwd
                                break
                            if lows[fwd] < stop_loss:
                                break  # Invalidated before breakout

                        if breakout_idx is not None:
                            entry_price = closes[breakout_idx]
                            risk = entry_price - stop_loss
                            if risk <= 0:
                                continue
                                
                            target_price = entry_price + (2.0 * risk)

                            eval_end = min(breakout_idx + 45, n_bars)
                            fwd_highs = highs[breakout_idx + 1 : eval_end]
                            fwd_lows = lows[breakout_idx + 1 : eval_end]
                            fwd_closes = closes[breakout_idx + 1 : eval_end]

                            if not fwd_closes:
                                continue

                            hit_target = False
                            hit_stop = False
                            for h_val, l_val in zip(fwd_highs, fwd_lows):
                                if h_val >= target_price:
                                    hit_target = True
                                    break
                                if l_val <= stop_loss:
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
                            idx = breakout_idx + 15  # Cooldown
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
        else:
            summary.append({
                "horizon": label,
                "signals": 0,
                "win_rate": 0.0,
                "avg_return": 0.0
            })

    with open(REPORT_FILE, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"✅ Clean backtest complete! Results saved to {REPORT_FILE}")

if __name__ == "__main__":
    run_realistic_backtest()
