import os
import json
import itertools
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover

def run_parameter_optimizer():
    print("🚀 Starting OBV Parameter Matrix Grid Search across Nifty 750 dataset...")

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as f:
                fundamentals = json.load(f)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", 
            "backtest_report.json", "active_trade_plan.json",
            "optimal_grid_matrix.json", "optimal_trade_plan.json"
        ]
    ]

    target_stocks = [f for f in stock_files if fundamentals and f.replace(".json", "") in fundamentals] or stock_files
    print(f"📊 Pre-loading {len(target_stocks)} institutional stock histories into memory...")

    loaded_data = {}
    for f_name in target_stocks:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)
        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
            if len(raw) >= 60:
                closes = np.array([float(x["close"]) for x in raw])
                highs = np.array([float(x.get("high", x["close"])) for x in raw])
                lows = np.array([float(x.get("low", x["close"])) for x in raw])
                vols = np.array([float(x.get("delivery_vol", 0)) for x in raw])
                N = len(closes)

                obvs = np.zeros(N)
                cur_obv = 0
                for idx in range(N):
                    if idx > 0:
                        if closes[idx] > closes[idx - 1]:
                            cur_obv += vols[idx]
                        elif closes[idx] < closes[idx - 1]:
                            cur_obv -= vols[idx]
                    else:
                        cur_obv = vols[idx]
                    obvs[idx] = cur_obv

                loaded_data[sym] = {
                    "closes": closes,
                    "highs": highs,
                    "lows": lows,
                    "vols": vols,
                    "obvs": obvs,
                    "turnovers": (closes * vols) / 1e7,
                    "N": N
                }
        except Exception:
            continue

    # Grid Search Space
    price_drops = [-3.0, -5.0, -7.5]
    obv_gains = [3.0, 5.0, 8.0, 12.0]
    exit_obv_drops = [-3.0, -5.0, -8.0]
    lookbacks = [5, 10, 15, 20]  # 1W to 4W

    all_grid_results = []
    param_combinations = list(itertools.product(price_drops, obv_gains, exit_obv_drops))
    print(f"🧪 Testing {len(param_combinations)} unique divergence permutations across historical data...")

    for p_drop, o_gain, exit_o_drop in param_combinations:
        trades = []

        for sym, s_data in loaded_data.items():
            closes = s_data["closes"]
            highs = s_data["highs"]
            lows = s_data["lows"]
            vols = s_data["vols"]
            obvs = s_data["obvs"]
            turnovers = s_data["turnovers"]
            N = s_data["N"]

            i = 25
            while i < N - 15:
                if closes[i] < 30.0 or np.mean(turnovers[max(0, i - 8):i + 1]) < MIN_DELIVERY_TURNOVER_CR:
                    i += 1
                    continue

                matched_lb = None
                for lb in lookbacks:
                    p_chg = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                    past_o = obvs[i - lb]
                    o_chg = ((obvs[i] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                    if p_chg <= p_drop and o_chg >= o_gain:
                        matched_lb = lb
                        break

                if not matched_lb:
                    i += 1
                    continue

                base_h = np.max(highs[i - matched_lb:i + 1])
                base_l = np.min(lows[i - matched_lb:i + 1])
                sl = round(base_l * 0.995, 2)

                trade_entered = False
                for k in range(i + 1, min(i + 16, N)):
                    if lows[k] < base_l * 0.97:
                        break

                    avg_v_10 = np.mean(vols[max(0, k - 10):k])
                    if closes[k] > base_h and vols[k] >= avg_v_10:
                        trade_entered = True
                        entry_p = closes[k]
                        entry_idx = k
                        exit_p = entry_p
                        exit_idx = entry_idx

                        for m in range(entry_idx + 1, min(entry_idx + 45, N)):
                            if lows[m] <= sl:
                                exit_p = sl
                                exit_idx = m
                                break

                            bars_held = m - entry_idx
                            if bars_held >= 5:
                                span = min(bars_held, 10)
                                p_post = ((closes[m] - closes[m - span]) / closes[m - span]) * 100
                                past_obv_m = obvs[m - span]
                                obv_post = ((obvs[m] - past_obv_m) / abs(past_obv_m)) * 100 if abs(past_obv_m) > 0 else 0

                                if (p_post >= -1.5 or closes[m] >= entry_p * 1.05) and obv_post <= exit_o_drop:
                                    exit_p = closes[m]
                                    exit_idx = m
                                    break

                            exit_p = closes[m]
                            exit_idx = m

                        pnl = ((exit_p - entry_p) / entry_p) * 100
                        trades.append(pnl)
                        i = exit_idx + 1
                        break

                if not trade_entered:
                    i += 1

        if len(trades) >= 30:
            arr = np.array(trades)
            wins = arr[arr > 0]
            losses = arr[arr <= 0]
            wr = (len(wins) / len(arr)) * 100
            pf = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999.0

            all_grid_results.append({
                "entry_drop": p_drop,
                "entry_obv": o_gain,
                "exit_obv": exit_o_drop,
                "trades": len(arr),
                "win_rate": round(wr, 1),
                "profit_factor": round(pf, 2),
                "avg_win": round(float(np.mean(wins)), 2) if len(wins) > 0 else 0.0,
                "avg_loss": round(float(np.mean(losses)), 2) if len(losses) > 0 else 0.0
            })

    all_grid_results.sort(key=lambda x: (x["profit_factor"], x["win_rate"]), reverse=True)
    best = all_grid_results[0]
    top_5 = all_grid_results[:5]

    print(f"\n🏆 Optimal Model: Entry (P <= {best['entry_drop']}%, OBV >= +{best['entry_obv']}%) | Exit (OBV <= {best['exit_obv']}%)")

    # Generate Active Trade Setups with Optimal Settings
    active_setups = []
    opt_p_drop = best["entry_drop"]
    opt_obv_gain = best["entry_obv"]

    for sym, s_data in loaded_data.items():
        closes = s_data["closes"]
        highs = s_data["highs"]
        lows = s_data["lows"]
        vols = s_data["vols"]
        obvs = s_data["obvs"]
        turnovers = s_data["turnovers"]
        N = s_data["N"]

        if N >= 30 and closes[-1] >= 30.0:
            sma_9_to = np.mean(turnovers[-9:])
            if sma_9_to >= MIN_DELIVERY_TURNOVER_CR:
                matched_sym = False
                for lb in lookbacks:
                    for offset in range(0, 10):
                        curr_pos = (N - 1) - offset
                        base_start = curr_pos - lb
                        if base_start < 0:
                            continue

                        p_drop = ((closes[curr_pos] - closes[base_start]) / closes[base_start]) * 100
                        past_o = obvs[base_start]
                        o_gain = ((obvs[curr_pos] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                        if p_drop <= opt_p_drop and o_gain >= opt_obv_gain:
                            base_h = np.max(highs[base_start:curr_pos + 1])
                            base_l = np.min(lows[base_start:curr_pos + 1])
                            sl = round(base_l * 0.995, 2)
                            risk = round(((closes[-1] - sl) / closes[-1]) * 100, 2)

                            avg_v = np.mean(vols[max(0, N - 11):N - 1])
                            is_triggered = (closes[-1] >= base_h) and (vols[-1] >= avg_v)

                            active_setups.append({
                                "Symbol": sym,
                                "Signal": "🟢 CHoCH TRIGGERED (Buy)" if is_triggered else "🟡 ACCUMULATING (Awaiting CHoCH)",
                                "LTP (₹)": round(closes[-1], 2),
                                "CHoCH Level": f"> ₹{round(base_h, 2)}",
                                "Stop Loss (₹)": sl,
                                "Risk %": f"{risk}%",
                                "Base Window": f"{lb//5}W Base",
                                "9D Turnover": f"₹{sma_9_to:.1f} Cr/d"
                            })
                            matched_sym = True
                            break
                    if matched_sym:
                        break

    active_setups.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk %"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "optimal_grid_matrix.json"), "w") as f:
        json.dump({"optimal": best, "top_matrix": top_5}, f, indent=2)

    with open(os.path.join(DATA_DIR, "optimal_trade_plan.json"), "w") as f:
        json.dump(active_setups, f, indent=2)

    print(f"🎉 Optimization complete! Saved outputs to data/optimal_grid_matrix.json and data/optimal_trade_plan.json")

if __name__ == "__main__":
    run_parameter_optimizer()
