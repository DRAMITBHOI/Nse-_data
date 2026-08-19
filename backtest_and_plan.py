import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 2.0  # Min ₹2 Cr/day delivery turnover (High liquidity)
TARGET_PROFIT_PCT = 20.0        # Target: +20%
MAX_HOLDING_DAYS = 50          # Max holding: ~10 weeks

def run_quant_engine():
    print("🚀 Running Institutional True Delivery OBV + Wyckoff Phase Backtest...")

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
        if f.endswith(".json") and f not in ["fundamentals.json", "screener_results.json", "backtest_report.json", "active_trade_plan.json"]
    ]
    
    target_stocks = [f for f in stock_files if fundamentals and f.replace(".json", "") in fundamentals] or stock_files
    print(f"📊 Analyzing {len(target_stocks)} institutional stocks...")

    loaded_data = {}
    for f_name in target_stocks:
        sym = f_name.replace(".json", "")
        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                d = json.load(f)
            if len(d) >= 80:
                closes = np.array([float(x["close"]) for x in d])
                highs = np.array([float(x.get("high", x["close"])) for x in d])
                lows = np.array([float(x.get("low", x["close"])) for x in d])
                vols = np.array([float(x.get("delivery_vol", 0)) for x in d])
                
                # True Demat Delivery OBV
                obv_arr = np.zeros(len(closes))
                cur_obv = 0
                for idx in range(len(closes)):
                    if idx > 0:
                        if closes[idx] > closes[idx-1]:
                            cur_obv += vols[idx]
                        elif closes[idx] < closes[idx-1]:
                            cur_obv -= vols[idx]
                    else:
                        cur_obv = vols[idx]
                    obv_arr[idx] = cur_obv

                loaded_data[sym] = {
                    "dates": [x.get("date", "") for x in d],
                    "closes": closes,
                    "highs": highs,
                    "lows": lows,
                    "vols": vols,
                    "obvs": obv_arr,
                    "turnovers": (closes * vols) / 1e7
                }
        except Exception:
            continue

    # 4 Robust Institutional Breakout Models
    models = [
        {"name": "Wyckoff Accumulation + 20SMA Reclaim", "lb": 20, "p_drop": -6.0, "obv_gain": 8.0, "sma_reclaim": True, "vol_mult": 1.5},
        {"name": "Base Absorption + 10D High Breakout", "lb": 25, "p_drop": -8.0, "obv_gain": 10.0, "sma_reclaim": True, "vol_mult": 1.2},
        {"name": "Aggressive Spring Reversal", "lb": 15, "p_drop": -5.0, "obv_gain": 6.0, "sma_reclaim": False, "vol_mult": 1.8},
        {"name": "Multi-Month Smart Money Base", "lb": 35, "p_drop": -10.0, "obv_gain": 12.0, "sma_reclaim": True, "vol_mult": 1.3}
    ]

    backtest_stats = []

    for m in models:
        lb = m["lb"]
        p_drop = m["p_drop"]
        obv_gain = m["obv_gain"]
        req_sma = m["sma_reclaim"]
        vol_mult = m["vol_mult"]

        trades = []

        for sym, s_data in loaded_data.items():
            closes = s_data["closes"]
            highs = s_data["highs"]
            lows = s_data["lows"]
            vols = s_data["vols"]
            obvs = s_data["obvs"]
            turnovers = s_data["turnovers"]

            i = lb + 20
            while i < len(closes) - 5:
                # Turnover gate & minimum price
                if closes[i] < 40.0 or np.mean(turnovers[max(0, i-8):i+1]) < MIN_DELIVERY_TURNOVER_CR:
                    i += 1
                    continue

                # Divergence check
                p_chg = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                obv_chg = ((obvs[i] - obvs[i - lb]) / abs(obvs[i - lb])) * 100 if abs(obvs[i - lb]) > 0 else 0

                if p_chg <= p_drop and obv_chg >= obv_gain:
                    # 1. 20-Day SMA Trend Reclaim Gate
                    sma_20 = np.mean(closes[i-19:i+1])
                    if req_sma and closes[i] < sma_20:
                        i += 1
                        continue

                    # 2. Institutional Volume Spike Confirmation
                    avg_vol_20 = np.mean(vols[max(0, i-19):i])
                    if vols[i] < avg_vol_20 * vol_mult:
                        i += 1
                        continue

                    # 3. Breakout above 10-day consolidation high
                    recent_10d_high = np.max(highs[i-10:i])
                    if closes[i] >= recent_10d_high:
                        entry_price = closes[i]
                        base_low = np.min(lows[i-lb:i+1])
                        stop_loss = round(base_low * 0.99, 2)
                        target_price = round(entry_price * (1 + TARGET_PROFIT_PCT / 100), 2)
                        risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                        if risk_pct <= 7.0:  # Max 7% risk allowance
                            outcome = "TIME_EXIT"
                            exit_price = entry_price
                            exit_idx = i

                            for j in range(i + 1, min(i + MAX_HOLDING_DAYS + 1, len(closes))):
                                if highs[j] >= target_price:
                                    outcome = "WIN (+20%)"
                                    exit_price = target_price
                                    exit_idx = j
                                    break
                                if lows[j] <= stop_loss:
                                    outcome = "STOP_LOSS"
                                    exit_price = stop_loss
                                    exit_idx = j
                                    break
                                exit_price = closes[j]
                                exit_idx = j

                            pnl = ((exit_price - entry_price) / entry_price) * 100
                            trades.append({
                                "pnl": pnl,
                                "outcome": outcome,
                                "holding": exit_idx - i
                            })
                            i = exit_idx + 1
                            continue
                i += 1

        if trades:
            df_t = pd.DataFrame(trades)
            wins = df_t[df_t["outcome"] == "WIN (+20%)"]
            losses = df_t[df_t["outcome"] == "STOP_LOSS"]
            win_rate = (len(wins) / len(df_t)) * 100
            pf = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) > 0 and losses["pnl"].sum() != 0 else 999.0

            backtest_stats.append({
                "model_name": m["name"],
                "params": m,
                "total_trades": len(df_t),
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": round(pf, 2),
                "avg_gain_pct": round(df_t["pnl"].mean(), 2),
                "avg_holding_days": round(df_t["holding"].mean(), 1)
            })

    best_model = max(backtest_stats, key=lambda x: (x["win_rate_pct"], x["profit_factor"]))
    print(f"\n🏆 Winning Model: {best_model['model_name']} -> {best_model['win_rate_pct']}% Win Rate (PF: {best_model['profit_factor']})")

    # Generate Active Trade Plan Using the Best Model
    active_plan = []
    opt = best_model["params"]

    for sym, s_data in loaded_data.items():
        closes = s_data["closes"]
        highs = s_data["highs"]
        lows = s_data["lows"]
        vols = s_data["vols"]
        obvs = s_data["obvs"]
        turnovers = s_data["turnovers"]

        if len(closes) < opt["lb"] + 20 or closes[-1] < 40.0:
            continue

        sma_9_to = np.mean(turnovers[-9:])
        if sma_9_to < MIN_DELIVERY_TURNOVER_CR:
            continue

        curr_c = closes[-1]
        past_c = closes[-opt["lb"] - 1]
        curr_obv = obvs[-1]
        past_obv = obvs[-opt["lb"] - 1]

        p_chg = ((curr_c - past_c) / past_c) * 100
        obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0

        # Divergence test
        if p_chg <= opt["p_drop"] and obv_chg >= opt["obv_gain"]:
            sma_20 = np.mean(closes[-20:])
            avg_vol_20 = np.mean(vols[-20:])
            recent_10d_high = np.max(highs[-11:-1])

            is_above_sma = curr_c >= sma_20
            is_vol_spike = vols[-1] >= avg_vol_20 * opt["vol_mult"]
            is_breakout = curr_c >= recent_10d_high

            is_triggered = is_above_sma and is_vol_spike and is_breakout

            base_low = np.min(lows[-opt["lb"]:])
            sl_price = round(base_low * 0.99, 2)
            risk_pct = round(((curr_c - sl_price) / curr_c) * 100, 2)

            if risk_pct <= 7.0:
                meta = fundamentals.get(sym, {})
                active_plan.append({
                    "Symbol": sym,
                    "Signal": "🟢 BUY TRIGGERED" if is_triggered else "🟡 PENDING BREAKOUT",
                    "Entry Trigger": f"₹{curr_c:.2f}" if is_triggered else f"Breakout > ₹{recent_10d_high:.2f}",
                    "Stop Loss (₹)": sl_price,
                    "Risk": f"{risk_pct}%",
                    "Target (+20%)": round(curr_c * 1.20, 2),
                    "9D Turnover": f"₹{sma_9_to:.1f} Cr/d",
                    "Category": meta.get("category", "Nifty 750")
                })

    active_plan.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump({"summary": backtest_stats, "winning_model": best_model}, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_plan, f, indent=2)

    print(f"🎉 Generated {len(active_plan)} disciplined setups.")

if __name__ == "__main__":
    run_quant_engine()
