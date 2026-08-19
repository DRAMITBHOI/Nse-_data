import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover
TARGET_PROFIT_PCT = 20.0        # Primary Target: +20% Gain
MAX_HOLDING_DAYS = 40          # Max holding period: 40 sessions (~8 weeks)

def run_grid_backtest_and_plan():
    print("🚀 Starting Nifty 750 Historical Backtest & Dynamic Strategy Engine...")

    # 1. Load institutional universe
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
    print(f"📊 Backtesting across {len(target_stocks)} institutional stocks...")

    # Load all stock histories into memory
    loaded_data = {}
    for f_name in target_stocks:
        sym = f_name.replace(".json", "")
        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                d = json.load(f)
            if len(d) >= 60:
                closes = np.array([float(x["close"]) for x in d])
                highs = np.array([float(x.get("high", x["close"])) for x in d])
                lows = np.array([float(x.get("low", x["close"])) for x in d])
                vols = np.array([float(x.get("delivery_vol", 0)) for x in d])
                
                # Calculate True Demat OBV
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

    # 2. Grid Search to Find the Highest-Probability Strategy Configuration
    # We test combinations of Lookback Windows, Price Drops, OBV Gains, and Confirmation filters
    param_grid = [
        {"name": "Standard Base", "lookback": 15, "min_p_drop": -5.0, "min_obv_gain": 5.0, "trigger": "5D_HIGH"},
        {"name": "Deep Accumulation Base", "lookback": 20, "min_p_drop": -6.0, "min_obv_gain": 7.5, "trigger": "5D_HIGH"},
        {"name": "Institutional Spring Base", "lookback": 25, "min_p_drop": -7.5, "min_obv_gain": 10.0, "trigger": "5D_HIGH"},
        {"name": "Fast Momentum Reversal", "lookback": 10, "min_p_drop": -4.0, "min_obv_gain": 4.0, "trigger": "3D_HIGH"}
    ]

    backtest_stats = []

    for param in param_grid:
        lb = param["lookback"]
        p_drop = param["min_p_drop"]
        obv_gain = param["min_obv_gain"]
        trig_type = param["trigger"]
        trig_window = 5 if trig_type == "5D_HIGH" else 3

        trades = []

        for sym, s_data in loaded_data.items():
            closes = s_data["closes"]
            highs = s_data["highs"]
            lows = s_data["lows"]
            vols = s_data["vols"]
            obvs = s_data["obvs"]
            turnovers = s_data["turnovers"]

            i = lb + trig_window
            while i < len(closes) - 5:
                # Filter: Minimum turnover & non-penny price
                if closes[i] < 30.0 or np.mean(turnovers[max(0, i-8):i+1]) < MIN_DELIVERY_TURNOVER_CR:
                    i += 1
                    continue

                # Divergence condition
                p_chg = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                obv_chg = ((obvs[i] - obvs[i - lb]) / abs(obvs[i - lb])) * 100 if abs(obvs[i - lb]) > 0 else 0

                if p_chg <= p_drop and obv_chg >= obv_gain:
                    # Breakout confirmation trigger
                    ref_high = np.max(highs[i - trig_window:i])
                    vol_surge = vols[i] >= np.mean(vols[max(0, i-8):i])

                    if closes[i] >= ref_high and vol_surge:
                        entry_price = closes[i]
                        base_low = np.min(lows[i - lb:i+1])
                        stop_loss = base_low * 0.99
                        target_price = entry_price * (1 + TARGET_PROFIT_PCT / 100)
                        risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                        if risk_pct <= 7.5:  # Strict Risk Guard: Skip if risk > 7.5%
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
                "model_name": param["name"],
                "params": param,
                "total_trades": len(df_t),
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": round(pf, 2),
                "avg_gain_pct": round(df_t["pnl"].mean(), 2),
                "avg_holding_days": round(df_t["holding"].mean(), 1)
            })

    # Pick the model with the highest Win Rate & Profit Factor
    best_model = max(backtest_stats, key=lambda x: (x["win_rate_pct"], x["profit_factor"]))
    print(f"\n🏆 Best Historical Model: {best_model['model_name']} ({best_model['win_rate_pct']}% Win Rate, PF: {best_model['profit_factor']})")

    # 3. Generate Today's High-Probability Action Plan Using the Winning Model
    active_plan = []
    opt_lb = best_model["params"]["lookback"]
    opt_p_drop = best_model["params"]["min_p_drop"]
    opt_obv_gain = best_model["params"]["min_obv_gain"]
    opt_trig_window = 5 if best_model["params"]["trigger"] == "5D_HIGH" else 3

    for sym, s_data in loaded_data.items():
        closes = s_data["closes"]
        highs = s_data["highs"]
        lows = s_data["lows"]
        vols = s_data["vols"]
        obvs = s_data["obvs"]
        turnovers = s_data["turnovers"]

        if len(closes) < opt_lb + 5 or closes[-1] < 30.0:
            continue

        sma_9_to = np.mean(turnovers[-9:])
        if sma_9_to < MIN_DELIVERY_TURNOVER_CR:
            continue

        # Check Winning Model Divergence
        curr_c = closes[-1]
        past_c = closes[-opt_lb - 1]
        curr_obv = obvs[-1]
        past_obv = obvs[-opt_lb - 1]

        p_chg = ((curr_c - past_c) / past_c) * 100
        obv_chg = ((curr_obv - past_obv) / abs(past_obv)) * 100 if abs(past_obv) > 0 else 0

        if p_chg <= opt_p_drop and obv_chg >= opt_obv_gain:
            ref_high = np.max(highs[-opt_trig_window - 1:-1])
            is_triggered = (curr_c >= ref_high) and (vols[-1] >= np.mean(vols[-9:]))

            base_low = np.min(lows[-opt_lb:])
            sl_price = round(base_low * 0.99, 2)
            risk_pct = round(((curr_c - sl_price) / curr_c) * 100, 2)

            if risk_pct <= 7.5:
                meta = fundamentals.get(sym, {})
                active_plan.append({
                    "Symbol": sym,
                    "Signal": "🟢 BUY TRIGGERED" if is_triggered else "🟡 PENDING BREAKOUT",
                    "Entry (₹)": round(curr_c, 2) if is_triggered else f"> ₹{round(ref_high, 2)}",
                    "Stop Loss (₹)": sl_price,
                    "Risk": f"{risk_pct}%",
                    "Target 1 (+10%)": round(curr_c * 1.10, 2),
                    "Target 2 (+20%)": round(curr_c * 1.20, 2),
                    "9D Turnover": f"₹{sma_9_to:.1f} Cr/d",
                    "Industry": meta.get("industry", "NSE Listed")
                })

    active_plan.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk"].replace("%", ""))), reverse=True)

    # 4. Save Outputs
    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump({"summary": backtest_stats, "winning_model": best_model}, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_plan, f, indent=2)

    print(f"✅ Generated {len(active_plan)} trades under the winning backtested rule set.")

if __name__ == "__main__":
    run_grid_backtest_and_plan()
