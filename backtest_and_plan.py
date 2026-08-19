import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.0  # Min ₹1.0 Cr/day delivery turnover
MAX_HOLDING_DAYS = 40          # Max holding duration: 40 sessions (~8 weeks)

def run_quant_engine():
    print("🚀 Running Nifty 750 True Delivery OBV Swing Backtest Engine...")

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
    print(f"📊 Loading history for {len(target_stocks)} institutional stocks...")

    loaded_data = {}
    for f_name in target_stocks:
        sym = f_name.replace(".json", "")
        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                d = json.load(f)
            if len(d) >= 40:
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

    # 4 Strategy Models (Testing different lookbacks & targets)
    models = [
        {"name": "Swing Momentum (+15% Target)", "lb": 10, "p_drop": -3.0, "obv_gain": 3.0, "target_pct": 15.0, "vol_mult": 1.1},
        {"name": "High-Conviction Base (+20% Target)", "lb": 15, "p_drop": -5.0, "obv_gain": 5.0, "target_pct": 20.0, "vol_mult": 1.2},
        {"name": "Deep Accumulation (+25% Target)", "lb": 20, "p_drop": -7.0, "obv_gain": 7.0, "target_pct": 25.0, "vol_mult": 1.2},
        {"name": "Quick Pulse (+10% Target)", "lb": 10, "p_drop": -3.0, "obv_gain": 4.0, "target_pct": 10.0, "vol_mult": 1.0}
    ]

    backtest_stats = []

    for m in models:
        lb = m["lb"]
        p_drop = m["p_drop"]
        obv_gain = m["obv_gain"]
        tgt_pct = m["target_pct"]
        vol_mult = m["vol_mult"]

        trades = []

        for sym, s_data in loaded_data.items():
            closes = s_data["closes"]
            highs = s_data["highs"]
            lows = s_data["lows"]
            vols = s_data["vols"]
            obvs = s_data["obvs"]
            turnovers = s_data["turnovers"]

            i = lb + 6
            while i < len(closes) - 5:
                if closes[i] < 30.0 or np.mean(turnovers[max(0, i-8):i+1]) < MIN_DELIVERY_TURNOVER_CR:
                    i += 1
                    continue

                # 1. Divergence Condition over lookback window
                p_chg = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                obv_chg = ((obvs[i] - obvs[i - lb]) / abs(obvs[i - lb])) * 100 if abs(obvs[i - lb]) > 0 else 0

                if p_chg <= p_drop and obv_chg >= obv_gain:
                    # 2. Breakout Trigger: Close > 5-Day High + Delivery volume confirmation
                    recent_5d_high = np.max(highs[i-5:i])
                    avg_vol_10 = np.mean(vols[max(0, i-10):i])

                    if closes[i] >= recent_5d_high and vols[i] >= avg_vol_10 * vol_mult:
                        entry_price = closes[i]
                        # Realistic Swing Stop Loss = Just below 5-day swing low
                        swing_low = np.min(lows[i-5:i+1])
                        stop_loss = round(swing_low * 0.99, 2)
                        target_price = round(entry_price * (1 + tgt_pct / 100), 2)
                        risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                        # Accept setups with risk between 2.0% and 6.5%
                        if 2.0 <= risk_pct <= 6.5:
                            outcome = "TIME_EXIT"
                            exit_price = entry_price
                            exit_idx = i

                            for j in range(i + 1, min(i + MAX_HOLDING_DAYS + 1, len(closes))):
                                if highs[j] >= target_price:
                                    outcome = f"WIN (+{int(tgt_pct)}%)"
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
            wins = df_t[df_t["outcome"].str.startswith("WIN")]
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

    if not backtest_stats:
        print("⚠️ No models qualified.")
        return

    best_model = max(backtest_stats, key=lambda x: (x["profit_factor"], x["win_rate_pct"]))
    print(f"\n🏆 Best Strategy: {best_model['model_name']} ({best_model['win_rate_pct']}% Win Rate, PF: {best_model['profit_factor']}, {best_model['total_trades']} trades)")

    # Generate Today's Trade Plan Using the Best Model
    active_plan = []
    opt = best_model["params"]

    for sym, s_data in loaded_data.items():
        closes = s_data["closes"]
        highs = s_data["highs"]
        lows = s_data["lows"]
        vols = s_data["vols"]
        turnovers = s_data["turnovers"]
        obvs = s_data["obvs"]

        if len(closes) < opt["lb"] + 10 or closes[-1] < 30.0:
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

        if p_chg <= opt["p_drop"] and obv_chg >= opt["obv_gain"]:
            recent_5d_high = np.max(highs[-6:-1])
            avg_vol_10 = np.mean(vols[-11:-1])
            
            is_triggered = (curr_c >= recent_5d_high) and (vols[-1] >= avg_vol_10 * opt["vol_mult"])
            
            swing_low = np.min(lows[-6:])
            sl_price = round(swing_low * 0.99, 2)
            risk_pct = round(((curr_c - sl_price) / curr_c) * 100, 2)

            if 2.0 <= risk_pct <= 6.5:
                meta = fundamentals.get(sym, {})
                target_val = round(curr_c * (1 + opt["target_pct"] / 100), 2)
                
                active_plan.append({
                    "Symbol": sym,
                    "Signal": "🟢 BUY TRIGGERED" if is_triggered else "🟡 PENDING BREAKOUT",
                    "Entry (₹)": round(curr_c, 2) if is_triggered else f"> ₹{recent_5d_high:.2f}",
                    "Stop Loss (₹)": sl_price,
                    "Risk": f"{risk_pct}%",
                    f"Target (+{int(opt['target_pct'])}%)": target_val,
                    "9D Turnover": f"₹{sma_9_to:.1f} Cr/d",
                    "Category": meta.get("category", "Nifty 750")
                })

    active_plan.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump({"summary": backtest_stats, "winning_model": best_model}, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_plan, f, indent=2)

    print(f"🎉 Generated {len(active_plan)} actionable trade setups.")

if __name__ == "__main__":
    run_quant_engine()
