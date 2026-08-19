import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover
MAX_HOLDING_DAYS = 40          # Max holding: ~8 weeks

def ema(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean().values

def run_supply_exhaustion_engine():
    print("🚀 Running Supply Exhaustion + Institutional Pivot Breakout Engine...")

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
    print(f"📊 Analyzing {len(target_stocks)} institutional stocks across historical data...")

    loaded_data = {}
    for f_name in target_stocks:
        sym = f_name.replace(".json", "")
        try:
            with open(os.path.join(DATA_DIR, f_name), "r") as f:
                d = json.load(f)
            if len(d) >= 100:
                closes = np.array([float(x["close"]) for x in d])
                highs = np.array([float(x.get("high", x["close"])) for x in d])
                lows = np.array([float(x.get("low", x["close"])) for x in d])
                vols = np.array([float(x.get("delivery_vol", 0)) for x in d])
                
                # Delivery percentages if available, else standard delivery volume
                deliv_pcts = np.array([float(x.get("delivery_pct", 45.0)) for x in d])

                loaded_data[sym] = {
                    "dates": [x.get("date", "") for x in d],
                    "closes": closes,
                    "highs": highs,
                    "lows": lows,
                    "vols": vols,
                    "deliv_pcts": deliv_pcts,
                    "ema20": ema(closes, 20),
                    "ema50": ema(closes, 50),
                    "turnovers": (closes * vols) / 1e7
                }
        except Exception:
            continue

    # 4 Quantitative Breakout Models with Dynamic Risk Management
    models = [
        {"name": "VCP Supply Dry-Up + Pivot Breakout (+15% Tgt)", "base_len": 15, "dry_mult": 0.65, "breakout_mult": 1.4, "target_pct": 15.0},
        {"name": "Stage 2 Delivery Climax (+18% Tgt)", "base_len": 20, "dry_mult": 0.70, "breakout_mult": 1.5, "target_pct": 18.0},
        {"name": "Tight Coil Momentum (+12% Tgt)", "base_len": 10, "dry_mult": 0.60, "breakout_mult": 1.3, "target_pct": 12.0},
        {"name": "High-Conviction Wyckoff Breakout (+20% Tgt)", "base_len": 25, "dry_mult": 0.65, "breakout_mult": 1.6, "target_pct": 20.0}
    ]

    backtest_stats = []

    for m in models:
        base_len = m["base_len"]
        dry_mult = m["dry_mult"]
        breakout_mult = m["breakout_mult"]
        tgt_pct = m["target_pct"]

        trades = []

        for sym, s_data in loaded_data.items():
            closes = s_data["closes"]
            highs = s_data["highs"]
            lows = s_data["lows"]
            vols = s_data["vols"]
            ema20 = s_data["ema20"]
            ema50 = s_data["ema50"]
            turnovers = s_data["turnovers"]

            i = 55
            while i < len(closes) - 5:
                # 1. Structural Filter: Stage 2 Uptrend (Close > 50 EMA & Price >= ₹35)
                if closes[i] < ema50[i] or closes[i] < 35.0:
                    i += 1
                    continue

                if np.mean(turnovers[max(0, i-8):i+1]) < MIN_DELIVERY_TURNOVER_CR:
                    i += 1
                    continue

                # 2. Check for Supply Dry-Up during recent base consolidation
                avg_vol_20 = np.mean(vols[max(0, i-20):i])
                recent_base_vols = vols[i - base_len:i - 1] if i - base_len >= 0 else vols[:i]
                min_base_vol = np.min(recent_base_vols) if len(recent_base_vols) > 0 else avg_vol_20

                # Dry-up condition: At least one day in base had volume < dry_mult * 20D average
                if min_base_vol > (avg_vol_20 * dry_mult):
                    i += 1
                    continue

                # 3. Pivot Breakout Trigger: Today's Close > Prior 15-Day High + Volume Surge
                pivot_high = np.max(highs[i - base_len:i])
                vol_surge = vols[i] >= (avg_vol_20 * breakout_mult)

                if closes[i] >= pivot_high and vol_surge:
                    entry_price = closes[i]
                    # Stop loss placed right under recent 5-day pivot low
                    pivot_low = np.min(lows[i-5:i+1])
                    stop_loss = round(pivot_low * 0.99, 2)
                    target_price = round(entry_price * (1 + tgt_pct / 100), 2)
                    risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                    # Accept trades with disciplined risk (2.5% to 6.0%)
                    if 2.5 <= risk_pct <= 6.0:
                        outcome = "TIME_EXIT"
                        exit_price = entry_price
                        exit_idx = i
                        trailing_sl = stop_loss

                        for j in range(i + 1, min(i + MAX_HOLDING_DAYS + 1, len(closes))):
                            # Target reached (+15% to +20%)
                            if highs[j] >= target_price:
                                outcome = f"WIN (+{int(tgt_pct)}%)"
                                exit_price = target_price
                                exit_idx = j
                                break
                            
                            # Breakeven Trail: If trade moves +7% in our favor, move SL to Entry
                            if highs[j] >= entry_price * 1.07 and trailing_sl < entry_price:
                                trailing_sl = entry_price

                            # Stop loss hit
                            if lows[j] <= trailing_sl:
                                outcome = "STOP_LOSS" if trailing_sl < entry_price else "BREAKEVEN"
                                exit_price = trailing_sl
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

    best_model = max(backtest_stats, key=lambda x: (x["win_rate_pct"], x["profit_factor"]))
    print(f"\n🏆 Best Strategy Model: {best_model['model_name']} -> {best_model['win_rate_pct']}% Win Rate, PF: {best_model['profit_factor']}")

    # 4. Generate Today's High-Probability Action Plan Using Winning Model
    active_plan = []
    opt = best_model["params"]

    for sym, s_data in loaded_data.items():
        closes = s_data["closes"]
        highs = s_data["highs"]
        lows = s_data["lows"]
        vols = s_data["vols"]
        turnovers = s_data["turnovers"]
        ema50 = s_data["ema50"]

        if len(closes) < 60 or closes[-1] < 35.0:
            continue

        # Trend filter
        if closes[-1] < ema50[-1]:
            continue

        sma_9_to = np.mean(turnovers[-9:])
        if sma_9_to < MIN_DELIVERY_TURNOVER_CR:
            continue

        curr_c = closes[-1]
        avg_vol_20 = np.mean(vols[-21:-1])
        base_vols = vols[-opt["base_len"] - 1:-1]
        min_base_vol = np.min(base_vols) if len(base_vols) > 0 else avg_vol_20

        # Check for Supply Exhaustion in the base
        has_supply_exhaustion = min_base_vol <= (avg_vol_20 * opt["dry_mult"])
        if not has_supply_exhaustion:
            continue

        pivot_high = np.max(highs[-opt["base_len"] - 1:-1])
        is_breakout = curr_c >= pivot_high
        is_vol_expansion = vols[-1] >= (avg_vol_20 * opt["breakout_mult"])

        is_triggered = is_breakout and is_vol_expansion

        pivot_low = np.min(lows[-6:])
        sl_price = round(pivot_low * 0.99, 2)
        risk_pct = round(((curr_c - sl_price) / curr_c) * 100, 2)

        if 2.5 <= risk_pct <= 6.0:
            meta = fundamentals.get(sym, {})
            target_val = round(curr_c * (1 + opt["target_pct"] / 100), 2)

            active_plan.append({
                "Symbol": sym,
                "Status": "🟢 BUY TRIGGERED" if is_triggered else "🟡 SUPPLY DRIED (Watching Breakout)",
                "Entry Level": f"₹{curr_c:.2f}" if is_triggered else f"Breakout > ₹{pivot_high:.2f}",
                "Stop Loss (₹)": sl_price,
                "Risk": f"{risk_pct}%",
                f"Target (+{int(opt['target_pct'])}%)": target_val,
                "9D Deliv (Cr)": f"₹{sma_9_to:.1f} Cr/d",
                "Category": meta.get("category", "Nifty 750")
            })

    active_plan.sort(key=lambda x: (x["Status"].startswith("🟢"), -float(x["Risk"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump({"summary": backtest_stats, "winning_model": best_model}, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_plan, f, indent=2)

    print(f"🎉 Generated {len(active_plan)} high-conviction supply-exhaustion trade setups.")

if __name__ == "__main__":
    run_supply_exhaustion_engine()
