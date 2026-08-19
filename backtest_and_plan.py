import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover

def run_choch_obv_backtest():
    print("🚀 Running True OBV Base -> CHoCH Breakout -> Distribution Exit Backtest...")

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

    trades = []
    active_setups = []
    lookback_windows = [5, 10, 15, 20]  # 1W, 2W, 3W, 4W

    for f_name in target_stocks:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception:
            continue

        if len(raw) < 60:
            continue

        closes = np.array([float(x["close"]) for x in raw])
        highs = np.array([float(x.get("high", x["close"])) for x in raw])
        lows = np.array([float(x.get("low", x["close"])) for x in raw])
        vols = np.array([float(x.get("delivery_vol", 0)) for x in raw])
        turnovers = (closes * vols) / 1e7

        # Compute True Demat Delivery OBV
        obvs = np.zeros(len(closes))
        cur_obv = 0
        for idx in range(len(closes)):
            if idx > 0:
                if closes[idx] > closes[idx - 1]:
                    cur_obv += vols[idx]
                elif closes[idx] < closes[idx - 1]:
                    cur_obv -= vols[idx]
            else:
                cur_obv = vols[idx]
            obvs[idx] = cur_obv

        # -----------------------------------------------------------
        # PART 1: 2-PHASE HISTORICAL BACKTEST ENGINE
        # -----------------------------------------------------------
        i = 25
        while i < len(closes) - 15:
            if closes[i] < 30.0 or np.mean(turnovers[max(0, i - 8):i + 1]) < MIN_DELIVERY_TURNOVER_CR:
                i += 1
                continue

            # Stage 1: Detect True Delivery Accumulation Base
            matched_lb = None
            for lb in lookback_windows:
                p_drop = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                past_o = obvs[i - lb]
                o_gain = ((obvs[i] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                if p_drop <= -5.0 and o_gain >= 5.0:
                    matched_lb = lb
                    break

            if not matched_lb:
                i += 1
                continue

            # Base parameters
            base_high = np.max(highs[i - matched_lb:i + 1])
            base_low = np.min(lows[i - matched_lb:i + 1])
            stop_loss = round(base_low * 0.995, 2)

            # Stage 2: Watch forward up to 15 trading days for a CHoCH Breakout
            trade_entered = False
            for k in range(i + 1, min(i + 16, len(closes))):
                # If price drops severely below base low before breaking out, base is invalidated
                if lows[k] < base_low * 0.97:
                    break

                # CHoCH Breakout Trigger: Close > Base High + Volume Confirmation
                avg_v_10 = np.mean(vols[max(0, k - 10):k])
                if closes[k] > base_high and vols[k] >= avg_v_10:
                    trade_entered = True
                    entry_price = closes[k]
                    entry_idx = k
                    risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                    exit_price = entry_price
                    exit_idx = entry_idx
                    exit_reason = "MAX_TIME_EXIT"

                    # Stage 3: Position Tracking & Distribution Exit
                    for m in range(entry_idx + 1, min(entry_idx + 50, len(closes))):
                        # Hard Stop Loss Check
                        if lows[m] <= stop_loss:
                            exit_price = stop_loss
                            exit_idx = m
                            exit_reason = "STOP_LOSS"
                            break

                        # Distribution Divergence Exit Check (rolling 5-10 day window)
                        bars_in_trade = m - entry_idx
                        if bars_in_trade >= 5:
                            span = min(bars_in_trade, 10)
                            p_chg_win = ((closes[m] - closes[m - span]) / closes[m - span]) * 100
                            ref_o = obvs[m - span]
                            obv_chg_win = ((obvs[m] - ref_o) / abs(ref_o)) * 100 if abs(ref_o) > 0 else 0

                            # Exit when price is steady/rising while OBV drops >= 5%
                            if (p_chg_win >= -1.5 or closes[m] >= entry_price * 1.05) and obv_chg_win <= -5.0:
                                exit_price = closes[m]
                                exit_idx = m
                                exit_reason = "OBV_DISTRIBUTION_EXIT"
                                break

                        exit_price = closes[m]
                        exit_idx = m

                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    trades.append({
                        "symbol": sym,
                        "pnl_pct": pnl_pct,
                        "holding_days": exit_idx - entry_idx,
                        "exit_reason": exit_reason,
                        "risk_pct": risk_pct
                    })
                    i = exit_idx + 1
                    break

            if not trade_entered:
                i += 1

        # -----------------------------------------------------------
        # PART 2: SCAN TODAY'S ACTIVE SETUPS
        # -----------------------------------------------------------
        if len(closes) >= 30 and closes[-1] >= 30.0:
            sma_9_to = np.mean(turnovers[-9:])
            if sma_9_to >= MIN_DELIVERY_TURNOVER_CR:
                # Check if stock formed an accumulation base in the last 15 days
                for lb in lookback_windows:
                    for offset in range(0, 10):
                        idx = -1 - offset
                        p_drop = ((closes[idx] - closes[idx - lb]) / closes[idx - lb]) * 100
                        past_o = obvs[idx - lb]
                        o_gain = ((obvs[idx] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                        if p_drop <= -5.0 and o_gain >= 5.0:
                            base_h = np.max(highs[idx - lb:idx + 1])
                            base_l = np.min(lows[idx - lb:idx + 1])
                            sl = round(base_l * 0.995, 2)
                            risk = round(((closes[-1] - sl) / closes[-1]) * 100, 2)

                            is_triggered = closes[-1] >= base_h and vols[-1] >= np.mean(vols[-11:-1])

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
                            break
                    if len(active_setups) and active_setups[-1]["Symbol"] == sym:
                        break

    # -----------------------------------------------------------
    # STATISTICAL EVALUATION
    # -----------------------------------------------------------
    df_t = pd.DataFrame(trades)
    if len(df_t) > 0:
        wins = df_t[df_t["pnl_pct"] > 0]
        losses = df_t[df_t["pnl_pct"] <= 0]
        win_rate = (len(wins) / len(df_t)) * 100
        pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else 999.0

        summary = {
            "total_trades": len(df_t),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "avg_gain_pct": round(wins["pnl_pct"].mean(), 2) if len(wins) > 0 else 0.0,
            "avg_loss_pct": round(losses["pnl_pct"].mean(), 2) if len(losses) > 0 else 0.0,
            "avg_holding_days": round(df_t["holding_days"].mean(), 1)
        }
    else:
        summary = {"total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0, "avg_gain_pct": 0.0, "avg_loss_pct": 0.0, "avg_holding_days": 0.0}

    active_setups.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk %"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_setups, f, indent=2)

    print(f"\n🎉 Backtest Completed: {summary['total_trades']} Trades | Win Rate: {summary['win_rate_pct']}% | PF: {summary['profit_factor']}")

if __name__ == "__main__":
    run_choch_obv_backtest()
