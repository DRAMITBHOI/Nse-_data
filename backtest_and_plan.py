import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover

def run_choch_obv_backtest():
    print("🚀 Starting True OBV Accumulation -> CHoCH Breakout -> OBV Distribution Exit Backtest...")

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
    print(f"📊 Loading history for {len(target_stocks)} Nifty 750 stocks...")

    trades = []
    active_setups = []
    lookback_windows = [5, 10, 15, 20]  # 1W, 2W, 3W, 4W trading days

    for f_name in target_stocks:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception:
            continue

        if len(raw) < 50:
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

        # --- PART 1: HISTORICAL BACKTEST SIMULATION ---
        i = 25
        while i < len(closes) - 5:
            if closes[i] < 30.0 or np.mean(turnovers[max(0, i - 8):i + 1]) < MIN_DELIVERY_TURNOVER_CR:
                i += 1
                continue

            # Check 1W, 2W, 3W, 4W Bullish Divergence
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

            # Check CHoCH Entry Trigger (Close surges above the swing high of the accumulation window)
            swing_high = np.max(highs[i - matched_lb:i])
            if closes[i] > swing_high and vols[i] >= np.mean(vols[max(0, i - 9):i]):
                entry_price = closes[i]
                entry_idx = i
                base_low = np.min(lows[i - matched_lb:i + 1])
                stop_loss = round(base_low * 0.995, 2)
                risk_pct = ((entry_price - stop_loss) / entry_price) * 100

                exit_price = entry_price
                exit_idx = i
                exit_reason = "MAX_HOLD_TIME"

                # Simulate Forward for Exit Condition
                for j in range(i + 1, min(i + 60, len(closes))):
                    # 1. Hard Stop Loss Check
                    if lows[j] <= stop_loss:
                        exit_price = stop_loss
                        exit_idx = j
                        exit_reason = "STOP_LOSS"
                        break

                    # 2. Check Bearish OBV Divergence Exit (Price stable/up, OBV drops >= 5%)
                    # Evaluate over rolling 5 to 15 day window after entry
                    post_entry_span = min(j - entry_idx, 15)
                    if post_entry_span >= 5:
                        p_chg_post = ((closes[j] - closes[j - post_entry_span]) / closes[j - post_entry_span]) * 100
                        ref_obv = obvs[j - post_entry_span]
                        obv_chg_post = ((obvs[j] - ref_obv) / abs(ref_obv)) * 100 if abs(ref_obv) > 0 else 0

                        # Exit condition: Price stable (>= -2%) or rising (>= +5%) while OBV drops <= -5%
                        if (p_chg_post >= -2.0 or closes[j] >= entry_price * 1.05) and obv_chg_post <= -5.0:
                            exit_price = closes[j]
                            exit_idx = j
                            exit_reason = "OBV_DISTRIBUTION_EXIT"
                            break

                    exit_price = closes[j]
                    exit_idx = j

                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    "symbol": sym,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "holding_days": exit_idx - entry_idx,
                    "exit_reason": exit_reason,
                    "risk_pct": risk_pct
                })
                i = exit_idx + 1
                continue
            i += 1

        # --- PART 2: SCAN TODAY'S ACTIVE SIGNALS ---
        if len(closes) >= 25 and closes[-1] >= 30.0:
            sma_9_to = np.mean(turnovers[-9:])
            if sma_9_to >= MIN_DELIVERY_TURNOVER_CR:
                for lb in lookback_windows:
                    p_drop = ((closes[-1] - closes[-lb - 1]) / closes[-lb - 1]) * 100
                    past_o = obvs[-lb - 1]
                    o_gain = ((obvs[-1] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                    if p_drop <= -5.0 and o_gain >= 5.0:
                        swing_high = np.max(highs[-lb - 1:-1])
                        base_low = np.min(lows[-lb - 1:])
                        sl_price = round(base_low * 0.995, 2)
                        risk_pct = round(((closes[-1] - sl_price) / closes[-1]) * 100, 2)
                        is_choch_triggered = closes[-1] >= swing_high and vols[-1] >= np.mean(vols[-10:-1])

                        meta = fundamentals.get(sym, {})
                        active_setups.append({
                            "Symbol": sym,
                            "Signal": "🟢 CHoCH ENTRY TRIGGERED" if is_choch_triggered else "🟡 ACCUMULATING (Awaiting CHoCH)",
                            "LTP (₹)": round(closes[-1], 2),
                            "CHoCH Trigger Level": f"> ₹{round(swing_high, 2)}",
                            "Stop Loss (₹)": sl_price,
                            "Risk %": f"{risk_pct}%",
                            "Lookback": f"{lb//5}W ({lb}D)",
                            "Price Drop": f"{round(p_drop, 1)}%",
                            "OBV Gain": f"+{round(o_gain, 1)}%",
                            "9D Turnover": f"₹{sma_9_to:.1f} Cr/d"
                        })
                        break

    # Calculate Statistics
    df_t = pd.DataFrame(trades)
    if len(df_t) > 0:
        wins = df_t[df_t["pnl_pct"] > 0]
        losses = df_t[df_t["pnl_pct"] <= 0]
        dist_exits = df_t[df_t["exit_reason"] == "OBV_DISTRIBUTION_EXIT"]
        win_rate = (len(wins) / len(df_t)) * 100
        pf = (wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())) if len(losses) > 0 and losses["pnl_pct"].sum() != 0 else 999.0

        summary = {
            "total_trades": len(df_t),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "avg_gain_pct": round(wins["pnl_pct"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loss_pct": round(losses["pnl_pct"].mean(), 2) if len(losses) > 0 else 0,
            "avg_holding_days": round(df_t["holding_days"].mean(), 1),
            "dist_exit_count": len(dist_exits)
        }
    else:
        summary = {"total_trades": 0, "win_rate_pct": 0, "profit_factor": 0, "avg_gain_pct": 0, "avg_loss_pct": 0, "avg_holding_days": 0}

    active_setups.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk %"].replace("%", ""))), reverse=True)

    with open(os.path.join(DATA_DIR, "backtest_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(DATA_DIR, "active_trade_plan.json"), "w") as f:
        json.dump(active_setups, f, indent=2)

    print(f"\n🎉 Backtest complete across {len(df_t)} trades! Win Rate: {summary['win_rate_pct']}%, Profit Factor: {summary['profit_factor']}")

if __name__ == "__main__":
    run_choch_obv_backtest()
