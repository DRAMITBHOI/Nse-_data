import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_DELIVERY_TURNOVER_CR = 1.5  # Min ₹1.5 Cr/day delivery turnover

# Optimal Parameters from Quantitative Grid Search
ENTRY_PRICE_DROP_PCT = -7.5     # Pullback >= -7.5%
ENTRY_OBV_GAIN_PCT = 8.0        # True OBV Accumulation >= +8.0%
EXIT_OBV_DROP_PCT = -8.0        # True OBV Distribution Exit <= -8.0%

def run_choch_obv_backtest():
    print("🚀 Running Optimized True OBV Base -> CHoCH Breakout -> Distribution Exit Backtest...")

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
        N = len(closes)

        # Compute True Demat Delivery OBV
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

        # -----------------------------------------------------------
        # PART 1: 2-PHASE HISTORICAL BACKTEST ENGINE
        # -----------------------------------------------------------
        i = 25
        while i < N - 15:
            if closes[i] < 30.0 or np.mean(turnovers[max(0, i - 8):i + 1]) < MIN_DELIVERY_TURNOVER_CR:
                i += 1
                continue

            # Stage 1: Detect True Delivery Accumulation Base
            matched_lb = None
            for lb in lookback_windows:
                p_drop = ((closes[i] - closes[i - lb]) / closes[i - lb]) * 100
                past_o = obvs[i - lb]
                o_gain = ((obvs[i] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                if p_drop <= ENTRY_PRICE_DROP_PCT and o_gain >= ENTRY_OBV_GAIN_PCT:
                    matched_lb = lb
                    break

            if not matched_lb:
                i += 1
                continue

            base_high = np.max(highs[i - matched_lb:i + 1])
            base_low = np.min(lows[i - matched_lb:i + 1])
            stop_loss = round(base_low * 0.995, 2)

            # Stage 2: Watch forward up to 15 trading days for a CHoCH Breakout
            trade_entered = False
            for k in range(i + 1, min(i + 16, N)):
                if lows[k] < base_low * 0.97:
                    break

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
                    for m in range(entry_idx + 1, min(entry_idx + 50, N)):
                        if lows[m] <= stop_loss:
                            exit_price = stop_loss
                            exit_idx = m
                            exit_reason = "STOP_LOSS"
                            break

                        bars_in_trade = m - entry_idx
                        if bars_in_trade >= 5:
                            span = min(bars_in_trade, 10)
                            p_chg_win = ((closes[m] - closes[m - span]) / closes[m - span]) * 100
                            ref_o = obvs[m - span]
                            obv_chg_win = ((obvs[m] - ref_o) / abs(ref_o)) * 100 if abs(ref_o) > 0 else 0

                            # Exit when price is stable/rising while OBV drops <= -8.0%
                            if (p_chg_win >= -1.5 or closes[m] >= entry_price * 1.05) and obv_chg_win <= EXIT_OBV_DROP_PCT:
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
        # PART 2: SCAN TODAY'S ACTIVE SETUPS (Safe Absolute Indexing)
        # -----------------------------------------------------------
        if N >= 30 and closes[-1] >= 30.0:
            sma_9_to = np.mean(turnovers[-9:])
            if sma_9_to >= MIN_DELIVERY_TURNOVER_CR:
                matched_for_sym = False
                for lb in lookback_windows:
                    for offset in range(0, 10):
                        curr_pos = (N - 1) - offset
                        base_start = curr_pos - lb

                        if base_start < 0:
                            continue

                        p_drop = ((closes[curr_pos] - closes[base_start]) / closes[base_start]) * 100
                        past_o = obvs[base_start]
                        o_gain = ((obvs[curr_pos] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                        if p_drop <= ENTRY_PRICE_DROP_PCT and o_gain >= ENTRY_OBV_GAIN_PCT:
                            base_h = np.max(highs[base_start : curr_pos + 1])
                            base_l = np.min(lows[base_start : curr_pos + 1])
                            sl = round(base_l * 0.995, 2)
                            risk = round(((closes[-1] - sl) / closes[-1]) * 100, 2)

                            avg_v = np.mean(vols[max(0, N - 11) : N - 1])
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
                            matched_for_sym = True
                            break
                    if matched_for_sym:
                        break

    # -----------------------------------------------------------
    # STATISTICAL EVALUATION & METRICS EXPORT
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
