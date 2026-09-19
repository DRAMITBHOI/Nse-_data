import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "swing2_backtest_report.json")
MIN_TURNOVER_CR = 2.0  # ₹2 Crore 20-day avg turnover

def calculate_indicators(df):
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["delivery_vol"] = pd.to_numeric(df.get("delivery_vol", df["volume"]), errors="coerce").fillna(0)
    df["deliv_pct"] = pd.to_numeric(df.get("deliv_pct", 0), errors="coerce").fillna(0)

    # 20-day average turnover in ₹ Crore
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_sma20"] = df["turnover_cr"].rolling(20).mean()

    # True Range & ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    
    df["atr_5"] = tr.rolling(5).mean()
    df["atr_50"] = tr.rolling(50).mean()
    df["deliv_sma20"] = df["delivery_vol"].rolling(20).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    return df

def evaluate_symbol(sym, records):
    if len(records) < 100:
        return []

    df = pd.DataFrame(records)
    df = calculate_indicators(df)

    trades = []
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    target_1 = 0.0
    entry_date = ""
    t1_hit = False

    # Backtest simulation loop
    for i in range(55, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        # --- TRADE MANAGEMENT ---
        if in_trade:
            # Check Stop Loss (Intraday Low breach)
            if curr["low"] <= stop_loss:
                exit_price = stop_loss
                pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
                trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": curr["time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "SL_HIT" if not t1_hit else "BE_OR_TRAIL_HIT"
                })
                in_trade = False
                t1_hit = False
                continue

            # Check Target 1 (2R Gain) -> Scale out / move stop to Break-Even
            risk = entry_price - stop_loss
            if not t1_hit and curr["high"] >= (entry_price + 2.0 * risk):
                t1_hit = True
                stop_loss = entry_price * 1.002  # Move to break-even (+0.2% buffer)

            # Check Trailing Stop (Close below 20 EMA after T1)
            if t1_hit and curr["close"] < curr["ema_20"]:
                exit_price = curr["close"]
                pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
                trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": curr["time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "EMA20_TRAIL_EXIT"
                })
                in_trade = False
                t1_hit = False
                continue

            continue

        # --- ENTRY CONDITIONS ---
        # 1. Turnover & Trend Floor
        if curr["turnover_sma20"] < MIN_TURNOVER_CR or curr["close"] < 30.0:
            continue
        if curr["close"] < curr["sma_50"]:
            continue

        # 2. Dynamic Base Detection: Check recent consolidation lengths (12 to 35 bars)
        found_base = False
        best_k = 0
        pivot_ceiling = 0.0
        pivot_floor = 0.0

        for k in [12, 18, 25, 35]:
            sub_high = df["high"].iloc[i-k:i].max()
            sub_low = df["low"].iloc[i-k:i].min()
            spread = (sub_high - sub_low) / curr["close"]

            # Base must be tight (<= 12% range)
            if spread <= 0.12:
                found_base = True
                best_k = k
                pivot_ceiling = sub_high
                pivot_floor = sub_low
                break

        if not found_base:
            continue

        # 3. Volatility Squeeze Check (ATR Contraction)
        if (prev["atr_5"] / prev["atr_50"]) > 0.70:
            continue

        # 4. Asymmetric Delivery Ratio in Base (Accumulation check)
        base_slice = df.iloc[i-best_k:i]
        up_days = base_slice[base_slice["close"] > base_slice["open"]]
        down_days = base_slice[base_slice["close"] < base_slice["open"]]

        avg_up_deliv = up_days["delivery_vol"].mean() if len(up_days) > 0 else 0
        avg_down_deliv = down_days["delivery_vol"].mean() if len(down_days) > 0 else 1
        deliv_ratio = avg_up_deliv / max(avg_down_deliv, 1)

        if deliv_ratio < 1.25:
            continue

        # 5. Breakout Trigger Bar: Price clears ceiling + Delivery surges
        is_breakout = curr["close"] > pivot_ceiling
        is_deliv_surge = (curr["delivery_vol"] >= 1.4 * curr["deliv_sma20"]) and (curr["deliv_pct"] >= 35.0)
        strong_close = ((curr["close"] - curr["low"]) / max((curr["high"] - curr["low"]), 0.01)) >= 0.65

        if is_breakout and is_deliv_surge and strong_close:
            in_trade = True
            entry_price = round(curr["close"], 2)
            entry_date = curr["time"]
            stop_loss = round(max(curr["low"] * 0.99, pivot_floor), 2)
            t1_hit = False

    return trades

def run_backtest():
    print("🚀 Initiating Swing 2.0 Backtest Engine...")
    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "backtest_report.json", "init_progress.json", "fno_history.json",
            "swing2_backtest_report.json"
        ]
    ]

    all_trades = []
    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        p = os.path.join(DATA_DIR, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            trades = evaluate_symbol(sym, data)
            all_trades.extend(trades)
        except Exception:
            continue

    if not all_trades:
        print("No completed trades recorded.")
        return

    df_trades = pd.DataFrame(all_trades)
    win_trades = df_trades[df_trades["pnl_pct"] > 0]
    loss_trades = df_trades[df_trades["pnl_pct"] <= 0]

    win_rate = round((len(win_trades) / len(df_trades)) * 100, 2)
    avg_gain = round(win_trades["pnl_pct"].mean(), 2) if len(win_trades) > 0 else 0
    avg_loss = round(loss_trades["pnl_pct"].mean(), 2) if len(loss_trades) > 0 else 0
    profit_factor = round(abs((win_trades["pnl_pct"].sum()) / max(abs(loss_trades["pnl_pct"].sum()), 0.01)), 2)

    summary = {
        "report_generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "total_trades": len(df_trades),
        "win_rate_pct": win_rate,
        "avg_gain_pct": avg_gain,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "recent_trades": all_trades[-100:]  # Store last 100 executed setups
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n📊 --- BACKTEST SUMMARY ---")
    print(f"Total Trades: {len(df_trades)} | Win Rate: {win_rate}% | Profit Factor: {profit_factor}")
    print(f"Avg Gain: +{avg_gain}% | Avg Loss: {avg_loss}%")
    print(f"📁 Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    run_backtest()
