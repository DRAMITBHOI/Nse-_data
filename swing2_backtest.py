import os
import io
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "swing2_backtest_report.json")
FUNDAMENTALS_FILE = os.path.join(DATA_DIR, "fundamentals.json")
MIN_TURNOVER_CR = 2.0  # ₹2 Crore turnover baseline
COOLDOWN_DAYS = 10     # Ticker cooldown after stop-out

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def load_nifty500_regime():
    """Loads Nifty benchmark history from local repo or remote fallback."""
    local_p = os.path.join(DATA_DIR, "nifty750.json")
    nifty_map = {}
    raw = None
    
    if os.path.exists(local_p):
        try:
            with open(local_p, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except Exception:
            raw = None

    if not raw:
        url = "https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/nifty750.json"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"⚠️ Nifty fallback notice: {e}")

    if raw and isinstance(raw, list):
        try:
            df_idx = pd.DataFrame(raw)
            df_idx["close"] = pd.to_numeric(df_idx["close"], errors="coerce")
            df_idx["sma_50"] = df_idx["close"].rolling(50).mean()
            df_idx["is_bullish"] = df_idx["close"] > df_idx["sma_50"]
            for _, r in df_idx.iterrows():
                t = str(r["time"])[:10]
                nifty_map[t] = bool(r["is_bullish"])
            print(f"✅ Loaded {len(nifty_map)} sessions of Nifty regime data.")
        except Exception as e:
            print(f"⚠️ Error parsing Nifty regime data: {e}")

    return nifty_map

def calculate_indicators(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delivery_vol"] = pd.to_numeric(df.get("delivery_vol", df["volume"]), errors="coerce").fillna(0)
    df["deliv_pct"] = pd.to_numeric(df.get("deliv_pct", 0), errors="coerce").fillna(0)

    # Turnover & ATR
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_sma20"] = df["turnover_cr"].rolling(20).mean()

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

    # Structural 5-day swing low for trailing exit rule comparison
    df["swing_low_5"] = df["low"].shift(1).rolling(5).min()

    return df

def simulate_trades(sym, df, category, nifty_map, exit_mode="EMA20"):
    trades = []
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    initial_risk = 0.0
    entry_date = ""
    t1_hit = False
    last_stopout_idx = -999

    for i in range(55, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        c_date = str(curr["time"])[:10]

        # --- TRADE MANAGEMENT ---
        if in_trade:
            # 1. Stop Loss check
            if curr["low"] <= stop_loss:
                exit_price = stop_loss
                pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
                trades.append({
                    "symbol": sym,
                    "category": category,
                    "entry_date": entry_date,
                    "exit_date": c_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "SL_HIT" if not t1_hit else "BE_HIT",
                    "nifty_above_50sma": nifty_map.get(entry_date, True),
                    "exit_mode": exit_mode
                })
                in_trade = False
                t1_hit = False
                last_stopout_idx = i  # Activate cooldown
                continue

            # 2. Break-Even Trigger: At 1.5R Gain
            if not t1_hit and curr["high"] >= (entry_price + 1.5 * initial_risk):
                t1_hit = True
                stop_loss = round(entry_price * 1.002, 2)  # BE +0.2% buffer

            # 3. Dynamic Trailing Exits (Active after 1.5R is hit)
            if t1_hit:
                if exit_mode == "EMA20" and curr["close"] < curr["ema_20"]:
                    pnl_pct = round(((curr["close"] - entry_price) / entry_price) * 100, 2)
                    trades.append({
                        "symbol": sym,
                        "category": category,
                        "entry_date": entry_date,
                        "exit_date": c_date,
                        "entry_price": entry_price,
                        "exit_price": round(curr["close"], 2),
                        "pnl_pct": pnl_pct,
                        "exit_reason": "EMA20_TRAIL_EXIT",
                        "nifty_above_50sma": nifty_map.get(entry_date, True),
                        "exit_mode": exit_mode
                    })
                    in_trade = False
                    t1_hit = False
                    continue

                elif exit_mode == "SWING_LOW" and curr["close"] < curr["swing_low_5"]:
                    pnl_pct = round(((curr["close"] - entry_price) / entry_price) * 100, 2)
                    trades.append({
                        "symbol": sym,
                        "category": category,
                        "entry_date": entry_date,
                        "exit_date": c_date,
                        "entry_price": entry_price,
                        "exit_price": round(curr["close"], 2),
                        "pnl_pct": pnl_pct,
                        "exit_reason": "SWING_LOW_EXIT",
                        "nifty_above_50sma": nifty_map.get(entry_date, True),
                        "exit_mode": exit_mode
                    })
                    in_trade = False
                    t1_hit = False
                    continue

            continue

        # --- ENTRY CONDITIONS ---
        # 1. 10-Session Ticker Cooldown Window
        if (i - last_stopout_idx) < COOLDOWN_DAYS:
            continue

        # 2. Liquidity & Baseline Trend
        if curr["turnover_sma20"] < MIN_TURNOVER_CR or curr["close"] < 30.0 or curr["close"] < curr["sma_50"]:
            continue

        # 3. Dynamic Base Consolidation (12 to 35 bars)
        found_base = False
        best_k = 0
        pivot_ceiling = 0.0
        pivot_floor = 0.0

        for k in [12, 18, 25, 35]:
            sub_high = df["high"].iloc[i-k:i].max()
            sub_low = df["low"].iloc[i-k:i].min()
            spread = (sub_high - sub_low) / curr["close"]
            if spread <= 0.12:
                found_base = True
                best_k = k
                pivot_ceiling = sub_high
                pivot_floor = sub_low
                break

        if not found_base:
            continue

        # 4. Volatility Squeeze (ATR Contraction)
        if (prev["atr_5"] / prev["atr_50"]) > 0.70:
            continue

        # 5. Asymmetric Delivery Accumulation
        base_slice = df.iloc[i-best_k:i]
        up_days = base_slice[base_slice["close"] > base_slice["open"]]
        down_days = base_slice[base_slice["close"] < base_slice["open"]]
        avg_up_deliv = up_days["delivery_vol"].mean() if len(up_days) > 0 else 0
        avg_down_deliv = down_days["delivery_vol"].mean() if len(down_days) > 0 else 1

        if (avg_up_deliv / max(avg_down_deliv, 1)) < 1.25:
            continue

        # 6. Breakout Ignition Trigger
        is_breakout = curr["close"] > pivot_ceiling
        is_deliv_surge = (curr["delivery_vol"] >= 1.4 * curr["deliv_sma20"]) and (curr["deliv_pct"] >= 35.0)
        strong_close = ((curr["close"] - curr["low"]) / max((curr["high"] - curr["low"]), 0.01)) >= 0.65

        if is_breakout and is_deliv_surge and strong_close:
            in_trade = True
            entry_price = round(curr["close"], 2)
            entry_date = c_date
            stop_loss = round(max(curr["low"] * 0.99, pivot_floor), 2)
            initial_risk = entry_price - stop_loss
            t1_hit = False

    return trades

def compute_metrics(df_sub):
    if df_sub.empty:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0, "avg_gain": 0, "avg_loss": 0}
    win = df_sub[df_sub["pnl_pct"] > 0]
    loss = df_sub[df_sub["pnl_pct"] <= 0]
    win_rate = round((len(win) / len(df_sub)) * 100, 2)
    avg_gain = round(win["pnl_pct"].mean(), 2) if not win.empty else 0
    avg_loss = round(loss["pnl_pct"].mean(), 2) if not loss.empty else 0
    profit_factor = round(abs(win["pnl_pct"].sum() / max(abs(loss["pnl_pct"].sum()), 0.01)), 2)
    return {
        "trades": len(df_sub),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss
    }

def run_comparative_backtest():
    print("🚀 Starting Refined Comparative Backtest Engine...")
    
    nifty_map = load_nifty500_regime()

    # Load Fundamentals mapping
    meta = {}
    if os.path.exists(FUNDAMENTALS_FILE):
        try:
            with open(FUNDAMENTALS_FILE, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "backtest_report.json", "init_progress.json", "fno_history.json",
            "swing2_backtest_report.json", "nifty750.json"
        ]
    ]

    all_trades_ema = []
    all_trades_swing = []

    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        json_p = os.path.join(DATA_DIR, f)
        try:
            with open(json_p, "r", encoding="utf-8") as fp:
                candles = json.load(fp)
            if not isinstance(candles, list) or len(candles) < 100:
                continue

            df = pd.DataFrame(candles)
            df = calculate_indicators(df)
            cat = meta.get(sym, {}).get("category", "Small/Micro Cap")

            # Simulate with EMA 20 Trailing Exit
            trades_ema = simulate_trades(sym, df, cat, nifty_map, exit_mode="EMA20")
            all_trades_ema.extend(trades_ema)

            # Simulate with Swing Low Trailing Exit
            trades_swing = simulate_trades(sym, df, cat, nifty_map, exit_mode="SWING_LOW")
            all_trades_swing.extend(trades_swing)
        except Exception:
            continue

    df_ema = pd.DataFrame(all_trades_ema)
    df_swing = pd.DataFrame(all_trades_swing)

    report = {
        "report_generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "cooldown_days": COOLDOWN_DAYS,
        "be_threshold_r": 1.5,
        "exit_rule_comparison": {
            "EMA20_Exit": compute_metrics(df_ema),
            "Swing_Low_Exit": compute_metrics(df_swing)
        },
        "nifty_regime_breakdown": {
            "All_Time": compute_metrics(df_ema),
            "When_Nifty_Above_50SMA": compute_metrics(df_ema[df_ema["nifty_above_50sma"] == True]),
            "When_Nifty_Below_50SMA": compute_metrics(df_ema[df_ema["nifty_above_50sma"] == False])
        },
        "market_cap_breakdown": {
            "Large_Cap": compute_metrics(df_ema[df_ema["category"].str.contains("500|Large", case=False, na=False)]),
            "Mid_Cap": compute_metrics(df_ema[df_ema["category"].str.contains("Midcap", case=False, na=False)]),
            "Small_Micro_Cap": compute_metrics(df_ema[df_ema["category"].str.contains("Small|Micro", case=False, na=False)])
        },
        "recent_trades": all_trades_ema[-150:]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"🎉 Comparative Backtest completed and saved to '{REPORT_FILE}'.")

if __name__ == "__main__":
    run_comparative_backtest()
