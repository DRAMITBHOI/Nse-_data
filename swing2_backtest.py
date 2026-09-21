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
    """Loads Nifty benchmark history to evaluate macro market regime."""
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
            print(f"⚠️ Nifty regime download notice: {e}")

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
            print(f"⚠️ Error parsing Nifty regime: {e}")

    return nifty_map

def calculate_indicators(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delivery_vol"] = pd.to_numeric(df.get("delivery_vol", df["volume"]), errors="coerce").fillna(0)
    df["deliv_pct"] = pd.to_numeric(df.get("deliv_pct", 0), errors="coerce").fillna(0)

    # 1. True Demat OBV (Vectorized)
    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    df["dobv"] = (direction * df["delivery_vol"]).cumsum()
    df["dobv_sma20"] = df["dobv"].rolling(20).mean()

    # 2. Turnover & Averages
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

    # Structural 5-day swing low for trailing comparison
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
    peak_gain_pct = 0.0
    last_stopout_idx = -999

    is_large = ("500" in category) or ("Large" in category)

    for i in range(55, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        c_date = str(curr["time"])[:10]

        # --- TRADE MANAGEMENT ---
        if in_trade:
            # Track Peak Gain (MFE) during trade lifespan
            curr_peak = ((curr["high"] - entry_price) / entry_price) * 100
            if curr_peak > peak_gain_pct:
                peak_gain_pct = round(curr_peak, 2)

            # 1. Stop Loss check
            if curr["low"] <= stop_loss:
                runner_exit_price = stop_loss
                runner_pnl = ((runner_exit_price - entry_price) / entry_price) * 100

                if t1_hit:
                    # 50% locked at +1.5R, remaining 50% at BE
                    locked_gain = (1.5 * initial_risk / entry_price) * 100
                    total_pnl = round(0.5 * locked_gain + 0.5 * runner_pnl, 2)
                    exit_reason = "BE_HIT_AFTER_T1"
                else:
                    total_pnl = round(runner_pnl, 2)
                    exit_reason = "SL_HIT"

                trades.append({
                    "symbol": sym,
                    "category": category,
                    "entry_date": entry_date,
                    "exit_date": c_date,
                    "entry_price": entry_price,
                    "exit_price": runner_exit_price,
                    "pnl_pct": total_pnl,
                    "peak_gain_pct": peak_gain_pct,
                    "exit_reason": exit_reason,
                    "nifty_above_50sma": nifty_map.get(entry_date, True),
                    "exit_mode": exit_mode
                })
                in_trade = False
                t1_hit = False
                last_stopout_idx = i
                continue

            # 2. 50% Profit Booking at +1.5R & Move Stop Loss to BE
            if not t1_hit and curr["high"] >= (entry_price + 1.5 * initial_risk):
                t1_hit = True
                stop_loss = round(entry_price * 1.002, 2)  # BE +0.2% slippage buffer

            # 3. Dynamic Runner Trailing Exit (Active after +1.5R achieved)
            if t1_hit:
                triggered_exit = False
                if exit_mode == "EMA20" and curr["close"] < curr["ema_20"]:
                    triggered_exit = True
                    exit_reason = "EMA20_TRAIL_EXIT"
                elif exit_mode == "SWING_LOW" and curr["close"] < curr["swing_low_5"]:
                    triggered_exit = True
                    exit_reason = "SWING_LOW_EXIT"

                if triggered_exit:
                    runner_pnl = ((curr["close"] - entry_price) / entry_price) * 100
                    locked_gain = (1.5 * initial_risk / entry_price) * 100
                    total_pnl = round(0.5 * locked_gain + 0.5 * runner_pnl, 2)

                    trades.append({
                        "symbol": sym,
                        "category": category,
                        "entry_date": entry_date,
                        "exit_date": c_date,
                        "entry_price": entry_price,
                        "exit_price": round(curr["close"], 2),
                        "pnl_pct": total_pnl,
                        "peak_gain_pct": peak_gain_pct,
                        "exit_reason": exit_reason,
                        "nifty_above_50sma": nifty_map.get(entry_date, True),
                        "exit_mode": exit_mode
                    })
                    in_trade = False
                    t1_hit = False
                    continue

            continue

        # --- ENTRY SCREENING ---
        # 1. 10-Session Cooldown
        if (i - last_stopout_idx) < COOLDOWN_DAYS:
            continue

        # 2. Liquidity & Price Floor (Lowered to ₹10 for split/bonus liquidity leaders)
        if curr["turnover_sma20"] < MIN_TURNOVER_CR or curr["close"] < 10.0:
            continue
        if curr["close"] < curr["sma_50"]:
            continue

        # 3. Base Architecture (Two-Stage Base Detection: Macro 15-45 bars + Micro Coil 7-10 bars)
        found_base = False
        pivot_ceiling = 0.0
        pivot_floor = 0.0

        for macro_k in [15, 25, 40]:
            sub_high = df["high"].iloc[i-macro_k:i].max()
            sub_low = df["low"].iloc[i-macro_k:i].min()
            macro_spread = (sub_high - sub_low) / curr["close"]

            # Macro depth <= 22%
            if macro_spread <= 0.22:
                # Check micro coil (last 7 bars tightness <= 10%)
                coil_high = df["high"].iloc[i-7:i].max()
                coil_low = df["low"].iloc[i-7:i].min()
                coil_spread = (coil_high - coil_low) / curr["close"]

                if coil_spread <= 0.10:
                    found_base = True
                    pivot_ceiling = sub_high
                    pivot_floor = coil_low
                    break

        if not found_base:
            continue

        # 4. Volatility Squeeze (ATR Contraction)
        if (prev["atr_5"] / prev["atr_50"]) > 0.72:
            continue

        # 5. Preceding 3-Day Delivery Squeeze (Volume Dry-Up / VDU)
        # At least one day in previous 3 bars must show delivery volume <= 60% of 20 SMA
        prior_3_days_deliv = df["delivery_vol"].iloc[i-3:i]
        prior_3_days_sma = df["deliv_sma20"].iloc[i-3:i]
        has_vdu = (prior_3_days_deliv < (0.60 * prior_3_days_sma)).any()
        if not has_vdu:
            continue

        # 6. True Demat OBV Trend Check
        # Ensure dOBV is holding above its 20-period baseline or rising in the base
        if curr["dobv"] < df["dobv_sma20"].iloc[i-1]:
            continue

        # 7. Ignition Bar & Wick Discipline
        is_breakout = curr["close"] > pivot_ceiling
        surge_mult = 1.25 if is_large else 1.40
        is_deliv_surge = (curr["delivery_vol"] >= surge_mult * curr["deliv_sma20"]) and (curr["deliv_pct"] >= 35.0)

        # Candle quality: Close in top 30% of range; upper wick <= 20% of range
        bar_span = max(curr["high"] - curr["low"], 0.01)
        close_pos = (curr["close"] - curr["low"]) / bar_span
        upper_wick = (curr["high"] - max(curr["open"], curr["close"])) / bar_span

        is_clean_candle = (close_pos >= 0.70) and (upper_wick <= 0.20)

        if is_breakout and is_deliv_surge and is_clean_candle:
            in_trade = True
            entry_price = round(curr["close"], 2)
            entry_date = c_date
            stop_loss = round(max(curr["low"] * 0.99, pivot_floor), 2)
            initial_risk = max(entry_price - stop_loss, entry_price * 0.015)
            t1_hit = False
            peak_gain_pct = 0.0

    return trades

def compute_metrics(df_sub):
    if df_sub.empty:
        return {"trades": 0, "win_rate": 0, "profit_factor": 0, "avg_gain": 0, "avg_loss": 0, "avg_peak_gain": 0}
    win = df_sub[df_sub["pnl_pct"] > 0]
    loss = df_sub[df_sub["pnl_pct"] <= 0]
    win_rate = round((len(win) / len(df_sub)) * 100, 2)
    avg_gain = round(win["pnl_pct"].mean(), 2) if not win.empty else 0
    avg_loss = round(loss["pnl_pct"].mean(), 2) if not loss.empty else 0
    avg_peak = round(df_sub["peak_gain_pct"].mean(), 2)
    profit_factor = round(abs(win["pnl_pct"].sum() / max(abs(loss["pnl_pct"].sum()), 0.01)), 2)
    return {
        "trades": len(df_sub),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "avg_peak_gain": avg_peak
    }

def run_comparative_backtest():
    print("🚀 Starting Refined Swing 2.0 Comprehensive Engine...")

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

            # Robust classification: if not Large or Midcap, tag as Small/Micro
            raw_cat = meta.get(sym, {}).get("category", "")
            if "500" in raw_cat or "Large" in raw_cat:
                cat = "Large_Cap"
            elif "Midcap" in raw_cat:
                cat = "Mid_Cap"
            else:
                cat = "Small_Micro_Cap"

            # Simulate both exit models side by side
            trades_ema = simulate_trades(sym, df, cat, nifty_map, exit_mode="EMA20")
            all_trades_ema.extend(trades_ema)

            trades_swing = simulate_trades(sym, df, cat, nifty_map, exit_mode="SWING_LOW")
            all_trades_swing.extend(trades_swing)
        except Exception:
            continue

    df_ema = pd.DataFrame(all_trades_ema)
    df_swing = pd.DataFrame(all_trades_swing)

    report = {
        "report_generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "features": "dOBV + VDU + Wick Discipline + 1.5R 50% Partial Book",
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
            "Large_Cap": compute_metrics(df_ema[df_ema["category"] == "Large_Cap"]),
            "Mid_Cap": compute_metrics(df_ema[df_ema["category"] == "Mid_Cap"]),
            "Small_Micro_Cap": compute_metrics(df_ema[df_ema["category"] == "Small_Micro_Cap"])
        },
        "recent_trades": all_trades_ema[-150:]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"🎉 Backtest complete. Total trades logged: {len(df_ema)}. Report saved to '{REPORT_FILE}'.")

if __name__ == "__main__":
    run_comparative_backtest()
