import os
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "swing2_backtest_report.json")
FUNDAMENTALS_FILE = os.path.join(DATA_DIR, "fundamentals.json")
MIN_TURNOVER_CR = 2.0
COOLDOWN_DAYS = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def load_market_benchmark():
    """Builds a normalized benchmark series using core market anchors."""
    for sym in ["RELIANCE", "HDFCBANK", "ICICIBANK", "TCS"]:
        p = os.path.join(DATA_DIR, f"{sym}.json")
        raw = None
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
            except Exception:
                raw = None
        if not raw:
            url = f"https://raw.githubusercontent.com/DRAMITBHOI/Nse-_data/main/data/{sym}.json"
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
            except Exception:
                continue

        if raw and isinstance(raw, list) and len(raw) > 100:
            df_b = pd.DataFrame(raw)
            df_b["dt"] = pd.to_datetime(df_b["time"]).dt.tz_localize(None).dt.normalize()
            df_b["close"] = pd.to_numeric(df_b["close"], errors="coerce")
            df_b = df_b.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)
            df_b["sma_50"] = df_b["close"].rolling(50).mean()
            df_b["is_bullish"] = df_b["close"] > df_b["sma_50"]
            s = df_b.set_index("dt")["is_bullish"]
            print(f"✅ Market benchmark anchor built using {sym} ({len(s)} sessions).")
            return s

    return pd.Series(dtype=bool)

def classify_stock(sym, meta, df):
    """Robust 3-Tier Categorization: Explicit Tags -> Market Cap -> Median Turnover."""
    m = meta.get(sym, {})
    raw_cat = str(m.get("category", "")).lower()

    if any(k in raw_cat for k in ["midcap", "mid cap", "nifty mid"]):
        return "Mid_Cap"
    if any(k in raw_cat for k in ["smallcap", "small cap", "microcap", "nifty small"]):
        return "Small_Micro_Cap"
    if any(k in raw_cat for k in ["large", "nifty 50", "nifty 100", "nifty next 50", "largecap"]):
        return "Large_Cap"

    mcap = m.get("market_cap") or m.get("market_cap_cr") or m.get("mcap", 0)
    try:
        mcap = float(mcap)
        if mcap >= 60000:
            return "Large_Cap"
        elif 15000 <= mcap < 60000:
            return "Mid_Cap"
        elif 0 < mcap < 15000:
            return "Small_Micro_Cap"
    except (ValueError, TypeError):
        pass

    median_to = df["turnover_sma20"].median()
    if pd.isna(median_to):
        median_to = 0.0

    if median_to >= 25.0:
        return "Large_Cap"
    elif median_to >= 7.0:
        return "Mid_Cap"
    else:
        return "Small_Micro_Cap"

def calculate_indicators(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delivery_vol"] = pd.to_numeric(df.get("delivery_vol", df["volume"]), errors="coerce").fillna(0)
    df["deliv_pct"] = pd.to_numeric(df.get("deliv_pct", 0), errors="coerce").fillna(0)

    df["dt"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["std_date"] = df["dt"].dt.strftime("%Y-%m-%d")

    price_diff = df["close"].diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    df["dobv"] = (direction * df["delivery_vol"]).cumsum()
    df["dobv_sma20"] = df["dobv"].rolling(20).mean()

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

    return df

def simulate_engine(sym, df, category, bench_series, engine_type="RUN_3"):
    trades = []
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    initial_risk = 0.0
    entry_date = ""
    is_entry_bullish = True
    t1_hit = False
    peak_gain_pct = 0.0
    last_stopout_idx = -999

    is_large = (category == "Large_Cap")

    for i in range(55, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        c_date = curr["std_date"]
        curr_dt = curr["dt"]

        # --- TRADE MANAGEMENT (50% @ 1.5R + BE, 20 EMA Runner) ---
        if in_trade:
            curr_peak = ((curr["high"] - entry_price) / entry_price) * 100
            if curr_peak > peak_gain_pct:
                peak_gain_pct = round(curr_peak, 2)

            if curr["low"] <= stop_loss:
                runner_exit_price = stop_loss
                runner_pnl = ((runner_exit_price - entry_price) / entry_price) * 100

                if t1_hit:
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
                    "pnl_pct": total_pnl,
                    "peak_gain_pct": peak_gain_pct,
                    "market_above_50sma": is_entry_bullish,
                    "engine": engine_type
                })
                in_trade = False
                t1_hit = False
                last_stopout_idx = i
                continue

            if not t1_hit and curr["high"] >= (entry_price + 1.5 * initial_risk):
                t1_hit = True
                stop_loss = round(entry_price * 1.002, 2)

            if t1_hit and curr["close"] < curr["ema_20"]:
                runner_pnl = ((curr["close"] - entry_price) / entry_price) * 100
                locked_gain = (1.5 * initial_risk / entry_price) * 100
                total_pnl = round(0.5 * locked_gain + 0.5 * runner_pnl, 2)

                trades.append({
                    "symbol": sym,
                    "category": category,
                    "entry_date": entry_date,
                    "exit_date": c_date,
                    "pnl_pct": total_pnl,
                    "peak_gain_pct": peak_gain_pct,
                    "market_above_50sma": is_entry_bullish,
                    "engine": engine_type
                })
                in_trade = False
                t1_hit = False
                continue

            continue

        # --- ENTRY SCREENING ---
        if (i - last_stopout_idx) < COOLDOWN_DAYS:
            continue

        # 1. Price Floor & Turnover
        if engine_type == "RUN_3":
            min_price = 10.0
        else:  # RUN_5
            min_price = 10.0 if curr["turnover_sma20"] >= 10.0 else 30.0

        if curr["turnover_sma20"] < MIN_TURNOVER_CR or curr["close"] < min_price:
            continue
        if curr["close"] < curr["sma_50"]:
            continue

        # 2. Base Depth Configuration
        if engine_type == "RUN_3":
            max_base_depth = 0.22
            max_coil_depth = 0.10
        else:  # RUN_5
            max_base_depth = 0.22 if is_large else 0.14
            max_coil_depth = 0.10 if is_large else 0.09

        found_base = False
        pivot_ceiling = 0.0
        pivot_floor = 0.0

        for macro_k in [15, 25, 40]:
            sub_high = df["high"].iloc[i-macro_k:i].max()
            sub_low = df["low"].iloc[i-macro_k:i].min()
            macro_spread = (sub_high - sub_low) / curr["close"]

            if macro_spread <= max_base_depth:
                coil_high = df["high"].iloc[i-7:i].max()
                coil_low = df["low"].iloc[i-7:i].min()
                coil_spread = (coil_high - coil_low) / curr["close"]

                if coil_spread <= max_coil_depth:
                    found_base = True
                    pivot_ceiling = sub_high
                    pivot_floor = coil_low
                    break

        if not found_base:
            continue

        # 3. ATR Squeeze
        atr_limit = 0.72 if engine_type == "RUN_3" else 0.70
        if (prev["atr_5"] / prev["atr_50"]) > atr_limit:
            continue

        # 4. Volume Dry-Up (VDU)
        if engine_type == "RUN_3":
            prior_3_days_deliv = df["delivery_vol"].iloc[i-3:i]
            prior_3_days_sma = df["deliv_sma20"].iloc[i-3:i]
            if not (prior_3_days_deliv < (0.60 * prior_3_days_sma)).any():
                continue
        else:  # RUN_5: Strict Eve Dry-Up
            if prev["delivery_vol"] > (0.75 * prev["deliv_sma20"]):
                continue

        # 5. True Demat OBV Trend
        if curr["dobv"] < df["dobv_sma20"].iloc[i-1]:
            continue

        # 6. Breakout Trigger & Candle Discipline
        is_breakout = curr["close"] > pivot_ceiling
        surge_mult = 1.25 if is_large else 1.40
        is_deliv_surge = (curr["delivery_vol"] >= surge_mult * curr["deliv_sma20"]) and (curr["deliv_pct"] >= 35.0)

        bar_span = max(curr["high"] - curr["low"], 0.01)
        close_pos = (curr["close"] - curr["low"]) / bar_span
        upper_wick = (curr["high"] - max(curr["open"], curr["close"])) / bar_span

        if engine_type == "RUN_3":
            is_clean_candle = (close_pos >= 0.70) and (upper_wick <= 0.20)
        else:  # RUN_5: Requires Solid Green Body
            body_fraction = (curr["close"] - curr["open"]) / bar_span
            is_clean_candle = (close_pos >= 0.70) and (upper_wick <= 0.20) and (body_fraction >= 0.45)

        if is_breakout and is_deliv_surge and is_clean_candle:
            in_trade = True
            entry_price = round(curr["close"], 2)
            entry_date = c_date
            stop_loss = round(max(curr["low"] * 0.99, pivot_floor), 2)
            initial_risk = max(entry_price - stop_loss, entry_price * 0.015)
            t1_hit = False
            peak_gain_pct = 0.0

            if not bench_series.empty and curr_dt in bench_series.index:
                is_entry_bullish = bool(bench_series.loc[curr_dt])
            elif not bench_series.empty:
                idx_pos = bench_series.index.searchsorted(curr_dt, side="right") - 1
                is_entry_bullish = bool(bench_series.iloc[max(0, idx_pos)])
            else:
                is_entry_bullish = True

    return trades

def compute_metrics(df_sub):
    if df_sub.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_gain": 0.0, "avg_loss": 0.0, "avg_peak_gain": 0.0}
    win = df_sub[df_sub["pnl_pct"] > 0]
    loss = df_sub[df_sub["pnl_pct"] <= 0]
    win_rate = round((len(win) / len(df_sub)) * 100, 2)
    avg_gain = round(win["pnl_pct"].mean(), 2) if not win.empty else 0.0
    avg_loss = round(loss["pnl_pct"].mean(), 2) if not loss.empty else 0.0
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

def run_simultaneous_backtest():
    print("🚀 Running Head-to-Head Shootout: Run 3 vs. Run 5 across all market caps...")

    bench_series = load_market_benchmark()

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

    trades_run3 = []
    trades_run5 = []

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
            cat = classify_stock(sym, meta, df)

            trades_run3.extend(simulate_engine(sym, df, cat, bench_series, engine_type="RUN_3"))
            trades_run5.extend(simulate_engine(sym, df, cat, bench_series, engine_type="RUN_5"))
        except Exception:
            continue

    df3 = pd.DataFrame(trades_run3)
    df5 = pd.DataFrame(trades_run5)

    head_to_head_totals = {
        "Run_3_Total": compute_metrics(df3),
        "Run_5_Total": compute_metrics(df5)
    }

    market_cap_comparison = {
        "Large_Cap (Run 3)": compute_metrics(df3[df3["category"] == "Large_Cap"]),
        "Large_Cap (Run 5)": compute_metrics(df5[df5["category"] == "Large_Cap"]),
        "Mid_Cap (Run 3)": compute_metrics(df3[df3["category"] == "Mid_Cap"]),
        "Mid_Cap (Run 5)": compute_metrics(df5[df5["category"] == "Mid_Cap"]),
        "Small_Micro (Run 3)": compute_metrics(df3[df3["category"] == "Small_Micro_Cap"]),
        "Small_Micro (Run 5)": compute_metrics(df5[df5["category"] == "Small_Micro_Cap"])
    }

    regime_comparison = {
        "Above_50SMA (Run 3)": compute_metrics(df3[df3["market_above_50sma"] == True]),
        "Above_50SMA (Run 5)": compute_metrics(df5[df5["market_above_50sma"] == True]),
        "Below_50SMA (Run 3)": compute_metrics(df3[df3["market_above_50sma"] == False]),
        "Below_50SMA (Run 5)": compute_metrics(df5[df5["market_above_50sma"] == False])
    }

    report = {
        "report_generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "iteration": "Simultaneous Shootout: Run 3 vs Run 5",
        "head_to_head_totals": head_to_head_totals,
        "market_cap_comparison": market_cap_comparison,
        "regime_comparison": regime_comparison,
        "recent_trades_run3": trades_run3[-150:],
        "recent_trades_run5": trades_run5[-150:]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"🎉 Shootout complete! Run 3: {len(df3)} trades | Run 5: {len(df5)} trades.")

if __name__ == "__main__":
    run_simultaneous_backtest()
