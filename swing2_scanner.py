import os
import json
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "swing2_scanner_results.json")
FUNDAMENTALS_FILE = os.path.join(DATA_DIR, "fundamentals.json")
MIN_TURNOVER_CR = 2.0
MIN_PRICE = 10.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def load_market_benchmark():
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
            last_bar = df_b.iloc[-1]
            is_bullish = bool(last_bar["close"] > last_bar["sma_50"])
            return is_bullish, float(last_bar["close"]), float(last_bar["sma_50"])

    return True, 0.0, 0.0

def classify_stock(sym, meta, df):
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

def scan_symbol(sym, df, category):
    if len(df) < 55:
        return None, None

    i = len(df) - 1
    curr = df.iloc[i]
    prev = df.iloc[i - 1]
    is_large = (category == "Large_Cap")

    if curr["turnover_sma20"] < MIN_TURNOVER_CR or curr["close"] < MIN_PRICE:
        return None, None
    if curr["close"] < curr["sma_50"]:
        return None, None

    # Base Depth Detection (Run 3 Universal: Macro <= 22%, Coil <= 10%)
    found_base = False
    pivot_ceiling = 0.0
    pivot_floor = 0.0

    for macro_k in [15, 25, 40]:
        sub_high = df["high"].iloc[i-macro_k:i].max()
        sub_low = df["low"].iloc[i-macro_k:i].min()
        macro_spread = (sub_high - sub_low) / curr["close"]

        if macro_spread <= 0.22:
            coil_high = df["high"].iloc[i-7:i].max()
            coil_low = df["low"].iloc[i-7:i].min()
            coil_spread = (coil_high - coil_low) / curr["close"]

            if coil_spread <= 0.10:
                found_base = True
                pivot_ceiling = sub_high
                pivot_floor = coil_low
                break

    if not found_base:
        return None, None

    # ATR Squeeze
    if (prev["atr_5"] / prev["atr_50"]) > 0.72:
        return None, None

    # Volume Dry-Up (Run 3: 3-day pocket)
    prior_3_days_deliv = df["delivery_vol"].iloc[i-3:i]
    prior_3_days_sma = df["deliv_sma20"].iloc[i-3:i]
    if not (prior_3_days_deliv < (0.60 * prior_3_days_sma)).any():
        return None, None

    # True Demat OBV
    if curr["dobv"] < df["dobv_sma20"].iloc[i-1]:
        return None, None

    # Common metrics
    entry_price = round(float(curr["close"]), 2)
    stop_loss = round(max(float(curr["low"]) * 0.99, pivot_floor), 2)
    initial_risk = max(entry_price - stop_loss, entry_price * 0.015)
    t1_price = round(entry_price + (1.5 * initial_risk), 2)
    risk_pct = round(((entry_price - stop_loss) / entry_price) * 100, 2)

    bar_span = max(float(curr["high"] - curr["low"]), 0.01)
    close_pos = (float(curr["close"]) - float(curr["low"])) / bar_span
    upper_wick = (float(curr["high"]) - max(float(curr["open"]), float(curr["close"]))) / bar_span
    surge_mult = 1.25 if is_large else 1.40

    is_deliv_surge = (curr["delivery_vol"] >= surge_mult * curr["deliv_sma20"]) and (curr["deliv_pct"] >= 35.0)
    is_clean_candle = (close_pos >= 0.70) and (upper_wick <= 0.20)

    # 1. Active Breakout Trigger
    if curr["close"] > pivot_ceiling and is_deliv_surge and is_clean_candle:
        return "TRIGGER", {
            "symbol": sym,
            "category": category,
            "scan_date": curr["std_date"],
            "close": entry_price,
            "pivot_ceiling": round(pivot_ceiling, 2),
            "stop_loss": stop_loss,
            "target_1_5r": t1_price,
            "risk_pct": risk_pct,
            "turnover_cr": round(float(curr["turnover_cr"]), 2),
            "deliv_pct": round(float(curr["deliv_pct"]), 1),
            "deliv_surge_x": round(float(curr["delivery_vol"] / max(curr["deliv_sma20"], 1.0)), 2),
            "status": "CONFIRMED_BREAKOUT"
        }

    # 2. Pre-Breakout Coil Watchlist (within 2.5% of ceiling)
    dist_to_ceiling = ((pivot_ceiling - curr["close"]) / curr["close"]) * 100
    if 0.0 <= dist_to_ceiling <= 2.5:
        return "WATCHLIST", {
            "symbol": sym,
            "category": category,
            "scan_date": curr["std_date"],
            "close": entry_price,
            "pivot_ceiling": round(pivot_ceiling, 2),
            "dist_to_pivot_pct": round(dist_to_ceiling, 2),
            "stop_loss": stop_loss,
            "risk_pct": risk_pct,
            "turnover_cr": round(float(curr["turnover_cr"]), 2),
            "status": "COILING_AT_PIVOT"
        }

    return None, None

def main():
    print("🔎 Running Swing 2.0 (Run 3 Engine) Production Scanner...")
    is_bullish, b_close, b_sma = load_market_benchmark()

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
            "swing2_backtest_report.json", "nifty750.json", "swing2_scanner_results.json"
        ]
    ]

    triggers = []
    watchlist = []

    for f in stock_files:
        sym = f.replace(".json", "").strip().upper()
        json_p = os.path.join(DATA_DIR, f)
        try:
            with open(json_p, "r", encoding="utf-8") as fp:
                candles = json.load(fp)
            if not isinstance(candles, list) or len(candles) < 55:
                continue

            df = pd.DataFrame(candles)
            df = calculate_indicators(df)
            cat = classify_stock(sym, meta, df)

            kind, item = scan_symbol(sym, df, cat)
            if kind == "TRIGGER":
                triggers.append(item)
            elif kind == "WATCHLIST":
                watchlist.append(item)
        except Exception:
            continue

    output = {
        "scan_time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "market_regime": {
            "is_market_above_50sma": is_bullish,
            "status": "BULLISH_TREND" if is_bullish else "CORRECTION_CHOP"
        },
        "engine": "Swing 2.0 (Run 3 Universal Model)",
        "total_tickers_scanned": len(stock_files),
        "total_triggers": len(triggers),
        "total_watchlist": len(watchlist),
        "active_triggers": sorted(triggers, key=lambda x: x["turnover_cr"], reverse=True),
        "pre_breakout_watch": sorted(watchlist, key=lambda x: x["dist_to_pivot_pct"])
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"🎯 Scan complete. Triggers: {len(triggers)} | Watchlist: {len(watchlist)}. Written to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
