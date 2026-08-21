import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
MAX_PE = 30.0
MIN_AVG_VOLUME_9D = 100000
MIN_PRICE_DROP_PCT = -7.5
MIN_OBV_GAIN_PCT = 8.0
MIN_LOOKBACK_BARS = 5    # 1 Week (5 trading days)
MAX_LOOKBACK_BARS = 40   # 8 Weeks (40 trading days)

def full_corporate_action_adjustment(raw_data):
    if not raw_data or len(raw_data) < 2:
        return raw_data
    clean = []
    for r in raw_data:
        clean.append({
            "time": r["time"],
            "open": float(r["open"]),
            "high": float(r.get("high", r["close"])),
            "low": float(r.get("low", r["close"])),
            "close": float(r["close"]),
            "delivery_vol": float(r.get("delivery_vol", 0)),
            "volume": float(r.get("volume", r.get("delivery_vol", 0))),
            "deliv_pct": float(r.get("deliv_pct", 0))
        })
    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(clean) - 1, 0, -1):
        prev_c = clean[i - 1]["close"]
        curr_o = clean[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj_factor = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj_factor = k
                        break
                if not adj_factor and 1.70 <= ratio <= 2.30:
                    adj_factor = 2.0
                elif not adj_factor and 4.30 <= ratio <= 5.50:
                    adj_factor = 5.0
                elif not adj_factor and 8.50 <= ratio <= 11.50:
                    adj_factor = 10.0
            if adj_factor:
                for j in range(0, i):
                    clean[j]["open"] = round(clean[j]["open"] / adj_factor, 2)
                    clean[j]["high"] = round(clean[j]["high"] / adj_factor, 2)
                    clean[j]["low"] = round(clean[j]["low"] / adj_factor, 2)
                    clean[j]["close"] = round(clean[j]["close"] / adj_factor, 2)
                    clean[j]["delivery_vol"] = clean[j]["delivery_vol"] * adj_factor
                    clean[j]["volume"] = clean[j]["volume"] * adj_factor
    return clean

def find_swing_lows(arr, left=3, right=3):
    lows = []
    for i in range(left, len(arr) - right):
        window = arr[i - left : i + right + 1]
        val = arr[i]
        if val == np.min(window) and list(window).count(val) == 1:
            lows.append(i)
    return lows

def scan_wyckoff_stocks():
    print("🚀 Running Nifty 750 scanobv(7.5,8,8) Scanner...")

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
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "backtest_report.json", "active_trade_plan.json",
            "wyckoff_screener_results.json"
        ]
    ]

    print(f"📊 Scanning across {len(stock_files)} stocks...")
    results = []

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

        # 1. P/E Filter (< 30)
        stock_fund = fundamentals.get(sym, {})
        pe_val = stock_fund.get("pe", None)
        pe_float = None
        if pe_val is not None:
            try:
                pe_float = float(pe_val)
                if pe_float <= 0 or pe_float >= MAX_PE:
                    continue
            except (ValueError, TypeError):
                pass

        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception:
            continue

        if len(raw) < 50:
            continue

        clean_history = full_corporate_action_adjustment(raw)
        closes = np.array([float(x["close"]) for x in clean_history])
        highs = np.array([float(x["high"]) for x in clean_history])
        lows = np.array([float(x["low"]) for x in clean_history])
        vols = np.array([float(x["delivery_vol"]) for x in clean_history])
        traded_vols = np.array([float(x.get("volume", x["delivery_vol"])) for x in clean_history])
        N = len(closes)

        # 2. 9-Day Average Volume Filter (> 100,000)
        sma_9_vol = float(np.mean(traded_vols[-9:]))
        if sma_9_vol < MIN_AVG_VOLUME_9D:
            continue

        # 3. Calculate True Demat Delivery OBV
        obvs = np.zeros(N)
        cur_obv = 0
        for idx in range(N):
            d_vol = min(vols[idx], traded_vols[idx]) if traded_vols[idx] > 0 else vols[idx]
            if idx > 0:
                if closes[idx] > closes[idx - 1]:
                    cur_obv += d_vol
                elif closes[idx] < closes[idx - 1]:
                    cur_obv -= d_vol
            else:
                cur_obv = d_vol
            obvs[idx] = cur_obv

        # 4. Swing Low Pivot Detection
        obv_lows = find_swing_lows(obvs, 3, 3)
        recent_window_start = max(0, N - 10)
        recent_min_idx = recent_window_start + int(np.argmin(obvs[recent_window_start:]))
        if recent_min_idx not in obv_lows and recent_min_idx >= N - 10:
            obv_lows.append(recent_min_idx)
            obv_lows.sort()

        matched_setup = None

        # 5. Scan for Bullish Divergence (1W to 8W lookback)
        for idx_b in reversed(obv_lows):
            if (N - 1 - idx_b) > 15:
                continue

            for idx_a in reversed(obv_lows):
                span = idx_b - idx_a
                if MIN_LOOKBACK_BARS <= span <= MAX_LOOKBACK_BARS:
                    p_drop = ((closes[idx_b] - closes[idx_a]) / closes[idx_a]) * 100
                    past_o = obvs[idx_a]
                    o_gain = ((obvs[idx_b] - past_o) / abs(past_o)) * 100 if abs(past_o) > 0 else 0

                    if p_drop <= MIN_PRICE_DROP_PCT and o_gain >= MIN_OBV_GAIN_PCT:
                        swing_high = np.max(highs[idx_a : idx_b + 1])
                        base_low = np.min(lows[idx_a : idx_b + 1])
                        sl_price = round(base_low * 0.995, 2)
                        risk_pct = round(((closes[-1] - sl_price) / closes[-1]) * 100, 2)

                        avg_10_vol = np.mean(traded_vols[max(0, N - 11):N - 1])
                        is_triggered = (closes[-1] >= swing_high) and (traded_vols[-1] >= avg_10_vol)

                        matched_setup = {
                            "Symbol": sym,
                            "Signal": "🟢 BUY BREAKOUT" if is_triggered else "🟡 ACCUMULATION BASE",
                            "LTP (₹)": round(closes[-1], 2),
                            "P/E": f"{pe_float:.1f}" if pe_float is not None else "N/A",
                            "Swing High (₹)": round(swing_high, 2),
                            "Stop Loss (₹)": sl_price,
                            "Risk %": f"{risk_pct}%",
                            "Price Drop %": f"{p_drop:.1f}%",
                            "OBV Gain %": f"+{o_gain:.1f}%",
                            "Divergence Span": f"{span // 5}W Base ({span}d)",
                            "9D Avg Volume": f"{int(sma_9_vol):,}"
                        }
                        break
            if matched_setup:
                break

        if matched_setup:
            results.append(matched_setup)

    results.sort(key=lambda x: (x["Signal"].startswith("🟢"), -float(x["Risk %"].replace("%", ""))), reverse=True)

    output_path = os.path.join(DATA_DIR, "wyckoff_screener_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"🎉 Scan Complete! Saved {len(results)} candidates to {output_path}")

if __name__ == "__main__":
    scan_wyckoff_stocks()
