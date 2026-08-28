import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.json")

def safe_float(v, default=0.0):
    try:
        if v is None: return default
        return float(str(v).replace(",", "").replace("%", ""))
    except Exception:
        return default

def calculate_sma(series, period):
    if len(series) < period:
        return [0.0] * len(series)
    return pd.Series(series).rolling(window=period).mean().fillna(0).tolist()

def calculate_rsi(closes, period=2):
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    delta = pd.Series(closes).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()

def run_multi_strategy_backtest():
    print("🚀 Running Multi-Strategy Backtest on NSE Historical Data...")
    if not os.path.exists(DATA_DIR):
        print(f"❌ '{DATA_DIR}' directory missing.")
        return

    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json",
            "wyckoff_screener_results.json", "active_trade_plan.json",
            "backtest_report.json"
        ]
    ]

    strategies = {
        "1. VCP Momentum Breakout": {"trades": 0, "wins": 0, "losses": 0, "returns": []},
        "2. Wyckoff True OBV Divergence": {"trades": 0, "wins": 0, "losses": 0, "returns": []},
        "3. High Tight Flag (Momentum)": {"trades": 0, "wins": 0, "losses": 0, "returns": []},
        "4. RSI-2 Climax Mean Reversion": {"trades": 0, "wins": 0, "losses": 0, "returns": []}
    }

    for f in files:
        path = os.path.join(DATA_DIR, f)
        try:
            with open(path, "r") as fp:
                raw = json.load(fp)
        except Exception:
            continue

        if not isinstance(raw, list) or len(raw) < 120:
            continue

        d_map = {}
        for r in raw:
            if isinstance(r, dict) and "time" in r:
                t = str(r["time"]).split(" ")[0].split("T")[0]
                d_map[t] = r
        clean = [d_map[k] for k in sorted(d_map.keys())]

        if len(clean) < 100:
            continue

        # Extract OHLVC & Delivery Data
        opens, highs, lows, closes, vols, d_vols = [], [], [], [], [], []
        cur_obv = 0
        deliv_obvs = []

        for i, r in enumerate(clean):
            c = safe_float(r.get("close"))
            h = safe_float(r.get("high"), c)
            l = safe_float(r.get("low"), c)
            o = safe_float(r.get("open"), c)
            dv = safe_float(r.get("delivery_vol"))
            tv = safe_float(r.get("volume"), dv)
            if tv > 0 and dv > tv: dv = tv

            if i > 0:
                if c > closes[-1]: cur_obv += dv
                elif c < closes[-1]: cur_obv -= dv
            
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            vols.append(tv)
            d_vols.append(dv)
            deliv_obvs.append(cur_obv)

        sma50 = calculate_sma(closes, 50)
        sma200 = calculate_sma(closes, 200)
        vol_ma20 = calculate_sma(vols, 20)
        rsi2 = calculate_rsi(closes, 2)
        n = len(closes)

        # -------------------------------------------------------------
        # Strategy 1: VCP Momentum Breakout
        # -------------------------------------------------------------
        idx = 60
        while idx < n - 40:
            if closes[idx] > sma50[idx] > sma200[idx] and sma50[idx] > 0:
                # 5-day volatility contraction
                c_range = (max(highs[idx-5:idx]) - min(lows[idx-5:idx])) / closes[idx]
                if c_range <= 0.05 and vols[idx] >= 1.4 * vol_ma20[idx]:
                    if closes[idx] > max(highs[idx-10:idx]):
                        entry = closes[idx]
                        stop = min(lows[idx-5:idx+1]) * 0.99
                        risk = entry - stop
                        if risk > 0 and (risk / entry) <= 0.08:
                            target = entry + (2.5 * risk)
                            eval_w = range(idx + 1, min(idx + 45, n))
                            hit_win = any(highs[j] >= target for j in eval_w)
                            hit_loss = any(lows[j] <= stop for j in eval_w)
                            
                            strategies["1. VCP Momentum Breakout"]["trades"] += 1
                            ret = ((closes[eval_w[-1]] - entry) / entry) * 100
                            strategies["1. VCP Momentum Breakout"]["returns"].append(ret)
                            if hit_win and not hit_loss:
                                strategies["1. VCP Momentum Breakout"]["wins"] += 1
                            else:
                                strategies["1. VCP Momentum Breakout"]["losses"] += 1
                            idx += 20
                            continue
            idx += 1

        # -------------------------------------------------------------
        # Strategy 2: Wyckoff True OBV Divergence (Confirmed Base)
        # -------------------------------------------------------------
        idx = 60
        while idx < n - 40:
            s = 20 # 4-week base
            p_start = closes[idx - s]
            p_low = min(closes[idx - s : idx + 1])
            p_drop = ((p_low - p_start) / p_start) * 100
            o_gain = ((deliv_obvs[idx] - deliv_obvs[idx-s]) / (abs(deliv_obvs[idx-s]) + 1e-9)) * 100

            if p_drop <= -6.0 and o_gain >= 8.0:
                swing_h = max(highs[idx-s:idx+1])
                stop = min(lows[idx-s:idx+1])
                # Check for breakout within 12 days
                for b_idx in range(idx + 1, min(idx + 12, n - 30)):
                    if closes[b_idx] > swing_h:
                        entry = closes[b_idx]
                        risk = entry - stop
                        if risk > 0 and (risk / entry) <= 0.10:
                            target = entry + (2.0 * risk)
                            eval_w = range(b_idx + 1, min(b_idx + 45, n))
                            hit_win = any(highs[j] >= target for j in eval_w)
                            hit_loss = any(lows[j] <= stop for j in eval_w)
                            
                            strategies["2. Wyckoff True OBV Divergence"]["trades"] += 1
                            ret = ((closes[eval_w[-1]] - entry) / entry) * 100
                            strategies["2. Wyckoff True OBV Divergence"]["returns"].append(ret)
                            if hit_win and not hit_loss:
                                strategies["2. Wyckoff True OBV Divergence"]["wins"] += 1
                            else:
                                strategies["2. Wyckoff True OBV Divergence"]["losses"] += 1
                            idx = b_idx + 15
                            break
                    if lows[b_idx] < stop: break
            idx += 1

        # -------------------------------------------------------------
        # Strategy 3: High Tight Flag (Momentum Surge)
        # -------------------------------------------------------------
        idx = 60
        while idx < n - 40:
            # 15-day surge >= 18%
            move = ((closes[idx-4] - closes[idx-18]) / closes[idx-18]) * 100
            if move >= 18.0:
                flag_high = max(highs[idx-4:idx+1])
                flag_low = min(lows[idx-4:idx+1])
                pullback = (flag_high - flag_low) / flag_high
                if pullback <= 0.08 and closes[idx] > flag_high:
                    entry = closes[idx]
                    stop = flag_low * 0.995
                    risk = entry - stop
                    if risk > 0:
                        target = entry + (2.5 * risk)
                        eval_w = range(idx + 1, min(idx + 35, n))
                        hit_win = any(highs[j] >= target for j in eval_w)
                        hit_loss = any(lows[j] <= stop for j in eval_w)

                        strategies["3. High Tight Flag (Momentum)"]["trades"] += 1
                        ret = ((closes[eval_w[-1]] - entry) / entry) * 100
                        strategies["3. High Tight Flag (Momentum)"]["returns"].append(ret)
                        if hit_win and not hit_loss:
                            strategies["3. High Tight Flag (Momentum)"]["wins"] += 1
                        else:
                            strategies["3. High Tight Flag (Momentum)"]["losses"] += 1
                        idx += 15
                        continue
            idx += 1

        # -------------------------------------------------------------
        # Strategy 4: RSI-2 Climax Mean Reversion
        # -------------------------------------------------------------
        idx = 60
        while idx < n - 30:
            if closes[idx] > sma200[idx] and sma200[idx] > 0:
                if rsi2[idx] < 8.0:
                    entry = closes[idx]
                    stop = entry * 0.95 # 5% stop
                    target = entry * 1.08 # 8% target or exit on RSI > 75
                    eval_w = range(idx + 1, min(idx + 25, n))
                    hit_win = any(highs[j] >= target or rsi2[j] >= 75.0 for j in eval_w)
                    hit_loss = any(lows[j] <= stop for j in eval_w)

                    strategies["4. RSI-2 Climax Mean Reversion"]["trades"] += 1
                    ret = ((closes[eval_w[-1]] - entry) / entry) * 100
                    strategies["4. RSI-2 Climax Mean Reversion"]["returns"].append(ret)
                    if hit_win and not hit_loss:
                        strategies["4. RSI-2 Climax Mean Reversion"]["wins"] += 1
                    else:
                        strategies["4. RSI-2 Climax Mean Reversion"]["losses"] += 1
                    idx += 10
                    continue
            idx += 1

    summary = []
    for name, stat in strategies.items():
        t = stat["trades"]
        if t > 0:
            wr = (stat["wins"] / t) * 100
            avg_ret = float(np.mean(stat["returns"]))
            summary.append({
                "strategy": name,
                "total_trades": t,
                "win_rate": round(wr, 1),
                "avg_return": round(avg_ret, 2)
            })

    with open(REPORT_FILE, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"✅ Multi-strategy backtest complete! Results saved to {REPORT_FILE}")

if __name__ == "__main__":
    run_multi_strategy_backtest()
