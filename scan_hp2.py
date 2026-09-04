import os
import io
import json
import time
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "scan_hp2_results.json")

# RANK #16 EXACT PARAMETERS
BASE_W = 40             # 40-day base window
VOL_M = 1.50            # 1.5x 20-day delivery volume SMA
PCT_M = 1.30            # 1.3x 50-day delivery percentage SMA
MIN_DOTS = 4            # >= 4 qualified institutional accumulation dots
MAX_RISK = 8.0          # Stop loss cap <= 8.0%
TARGET_PCT = 15.0       # Book 50% at +15.0%
BE_TRIGGER = 10.0       # Shift stop loss to breakeven at +10.0%
MAX_PE = 35.0
MIN_AVG_VOLUME_9D = 50000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
}

def get_nifty_750_universe():
    local_path = os.path.join(DATA_DIR, "nifty750.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as fp:
                data = json.load(fp)
                if data:
                    return set(data)
        except Exception:
            pass

    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]
    symbols = set()
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
                df.columns = df.columns.str.strip()
                if "Symbol" in df.columns:
                    clean = df["Symbol"].dropna().astype(str).str.strip().str.upper()
                    symbols.update(clean.tolist())
        except Exception:
            pass

    sorted_universe = sorted(list(symbols))
    if sorted_universe:
        try:
            with open(local_path, "w") as fp:
                json.dump(sorted_universe, fp, indent=2)
        except Exception:
            pass
        return set(sorted_universe)
    return set()

def clean_data_fast(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = r.get("time", "")
        if not raw_t:
            continue
        d_str = str(raw_t)[:10]
        c = float(r.get("close", 0) or 0)
        if c <= 0:
            continue
        
        entry = {
            "time": d_str,
            "open": float(r.get("open", c) or c),
            "high": float(r.get("high", c) or c),
            "low": float(r.get("low", c) or c),
            "close": c,
            "delivery_vol": float(r.get("delivery_vol", 0) or 0),
            "volume": float(r.get("volume", 0) or 0),
            "deliv_pct": float(r.get("deliv_pct", 0) or 0)
        }
        if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
            date_map[d_str] = entry

    sorted_dates = sorted(date_map.keys())
    if len(sorted_dates) < 60:
        return []

    clean = [date_map[k] for k in sorted_dates]

    deduped = []
    for r in clean:
        if deduped:
            prev = deduped[-1]
            if (r["open"] == prev["open"] and r["high"] == prev["high"] and 
                r["low"] == prev["low"] and r["close"] == prev["close"]):
                if r["volume"] <= prev["volume"]:
                    continue
                else:
                    deduped.pop()
        deduped.append(r)

    known_multipliers = [2.0, 5.0, 10.0, 1.5, 2.5, 3.0, 4.0]
    for i in range(len(deduped) - 1, 0, -1):
        prev_c = deduped[i - 1]["close"]
        curr_o = deduped[i]["open"]
        if prev_c > 0 and curr_o > 0:
            ratio = prev_c / curr_o
            adj = None
            if ratio >= 1.35:
                for k in known_multipliers:
                    if abs(ratio - k) / k < 0.15:
                        adj = k
                        break
                if not adj and 1.70 <= ratio <= 2.30: adj = 2.0
                elif not adj and 4.30 <= ratio <= 5.50: adj = 5.0
                elif not adj and 8.50 <= ratio <= 11.50: adj = 10.0
            if adj:
                for j in range(0, i):
                    deduped[j]["open"] = round(deduped[j]["open"] / adj, 2)
                    deduped[j]["high"] = round(deduped[j]["high"] / adj, 2)
                    deduped[j]["low"] = round(deduped[j]["low"] / adj, 2)
                    deduped[j]["close"] = round(deduped[j]["close"] / adj, 2)
                    deduped[j]["delivery_vol"] = deduped[j]["delivery_vol"] * adj
                    deduped[j]["volume"] = deduped[j]["volume"] * adj

    running_vol = 50000.0
    for i in range(len(deduped)):
        v = deduped[i]["volume"]
        dv = deduped[i]["delivery_vol"]
        pct = deduped[i]["deliv_pct"]

        if v > 0: running_vol = 0.9 * running_vol + 0.1 * v
        else: deduped[i]["volume"] = running_vol; v = running_vol

        if dv <= 0:
            deduped[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            deduped[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            deduped[i]["delivery_vol"] = v
            deduped[i]["deliv_pct"] = 100.0

    return deduped

def fast_rolling_mean(arr, window):
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    res = np.empty_like(arr, dtype=float)
    res[:window-1] = np.nan
    res[window-1:] = ret[window-1:] / window
    res[:window-1] = res[window-1] if len(res) >= window else 1.0
    return res

def scan_hp2():
    print("🚀 Running Scan HP2 Production Scanner (Rank #16 Configuration)...")
    print(f"⚙️ Settings: Base={BASE_W}d | VolMult={VOL_M}x | DelivMult={PCT_M}x | MinDots={MIN_DOTS} | MaxRisk={MAX_RISK}% | Target={TARGET_PCT}% | BE={BE_TRIGGER}%")

    universe = get_nifty_750_universe()

    fund_path = os.path.join(DATA_DIR, "fundamentals.json")
    fundamentals = {}
    if os.path.exists(fund_path):
        try:
            with open(fund_path, "r") as fp:
                fundamentals = json.load(fp)
        except Exception:
            pass

    stock_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".json") and f not in [
            "fundamentals.json", "screener_results.json", "nifty750.json",
            "NIFTY50.json", "NIFTY.json", "fno_history.json",
            "wyckoff_screener_results.json", "obv_backtest_report.json",
            "scana_vs_absorption_report.json", "scana_candidates.json",
            "optimal_strategies.json", "scana_sensitivity_report.json",
            "scana_optimized_report.json", "scana_combo_winrate_leaderboard.json",
            "scan_hp1_results.json", "scan_hp2_results.json"
        ]
    ]

    if universe and len(universe) > 100:
        stock_files = [f for f in stock_files if f.replace(".json", "").strip().upper() in universe]

    live_candidates = []
    trade_visualizer_cache = {}

    for f_name in stock_files:
        sym = f_name.replace(".json", "").strip().upper()
        json_path = os.path.join(DATA_DIR, f_name)

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
            with open(json_path, "r") as fp:
                raw = json.load(fp)
        except Exception:
            continue

        clean = clean_data_fast(raw)
        if len(clean) < (BASE_W + 15):
            continue

        closes = np.array([r["close"] for r in clean], dtype=float)
        highs = np.array([r["high"] for r in clean], dtype=float)
        lows = np.array([r["low"] for r in clean], dtype=float)
        volumes = np.array([r["volume"] for r in clean], dtype=float)
        pcts = np.array([r["deliv_pct"] for r in clean], dtype=float)
        d_vols = np.array([r["delivery_vol"] for r in clean], dtype=float)
        times = [r["time"] for r in clean]
        N = len(closes)

        # Precompute indicators
        deliv_sma20 = fast_rolling_mean(d_vols, 20)
        pct_sma50 = fast_rolling_mean(pcts, 50)
        vol_sma9 = fast_rolling_mean(volumes, 9)

        # Demat OBV
        obvs = np.zeros(N, dtype=float)
        cur_obv = 0.0
        for idx in range(N):
            dv = d_vols[idx]
            if idx > 0:
                if closes[idx] > closes[idx - 1]: cur_obv += dv
                elif closes[idx] < closes[idx - 1]: cur_obv -= dv
            else:
                cur_obv = dv
            obvs[idx] = cur_obv

        # Institutional accumulation dots (Rank #16: 1.5x Vol, 1.3x Deliv%)
        dots = (pcts >= (PCT_M * pct_sma50)) & (d_vols >= (VOL_M * deliv_sma20))
        dot_indices = [int(x) for x in np.where(dots)[0]]
        dot_cumsum = np.cumsum(dots.astype(int))

        # -----------------------------------------------------------------
        # 1. HISTORICAL TRADES SIMULATION
        # -----------------------------------------------------------------
        symbol_trades = []
        cooldown = 0

        for i in range(BASE_W + 10, N):
            if i < cooldown:
                continue

            if vol_sma9[i] < MIN_AVG_VOLUME_9D:
                continue

            num_dots = dot_cumsum[i - 1] - dot_cumsum[max(0, i - 1 - BASE_W)]
            if num_dots < MIN_DOTS:
                continue

            base_highs = highs[i - BASE_W : i]
            sw_idx = int(np.argmax(base_highs))
            swing_high = base_highs[sw_idx]
            swing_obv = obvs[i - BASE_W + sw_idx]

            if closes[i] > swing_high and closes[i - 1] <= swing_high and obvs[i] > swing_obv:
                entry_p = closes[i]
                lookback_sl = min(12, BASE_W)
                recent_low = float(np.min(lows[i - lookback_sl : i]))
                sl_p = round(recent_low * 0.995, 2)
                risk_pct = round(((entry_p - sl_p) / entry_p) * 100.0, 2)

                if risk_pct <= 0 or risk_pct > MAX_RISK:
                    continue

                fwd_end = min(N, i + 1 + 90)
                f_highs = highs[i + 1 : fwd_end]
                f_lows = lows[i + 1 : fwd_end]
                f_closes = closes[i + 1 : fwd_end]

                max_run = 0.0
                active_sl = sl_p
                booked_partial = False
                be_shifted = False
                partial_booked_idx = None
                exit_p = f_closes[-1] if len(f_closes) > 0 else entry_p
                exit_idx = fwd_end - 1 if len(f_closes) > 0 else i
                is_closed = False

                for bar_idx in range(len(f_highs)):
                    curr_bar = i + 1 + bar_idx
                    gain = ((f_highs[bar_idx] - entry_p) / entry_p) * 100.0
                    if gain > max_run:
                        max_run = gain

                    # Shift SL to BE at +10%
                    if max_run >= BE_TRIGGER and not be_shifted and active_sl < entry_p:
                        active_sl = entry_p
                        be_shifted = True

                    # Book 50% at +15%
                    if max_run >= TARGET_PCT and not booked_partial:
                        booked_partial = True
                        partial_booked_idx = curr_bar
                        active_sl = entry_p

                    # Trail 10-day swing lows on remainder
                    if booked_partial and bar_idx >= 10:
                        t_low = float(np.min(lows[curr_bar - 10 : curr_bar]))
                        if t_low > active_sl:
                            active_sl = t_low

                    if f_lows[bar_idx] <= active_sl:
                        exit_p = min(f_closes[bar_idx], active_sl)
                        exit_idx = curr_bar
                        is_closed = True
                        break

                raw_ret = ((exit_p - entry_p) / entry_p) * 100.0
                realized_ret = round((TARGET_PCT * 0.50) + (raw_ret * 0.50), 2) if booked_partial else round(raw_ret, 2)

                symbol_trades.append({
                    "entry_idx": i,
                    "entry_date": times[i],
                    "entry_price": entry_p,
                    "initial_sl": sl_p,
                    "target_price": round(entry_p * (1.0 + TARGET_PCT / 100.0), 2),
                    "partial_booked_idx": partial_booked_idx,
                    "exit_idx": exit_idx if is_closed else None,
                    "exit_date": times[exit_idx] if is_closed else None,
                    "exit_price": round(exit_p, 2) if is_closed else None,
                    "risk_pct": risk_pct,
                    "realized_ret": realized_ret if is_closed else round(raw_ret, 2),
                    "is_closed": is_closed
                })

                cooldown = exit_idx + 8 if is_closed else N

        if symbol_trades:
            trade_visualizer_cache[sym] = {
                "dots": dot_indices,
                "trades": symbol_trades
            }

        # -----------------------------------------------------------------
        # 2. LIVE TODAY SCREENER EVALUATION (3-Stage Radar)
        # -----------------------------------------------------------------
        curr_idx = N - 1
        num_dots_curr = dot_cumsum[curr_idx] - dot_cumsum[max(0, curr_idx - BASE_W)]
        
        # Candidate requires >= 4 dots in 40-day base
        if num_dots_curr >= MIN_DOTS and vol_sma9[curr_idx] >= MIN_AVG_VOLUME_9D:
            base_highs = highs[curr_idx - BASE_W : curr_idx]
            sw_idx = int(np.argmax(base_highs))
            curr_swing_high = base_highs[sw_idx]
            curr_swing_obv = obvs[curr_idx - BASE_W + sw_idx]

            lookback_sl = min(12, BASE_W)
            recent_low = float(np.min(lows[curr_idx - lookback_sl : curr_idx]))
            sl_price = round(recent_low * 0.995, 2)
            curr_close = closes[curr_idx]
            risk_pct = round(((curr_close - sl_price) / curr_close) * 100.0, 2)

            is_triggered = (curr_close >= curr_swing_high) and (obvs[curr_idx] > curr_swing_obv)
            dist_pct = round(((curr_swing_high - curr_close) / curr_swing_high) * 100.0, 2)

            if is_triggered and (0 < risk_pct <= MAX_RISK):
                status = "🟢 BREAKOUT TRIGGERED"
            elif 0.0 <= dist_pct <= 3.0:
                status = "🟡 READY AT CEILING"
            elif 3.0 < dist_pct <= 12.0:
                status = "🔵 DEEP ACCUMULATION"
            else:
                status = None

            if status:
                live_candidates.append({
                    "Symbol": sym,
                    "Status": status,
                    "LTP": round(curr_close, 2),
                    "Breakout Level": round(curr_swing_high, 2),
                    "Stop Loss": sl_price,
                    "Risk %": f"{risk_pct}%" if risk_pct > 0 else "N/A",
                    "50% Target (+15%)": round(curr_close * 1.15, 2),
                    "Demat Dots": f"{num_dots_curr} dots (≥1.5x Vol, 1.3x Del)",
                    "Distance to Breakout": "0.0%" if is_triggered else f"-{dist_pct}%",
                    "P/E": f"{pe_float:.1f}" if pe_float else "N/A"
                })

    status_weights = {
        "🟢 BREAKOUT TRIGGERED": 0,
        "🟡 READY AT CEILING": 1,
        "🔵 DEEP ACCUMULATION": 2
    }
    live_candidates.sort(key=lambda x: (
        status_weights.get(x["Status"], 99),
        float(x["Distance to Breakout"].replace("%", "").replace("-", ""))
    ))

    payload = {
        "Scan Timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "Rank Config": "Rank #16: Base 40d | ≥4 Dots (1.5x Vol, 1.3x Deliv) | Max SL 8% | Book +15% | BE +10%",
        "Total Live Candidates": len(live_candidates),
        "Candidates": live_candidates,
        "Visualizer Data": trade_visualizer_cache
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"✅ HP2 Scan Complete! Found {len(live_candidates)} candidates across all 3 stages. Saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    scan_hp2()
