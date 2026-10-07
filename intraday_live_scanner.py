import os
import json
import glob
import time
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_JSON = os.path.join(DATA_DIR, "live_intraday_breakouts.json")
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
MIN_TURNOVER_CR = 10.0

os.makedirs(DATA_DIR, exist_ok=True)

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def clean_and_prepare(raw_data):
    if not raw_data or not isinstance(raw_data, list):
        return []
    date_map = {}
    for r in raw_data:
        if not isinstance(r, dict):
            continue
        raw_t = str(r.get("time", "")).strip()
        if not raw_t:
            continue
        d_str = raw_t[:10]
        try:
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
                "deliv_pct": float(r.get("deliv_pct", 0) or 0),
            }
            if d_str not in date_map or entry["volume"] > date_map[d_str]["volume"]:
                date_map[d_str] = entry
        except Exception:
            continue

    clean = [date_map[k] for k in sorted(date_map.keys())]

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


def load_nifty_benchmark():
    for f in ["nifty.json", "nifty50.json", "NIFTY.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) > 50:
                    df = pd.DataFrame(clean)
                    df["perf_60d"] = df["close"].pct_change(60).fillna(0)
                    return {r["time"]: float(r["perf_60d"]) for _, r in df.iterrows()}
            except Exception:
                continue
    return {}


def load_nifty750_symbols():
    if os.path.exists(NIFTY750_FILE):
        try:
            with open(NIFTY750_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                return {str(x).strip().upper() for x in data}
            if isinstance(data, dict):
                return {str(x).strip().upper() for x in data.keys()}
        except Exception:
            pass
    return set()


def extract_breakout_target(clean_data, nifty_perf_map, is_nifty750):
    if len(clean_data) < 60:
        return None

    df = pd.DataFrame(clean_data)
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=5).mean()
    df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
    df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
    df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
    df["perf_60d"] = df["close"].pct_change(60).fillna(0)

    last_i = len(df) - 1
    c = df["close"].values[last_i]
    to_50 = df["turnover_50d"].values[last_i]
    sma200 = df["sma_200"].values[last_i]
    p60 = df["perf_60d"].values[last_i]
    t = df["time"].values[last_i]
    v_sma20 = df["vol_sma20"].values[last_i]

    if to_50 < MIN_TURNOVER_CR:
        return None

    is_stage2 = bool(c >= sma200) if pd.notnull(sma200) else False
    n_p = nifty_perf_map.get(t, 0.001)
    is_rs = bool(p60 > n_p)

    if not (is_stage2 and is_rs):
        return None

    highs = df["high"].values
    lows = df["low"].values

    # 1. Engine 1: V-Reversal Reclaim Level (Flush >= 18%)
    r20_h = highs[last_i - 20:last_i + 1].max()
    pullback = (r20_h - lows[last_i]) / r20_h
    if pullback >= 0.18:
        prior_3d_high = round(float(highs[last_i - 3:last_i + 1].max()), 2)
        recent_trough = round(float(lows[last_i - 4:last_i + 1].min() * 0.995), 2)
        dist_pct = round(((prior_3d_high - c) / c) * 100.0, 2)
        if -2.0 <= dist_pct <= 6.5:
            return {
                "setup": "V-REVERSAL",
                "universe": "NIFTY 750" if is_nifty750 else "NON-N750",
                "pivot_ceiling": prior_3d_high,
                "stop_loss": recent_trough,
                "risk_%": round(((prior_3d_high - recent_trough) / prior_3d_high) * 100, 2),
                "vol_sma20": round(float(v_sma20), 0),
            }

    # 2. Engine 2: Launchpad Base Ceiling (Spread <= 8% for Nifty 750, <= 13% for Non-N750)
    max_allowed_spread = 0.08 if is_nifty750 else 0.13
    for shelf_len in range(10, 19):
        s_h = highs[last_i - shelf_len:last_i + 1].max()
        s_l = lows[last_i - shelf_len:last_i + 1].min()
        if s_l > 0:
            spread = (s_h - s_l) / s_l
            if spread <= max_allowed_spread:
                p_ceil = round(float(s_h), 2)
                p_sl = round(float(s_l), 2)
                dist_pct = round(((p_ceil - c) / c) * 100.0, 2)
                if -2.0 <= dist_pct <= 6.5:
                    return {
                        "setup": "LAUNCHPAD",
                        "universe": "NIFTY 750" if is_nifty750 else "NON-N750",
                        "pivot_ceiling": p_ceil,
                        "stop_loss": p_sl,
                        "risk_%": round(((p_ceil - p_sl) / p_ceil) * 100, 2),
                        "vol_sma20": round(float(v_sma20), 0),
                    }
                break

    return None


def get_nse_session():
    cookie_req = urllib.request.Request("https://www.nseindia.com", headers=NSE_HEADERS)
    try:
        resp = urllib.request.urlopen(cookie_req, timeout=10)
        return resp.headers.get("Set-Cookie", "")
    except Exception:
        return ""


def fetch_live_quote(symbol, cookies):
    sym_encoded = urllib.parse.quote(symbol.strip().upper())
    url = f"https://www.nseindia.com/api/quote-equity?symbol={sym_encoded}"
    headers = dict(NSE_HEADERS)
    if cookies:
        headers["Cookie"] = cookies

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-
