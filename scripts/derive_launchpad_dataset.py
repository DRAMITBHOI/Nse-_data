import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "launchpad_dataset.csv")


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

    # Split / bonus corporate action adjustments
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

    running_vol = 50000.0
    for i in range(len(clean)):
        v = clean[i]["volume"]
        dv = clean[i]["delivery_vol"]
        pct = clean[i]["deliv_pct"]
        if v > 0:
            running_vol = 0.9 * running_vol + 0.1 * v
        else:
            clean[i]["volume"] = running_vol
            v = running_vol
        if dv <= 0:
            clean[i]["delivery_vol"] = v * (pct / 100.0 if pct > 0 else 0.50)
            clean[i]["deliv_pct"] = pct if pct > 0 else 50.0
        elif dv > v:
            clean[i]["delivery_vol"] = v
            clean[i]["deliv_pct"] = 100.0

    return clean


def load_nifty_benchmark():
    for f in ["nifty.json", "nifty50.json", "NIFTY.json", "NIFTY50.json"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                clean = clean_and_prepare(raw)
                if len(clean) > 50:
                    df = pd.DataFrame(clean)
                    df["perf_60d"] = df["close"].pct_change(60).fillna(0)
                    return {str(r["time"])[:10]: float(r["perf_60d"]) for _, r in df.iterrows()}
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


def extract_launchpad_dataset():
    nifty_map = load_nifty_benchmark()
    nifty_dates = sorted(nifty_map.keys())
    n750_set = load_nifty750_symbols()

    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    excluded = {
        "nifty750.json", "nifty50.json", "nifty.json", "NIFTY.json", "NIFTY50.json",
        "fundamentals.json", "backtest_results.json", "swing_ultra750_results.json",
        "swing_ultraN750_results.json", "live_intraday_breakouts.json",
        "live_intraday_breakouts_v2.json", "accuracy_audit_log.json", "my_trades.json"
    }

    records = []
    print(f"🔬 Extracting Launchpad Breakout Dataset across {len(json_files)} symbols...")

    for path in sorted(json_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym in excluded:
            continue

        is_n750 = 1 if (n750_set and sym in n750_set) else 0

        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if len(clean) < 100:
                continue

            df = pd.DataFrame(clean)
            df["vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
            df["perf_60d"] = df["close"].pct_change(60).fillna(0)

            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            # Demat Cumulative OBV
            p_diff = df["close"].diff()
            direction = np.where(p_diff > 0, 1.0, np.where(p_diff < 0, -1.0, 0.0))
            df["deliv_obv"] = (direction * df["delivery_vol"]).cumsum()
            df["dobv_max20"] = df["deliv_obv"].rolling(20, min_periods=5).max()

            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            opens = df["open"].values
            volumes = df["volume"].values
            vol_avgs = df["vol_sma20"].values
            deliv_vols = df["delivery_vol"].values
            deliv_avgs = df["deliv_sma20"].values
            to_50d = df["turnover_50d"].values
            rcs = df["range_closeness"].values
            ema20s = df["ema_20"].values
            sma200s = df["sma_200"].values
            perf_60s = df["perf_60d"].values
            dobvs = df["deliv_obv"].values
            dobv_maxes = df["dobv_max20"].values
            times = df["time"].values

            # Scan historical bars for coiling shelves
            for i in range(40, len(df) - 45):
                t = str(times[i])[:10]
                c = closes[i]
                v = volumes[i]
                v_avg = vol_avgs[i]
                dv = deliv_vols[i]
                dv_avg = deliv_avgs[i]
                to = to_50d[i]
                sma200 = sma200s[i]

                # Relaxed pre-filter: allow setups down to ₹2 Cr turnover to test liquidity floors later
                if to < 2.0 or (pd.notnull(sma200) and c < sma200 * 0.92):
                    continue

                # 1. Base Shelf Detection (10 to 18 sessions)
                shelf_found = False
                lp_h = 0.0
                lp_l = 0.0
                spread = 0.0
                max_allowed_spread = 18.0

                for s_len in range(10, 19):
                    s_h = highs[max(0, i - s_len):i].max()
                    s_l = lows[max(0, i - s_len):i].min()
                    if s_l <= 0:
                        continue
                    sp = ((s_h - s_l) / s_l) * 100.0
                    if sp <= max_allowed_spread:
                        shelf_found = True
                        lp_h = s_h
                        lp_l = s_l
                        spread = sp
                        break

                if not shelf_found:
                    continue

                # 2. Breakout Trigger
                c_bo = (c >= lp_h) and (closes[i - 1] < lp_h)
                if not c_bo:
                    continue

                # 3. VDU (Volume Dry-Up) in Base
                min_5d_vol = pd.Series(volumes[max(0, i - 15):i]).rolling(5, min_periods=1).mean().min()
                vdu_ratio = round(min_5d_vol / v_avg, 2) if v_avg > 0 else 1.0

                # 4. Demat Accumulation Footprint
                up_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] >= closes[k - 1])
                dn_deliv = sum(deliv_vols[k] for k in range(max(0, i - 15), i) if closes[k] < closes[k - 1])
                cumul_deliv_ratio = round((up_deliv / dn_deliv), 2) if dn_deliv > 0 else 1.50
                dobv_breakout = 1 if (i > 0 and dobvs[i] >= dobv_maxes[i - 1]) else 0

                vol_surge = round(v / v_avg, 2) if v_avg > 0 else 1.0
                deliv_surge = round(dv / dv_avg, 2) if dv_avg > 0 else 1.0
                rc = round(rcs[i], 2)

                # Relative Strength
                n_perf = nifty_map.get(t)
                if n_perf is None:
                    p_dates = [d for d in nifty_dates if d <= t]
                    n_perf = nifty_map[p_dates[-1]] if p_dates else 0.001
                rs_stock_nifty = round(perf_60s[i] - n_perf, 3)
                conv_score = round(deliv_surge * max(0.1, 1.0 + rs_stock_nifty), 3)

                entry_price = round(lp_h, 2)
                sl_price = round(lp_l * 0.995, 2)
                r_dist = entry_price - sl_price
                if r_dist <= 0 or (r_dist / entry_price) > 0.16:
                    continue
                risk_pct = round((r_dist / entry_price) * 100.0, 2)

                # 5. Forward Simulation (45 Sessions)
                fwd_highs = highs[i + 1:i + 46]
                fwd_lows = lows[i + 1:i + 46]
                fwd_closes = closes[i + 1:i + 46]
                fwd_opens = opens[i + 1:i + 46]
                fwd_ema20 = ema20s[i + 1:i + 46]
                fwd_vols = volumes[i + 1:i + 46]
                fwd_v_avgs = vol_avgs[i + 1:i + 46]
                fwd_dv = deliv_vols[i + 1:i + 46]
                fwd_dv_avgs = deliv_avgs[i + 1:i + 46]
                fwd_rc = rcs[i + 1:i + 46]

                mfe_pct = 0.0
                mae_pct = 0.0
                bar_of_max_gain = 0

                for day_idx in range(len(fwd_closes)):
                    cur_h = fwd_highs[day_idx]
                    cur_l = fwd_lows[day_idx]
                    cur_g = ((cur_h - entry_price) / entry_price) * 100.0
                    cur_d = ((cur_l - entry_price) / entry_price) * 100.0
                    if cur_g > mfe_pct:
                        mfe_pct = cur_g
                        bar_of_max_gain = day_idx + 1
                    if cur_d < mae_pct:
                        mae_pct = cur_d
                    if cur_l <= sl_price:
                        break

                # Exit 2.2 Model
                pnl_exit22 = 0.0
                peak_g = 0.0
                vdu_under = 0
                for d in range(len(fwd_closes)):
                    co, ch, cl, cc = fwd_opens[d], fwd_highs[d], fwd_lows[d], fwd_closes[d]
                    c_gain = ((cc - entry_price) / entry_price) * 100.0
                    peak_g = max(peak_g, c_gain)

                    if co < sl_price:
                        pnl_exit22 = ((co - entry_price) / entry_price) * 100.0
                        break
                    elif cl <= sl_price:
                        pnl_exit22 = ((sl_price - entry_price) / entry_price) * 100.0
                        break
                    elif (fwd_dv[d] >= 2.0 * fwd_dv_avgs[d]) and (fwd_rc[d] <= 0.35) and (peak_g >= 20.0):
                        pnl_exit22 = c_gain
                        break
                    elif cc < fwd_ema20[d]:
                        if (fwd_vols[d] >= 1.25 * fwd_v_avgs[d]) or (fwd_dv[d] >= 1.25 * fwd_dv_avgs[d]):
                            pnl_exit22 = c_gain
                            break
                        else:
                            vdu_under += 1
                            if vdu_under > 5:
                                pnl_exit22 = c_gain
                                break
                    if d == len(fwd_closes) - 1:
                        pnl_exit22 = c_gain

                records.append({
                    "symbol": sym,
                    "is_n750": is_n750,
                    "trigger_date": t,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "risk_pct": risk_pct,
                    "base_spread_pct": round(spread, 2),
                    "vdu_ratio": vdu_ratio,
                    "cumul_deliv_ratio": cumul_deliv_ratio,
                    "dobv_breakout": dobv_breakout,
                    "vol_surge": vol_surge,
                    "deliv_surge": deliv_surge,
                    "range_closeness": rc,
                    "conv_score": conv_score,
                    "turnover_50d": round(to, 2),
                    "mfe_pct": round(mfe_pct, 2),
                    "mae_pct": round(mae_pct, 2),
                    "bar_of_max_gain": bar_of_max_gain,
                    "pnl_exit22": round(pnl_exit22, 2)
                })
        except Exception:
            continue

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Launchpad Extraction Complete! Saved {len(df_out):,} setups to {OUTPUT_CSV}")
    if "is_n750" in df_out.columns:
        print(f"   • N750 setups: {(df_out['is_n750'] == 1).sum():,}")
        print(f"   • Non-N750 setups: {(df_out['is_n750'] == 0).sum():,}")


if __name__ == "__main__":
    extract_launchpad_dataset()
