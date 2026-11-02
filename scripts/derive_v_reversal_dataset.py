import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "v_reversal_dataset.csv")


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
                    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
                    res = {}
                    for _, r in df.iterrows():
                        res[str(r["time"])[:10]] = {
                            "perf_60d": float(r["perf_60d"]),
                            "above_50ema": float(r["close"]) >= float(r["ema_50"])
                        }
                    return res
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


def extract_v_reversals():
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
    print(f"🔬 Extracting candidate V-Reversals across {len(json_files)} files...")

    for path in sorted(json_files):
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        if sym in excluded or (n750_set and sym not in n750_set):
            continue

        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            clean = clean_and_prepare(raw)
            if len(clean) < 100:
                continue

            df = pd.DataFrame(clean)
            df["gross_vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
            df["deliv_sma20"] = df["delivery_vol"].rolling(20, min_periods=1).mean()
            df["turnover_cr"] = (df["close"] * df["volume"]) / 1e7
            df["turnover_50d"] = df["turnover_cr"].rolling(50, min_periods=10).mean().fillna(0)
            df["deliv_pct_50d"] = df["deliv_pct"].rolling(50, min_periods=10).mean().fillna(35.0)
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["sma_200"] = df["close"].rolling(200, min_periods=50).mean()
            df["perf_60d"] = df["close"].pct_change(60).fillna(0)

            # ATR 14
            prev_close = df["close"].shift(1)
            tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(14, min_periods=1).mean()

            c_range = df["high"] - df["low"]
            df["range_closeness"] = np.where(c_range > 0, (df["close"] - df["low"]) / c_range, 0.50)

            # 12-day Distribution Count
            is_dist_day = (
                (df["close"] < df["close"].shift(1)) &
                ((df["delivery_vol"] >= 1.35 * df["deliv_sma20"]) | (df["deliv_pct"] >= 1.15 * df["deliv_pct_50d"])) &
                (df["range_closeness"] <= 0.45)
            ).astype(int)
            df["dist_count_12d"] = is_dist_day.rolling(12, min_periods=1).sum()

            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            opens = df["open"].values
            volumes = df["volume"].values
            vol_avgs = df["gross_vol_sma20"].values
            deliv_vols = df["delivery_vol"].values
            deliv_avgs = df["deliv_sma20"].values
            deliv_pcts = df["deliv_pct"].values
            deliv_pct_avgs = df["deliv_pct_50d"].values
            to_50d = df["turnover_50d"].values
            range_closes = df["range_closeness"].values
            ema20s = df["ema_20"].values
            sma200s = df["sma_200"].values
            perf_60s = df["perf_60d"].values
            dist_12ds = df["dist_count_12d"].values
            times = df["time"].values

            for i in range(50, len(df) - 45):
                t = str(times[i])[:10]
                c = closes[i]
                h = highs[i]
                l = lows[i]
                v = volumes[i]
                v_avg = vol_avgs[i]
                dv = deliv_vols[i]
                dv_avg = deliv_avgs[i]
                sma200 = sma200s[i]

                # Relaxed pre-filter: Minimum turnover >= ₹5 Cr and above/near 200 SMA
                if to_50d[i] < 5.0 or (pd.notnull(sma200) and c < sma200 * 0.90):
                    continue

                # --- 1. CAPITULATION DEPTH CHECK ---
                recent_20_h = highs[max(0, i - 20):i].max()
                if recent_20_h <= 0:
                    continue
                drop_depth_pct = ((recent_20_h - l) / recent_20_h) * 100.0

                # Capture candidate drops of >= 9% (relaxed from 18%)
                if drop_depth_pct < 9.0:
                    continue

                # --- 2. MULTI-HORIZON SWING RECLAIM CHECKS ---
                prior_2d_h = highs[max(0, i - 2):i].max()
                prior_3d_h = highs[max(0, i - 4):i].max()
                prior_5d_h = highs[max(0, i - 6):i].max()

                reclaims_2d = int(c >= prior_2d_h and c > closes[i - 1])
                reclaims_3d = int(c >= prior_3d_h and c > closes[i - 1])
                reclaims_5d = int(c >= prior_5d_h and c > closes[i - 1])

                # Require at least the 2-day or 3-day swing high to be reclaimed
                if not (reclaims_2d or reclaims_3d):
                    continue

                # Standard entry reference: prior 3-day high (or close if higher)
                entry_price = round(max(prior_3d_h, closes[i - 1]), 2)
                
                # Compare 3 Stop Loss Candidates:
                sl_5d = round(lows[max(0, i - 5):i + 1].min() * 0.995, 2)
                sl_3d = round(lows[max(0, i - 3):i + 1].min() * 0.995, 2)
                sl_bar = round(l * 0.995, 2)

                # Primary test uses 5-day structural low
                sl_price = sl_5d
                r_dist = entry_price - sl_price
                if r_dist <= 0 or (r_dist / entry_price) > 0.16:
                    continue
                risk_pct = round((r_dist / entry_price) * 100.0, 2)

                # --- 3. VOLUME & DEMAT FOOTPRINT ---
                vol_surge = round(v / v_avg, 2) if v_avg > 0 else 1.0
                deliv_vol_surge = round(dv / dv_avg, 2) if dv_avg > 0 else 1.0
                deliv_pct_surge = round(deliv_pcts[i] / deliv_pct_avgs[i], 2) if deliv_pct_avgs[i] > 0 else 1.0

                # Capitulation Low Signature (volume ratio at the lowest bar in drop)
                drop_low_idx = max(0, i - 5) + int(np.argmin(lows[max(0, i - 5):i + 1]))
                washout_vol_ratio = round(volumes[drop_low_idx] / vol_avgs[drop_low_idx], 2) if vol_avgs[drop_low_idx] > 0 else 1.0

                # Relative Strength vs Nifty
                n_info = nifty_map.get(t)
                if not n_info:
                    p_dates = [d for d in nifty_dates if d <= t]
                    n_info = nifty_map[p_dates[-1]] if p_dates else {"perf_60d": 0.001, "above_50ema": True}
                rs_stock_nifty = round(perf_60s[i] - n_info["perf_60d"], 3)
                conv_score = round(deliv_vol_surge * max(0.1, 1.0 + rs_stock_nifty), 3)

                # --- 4. FORWARD EXECUTION SIMULATION (UP TO 45 SESSIONS) ---
                fwd_highs = highs[i + 1: i + 46]
                fwd_lows = lows[i + 1: i + 46]
                fwd_closes = closes[i + 1: i + 46]
                fwd_opens = opens[i + 1: i + 46]
                fwd_ema20 = ema20s[i + 1: i + 46]
                fwd_vols = volumes[i + 1: i + 46]
                fwd_v_avgs = vol_avgs[i + 1: i + 46]
                fwd_dv = deliv_vols[i + 1: i + 46]
                fwd_dv_avgs = deliv_avgs[i + 1: i + 46]
                fwd_rc = range_closes[i + 1: i + 46]
                fwd_dist12 = dist_12ds[i + 1: i + 46]

                mfe_pct = 0.0
                mae_pct = 0.0
                bar_of_max_gain = 0
                hit_target_6pct_in_15 = 0
                stopped_out_before_target = 0

                target_6pct = entry_price * 1.06
                target_2r = entry_price + (2.0 * r_dist)

                for day_idx in range(len(fwd_closes)):
                    cur_o = fwd_opens[day_idx]
                    cur_h = fwd_highs[day_idx]
                    cur_l = fwd_lows[day_idx]

                    cur_gain = ((cur_h - entry_price) / entry_price) * 100.0
                    cur_drop = ((cur_l - entry_price) / entry_price) * 100.0
                    if cur_gain > mfe_pct:
                        mfe_pct = cur_gain
                        bar_of_max_gain = day_idx + 1
                    if cur_drop < mae_pct:
                        mae_pct = cur_drop

                    # Stop breach: overnight gap or intraday low touch
                    if cur_o < sl_price or cur_l <= sl_price:
                        if day_idx < 15 and hit_target_6pct_in_15 == 0:
                            stopped_out_before_target = 1
                        break

                    if day_idx < 15 and cur_h >= target_6pct and stopped_out_before_target == 0:
                        hit_target_6pct_in_15 = 1

                is_successful = 1 if (hit_target_6pct_in_15 == 1 and stopped_out_before_target == 0) else 0

                # Simulation: Exit 2.2 Model
                pnl_exit22 = 0.0
                peak_g = 0.0
                p_close = entry_price
                vdu_under = 0
                for d in range(len(fwd_closes)):
                    co, ch, cl, cc = fwd_opens[d], fwd_highs[d], fwd_lows[d], fwd_closes[d]
                    p_close = max(p_close, cc)
                    c_gain = ((cc - entry_price) / entry_price) * 100.0
                    peak_g = max(peak_g, c_gain)

                    if co < sl_price:
                        pnl_exit22 = ((co - entry_price) / entry_price) * 100.0; break
                    elif cl <= sl_price:
                        pnl_exit22 = ((sl_price - entry_price) / entry_price) * 100.0; break
                    elif (fwd_dv[d] >= 2.0 * fwd_dv_avgs[d]) and (fwd_rc[d] <= 0.35) and (peak_g >= 20.0):
                        pnl_exit22 = c_gain; break
                    elif (fwd_dist12[d] >= 3) and (cc < fwd_ema20[d]):
                        pnl_exit22 = c_gain; break
                    elif peak_g >= 50.0 and cc < max(fwd_ema20[d], p_close * 0.85):
                        pnl_exit22 = c_gain; break
                    elif cc < fwd_ema20[d]:
                        if (fwd_vols[d] >= 1.25 * fwd_v_avgs[d]) or (fwd_dv[d] >= 1.25 * fwd_dv_avgs[d]):
                            pnl_exit22 = c_gain; break
                        else:
                            vdu_under += 1
                            if vdu_under > 5:
                                pnl_exit22 = c_gain; break
                    if d == len(fwd_closes) - 1:
                        pnl_exit22 = c_gain

                # Simulation: Staged 2R Model
                pnl_staged = 0.0
                active_sl = sl_price
                t1_hit = False
                for d in range(len(fwd_closes)):
                    co, ch, cl, cc = fwd_opens[d], fwd_highs[d], fwd_lows[d], fwd_closes[d]
                    if not t1_hit and ch >= target_2r:
                        t1_hit = True
                        active_sl = entry_price * 1.003
                    if co < active_sl:
                        fill = co
                        ret = ((fill - entry_price) / entry_price) * 100.0
                        pnl_staged = (0.5 * ((target_2r - entry_price) / entry_price) * 100.0 + 0.5 * ret) if t1_hit else ret
                        break
                    elif cl <= active_sl:
                        fill = active_sl
                        ret = ((fill - entry_price) / entry_price) * 100.0
                        pnl_staged = (0.5 * ((target_2r - entry_price) / entry_price) * 100.0 + 0.5 * ret) if t1_hit else ret
                        break
                    elif t1_hit and cc < fwd_ema20[d]:
                        ret = ((cc - entry_price) / entry_price) * 100.0
                        pnl_staged = (0.5 * ((target_2r - entry_price) / entry_price) * 100.0 + 0.5 * ret)
                        break
                    if d == len(fwd_closes) - 1:
                        ret = ((cc - entry_price) / entry_price) * 100.0
                        pnl_staged = (0.5 * ((target_2r - entry_price) / entry_price) * 100.0 + 0.5 * ret) if t1_hit else ret

                records.append({
                    "symbol": sym,
                    "trigger_date": t,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "risk_pct": risk_pct,
                    "drop_depth_pct": round(drop_depth_pct, 2),
                    "reclaims_2d": reclaims_2d,
                    "reclaims_3d": reclaims_3d,
                    "reclaims_5d": reclaims_5d,
                    "vol_surge": vol_surge,
                    "deliv_vol_surge": deliv_vol_surge,
                    "deliv_pct_surge": deliv_pct_surge,
                    "washout_vol_ratio": washout_vol_ratio,
                    "range_closeness": round(range_closes[i], 2),
                    "rs_stock_nifty": rs_stock_nifty,
                    "conv_score": conv_score,
                    "mfe_pct": round(mfe_pct, 2),
                    "mae_pct": round(mae_pct, 2),
                    "bar_of_max_gain": bar_of_max_gain,
                    "is_successful": is_successful,
                    "pnl_exit22": round(pnl_exit22, 2),
                    "pnl_staged_2r": round(pnl_staged, 2)
                })
        except Exception:
            continue

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ V-Reversal Extraction Complete! Saved {len(df_out)} candidate setups to {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_v_reversals()
