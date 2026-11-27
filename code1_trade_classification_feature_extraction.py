
import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
START_DATE = "2021-01-01"
MIN_ROWS = 260
MIN_DELIVERY_COVERAGE = 0.80
H25 = 25
H60 = 60
MIN_RISK_PCT = 0.01
MAX_RISK_PCT = 0.15

EXCLUDED_FILES = {
    "nifty750.json", "nifty50.json", "nifty.json",
    "fundamentals.json", "accuracy_audit_log.json"
}
EXCLUDED_TOKENS = (
    "report", "results", "scanner", "watchlist", "leaderboard",
    "sweep", "audit", "backtest", "champion", "history", "nifty"
)


def to_num(value):
    if value is None or str(value).strip() == "":
        return np.nan
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def is_stock_json(path):
    name = os.path.basename(path).lower()
    return (
        name.endswith(".json")
        and name not in EXCLUDED_FILES
        and not any(token in name for token in EXCLUDED_TOKENS)
    )


def load_nifty750_symbols():
    if not os.path.exists(NIFTY750_FILE):
        print("WARNING: nifty750.json not found.")
        return set()

    with open(NIFTY750_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("symbols"), list):
        items = raw["symbols"]
    elif isinstance(raw, dict) and isinstance(raw.get("constituents"), list):
        items = raw["constituents"]
    elif isinstance(raw, dict):
        items = list(raw.keys())
    else:
        raise RuntimeError("Unsupported nifty750.json format.")

    symbols = {
        str(x).strip().upper()
        for x in items if str(x).strip()
    }

    print(f"Current Nifty 750 symbols loaded: {len(symbols)}")

    if len(symbols) < 400:
        print("WARNING: Fewer than 400 symbols found. Check nifty750.json.")

    return symbols


def row_quality(row):
    return (
        int(row["delivery_observed"]),
        int(np.isfinite(row["deliv_pct_raw"])),
        int(
            np.isfinite(row["delivery_vol"])
            and row["delivery_vol"] > 0
        ),
        row["volume"],
    )


def clean_stock(raw, symbol, stock_audit, duplicate_audit, split_audit):
    if not isinstance(raw, list):
        stock_audit.append({
            "symbol": symbol,
            "status": "EXCLUDED_NOT_LIST",
            "rows": 0
        })
        return None

    by_date = {}
    malformed_delivery = 0
    invalid_ohlcv = 0

    for rec in raw:
        if not isinstance(rec, dict):
            continue

        date = str(rec.get("time", ""))[:10]

        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue

        vals = [
            to_num(rec.get(k))
            for k in ("open", "high", "low", "close", "volume")
        ]

        if not np.all(np.isfinite(vals)):
            invalid_ohlcv += 1
            continue

        op, high, low, close, volume = vals

        if (
            min(vals) <= 0
            or high < max(op, close) - 0.01
            or low > min(op, close) + 0.01
            or high < low
        ):
            invalid_ohlcv += 1
            continue

        delivery_vol = to_num(rec.get("delivery_vol"))
        delivery_pct = to_num(rec.get("deliv_pct"))

        delivery_observed = int(
            np.isfinite(delivery_vol) and delivery_vol >= 0
        )

        if rec.get("delivery_vol") not in (None, "") and not delivery_observed:
            malformed_delivery += 1

        if not np.isfinite(delivery_pct) or not 0 <= delivery_pct <= 100:
            if rec.get("deliv_pct") not in (None, ""):
                malformed_delivery += 1
            delivery_pct = np.nan

        row = {
            "date": date,
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "delivery_vol": delivery_vol if delivery_observed else np.nan,
            "deliv_pct_raw": delivery_pct,
            "delivery_observed": delivery_observed,
        }

        if date not in by_date:
            by_date[date] = row
        else:
            old = by_date[date]

            ohlc_diff = max(
                abs(old[k] - row[k])
                for k in ("open", "high", "low", "close")
            )

            differs = (
                ohlc_diff > 0.05
                or old["volume"] != row["volume"]
                or not (
                    (
                        np.isnan(old["delivery_vol"])
                        and np.isnan(row["delivery_vol"])
                    )
                    or old["delivery_vol"] == row["delivery_vol"]
                )
            )

            if differs:
                keep, drop = (
                    (row, old)
                    if row_quality(row) >= row_quality(old)
                    else (old, row)
                )

                duplicate_audit.append({
                    "symbol": symbol,
                    "date": date,
                    "ohlc_max_diff": round(ohlc_diff, 4),
                    "kept_volume": keep["volume"],
                    "dropped_volume": drop["volume"],
                    "kept_delivery": keep["delivery_vol"],
                    "dropped_delivery": drop["delivery_vol"],
                })

                by_date[date] = keep

    rows = [by_date[d] for d in sorted(by_date)]

    if len(rows) < MIN_ROWS:
        stock_audit.append({
            "symbol": symbol,
            "status": "EXCLUDED_TOO_FEW_ROWS",
            "rows": len(rows),
            "malformed_delivery_fields": malformed_delivery,
            "invalid_ohlcv_rows": invalid_ohlcv,
        })
        return None

    # Corporate-action heuristic.
    # All suspected adjustments are recorded for manual audit.
    multiples = [2.0, 3.0, 4.0, 5.0, 10.0]

    for i in range(1, len(rows)):
        prev_close = rows[i - 1]["close"]
        curr_open = rows[i]["open"]

        ratio = prev_close / curr_open if curr_open > 0 else 0

        vol_jump = (
            rows[i]["volume"] / rows[i - 1]["volume"]
            if rows[i - 1]["volume"] > 0 else 1
        )

        adj = next(
            (
                m for m in multiples
                if ratio >= 1.75
                and vol_jump >= 1.25
                and abs(ratio - m) / m <= 0.12
            ),
            None
        )

        if adj:
            split_audit.append({
                "symbol": symbol,
                "date": rows[i]["date"],
                "previous_close": prev_close,
                "current_open": curr_open,
                "ratio": ratio,
                "volume_jump": vol_jump,
                "adjustment_multiple": adj,
            })

            for j in range(i):
                for key in ("open", "high", "low", "close"):
                    rows[j][key] /= adj

                rows[j]["volume"] *= adj

                if rows[j]["delivery_observed"]:
                    rows[j]["delivery_vol"] *= adj

    df = pd.DataFrame(rows)

    df["delivery_vol_clean"] = np.where(
        df.delivery_observed == 1,
        np.minimum(df.delivery_vol, df.volume),
        np.nan
    )

    df["delivery_pct_clean"] = np.where(
        df.delivery_observed == 1,
        df.delivery_vol_clean / df.volume * 100,
        np.nan
    )

    positive_delivery = int(
        (
            (df.delivery_observed == 1)
            & (df.delivery_vol_clean > 0)
        ).sum()
    )

    stock_audit.append({
        "symbol": symbol,
        "status": "INCLUDED",
        "rows": len(df),
        "first_date": df.date.iloc[0],
        "last_date": df.date.iloc[-1],
        "delivery_observed_days": int(df.delivery_observed.sum()),
        "delivery_positive_days": positive_delivery,
        "delivery_coverage_pct": round(
            df.delivery_observed.mean() * 100, 2
        ),
        "delivery_gt_volume_rows": int(
            (
                (df.delivery_observed == 1)
                & (df.delivery_vol > df.volume)
            ).sum()
        ),
        "malformed_delivery_fields": malformed_delivery,
        "invalid_ohlcv_rows": invalid_ohlcv,
    })

    return df


def add_indicators(df):
    x = df.copy()

    # Prior-session volume averages.
    x["vol_sma20_prior"] = (
        x.volume.shift(1).rolling(20, min_periods=20).mean()
    )

    obs20 = (
        x.delivery_observed.shift(1)
        .rolling(20, min_periods=20).mean()
    )

    x["delivery_coverage20_prior"] = obs20

    x["delivery_sma20_prior"] = (
        x.delivery_vol_clean.shift(1)
        .rolling(20, min_periods=1).mean()
        .where(obs20 >= MIN_DELIVERY_COVERAGE)
    )

    dp_obs50 = (
        x.delivery_pct_clean.notna().astype(int)
        .shift(1)
        .rolling(50, min_periods=50).mean()
    )

    prior_dp = x.delivery_pct_clean.shift(1)

    x["delivery_coverage50_prior"] = dp_obs50

    x["delivery_pct_mean50_prior"] = (
        prior_dp.rolling(50, min_periods=1)
        .mean()
        .where(dp_obs50 >= MIN_DELIVERY_COVERAGE)
    )

    x["delivery_pct_std50_prior"] = (
        prior_dp.rolling(50, min_periods=1)
        .std()
        .where(dp_obs50 >= MIN_DELIVERY_COVERAGE)
        .replace(0, np.nan)
    )

    x["delivery_pct_z"] = (
        (x.delivery_pct_clean - x.delivery_pct_mean50_prior)
        / x.delivery_pct_std50_prior
    )

    x["turnover_cr"] = x.close * x.volume / 1e7

    x["turnover50_prior_cr"] = (
        x.turnover_cr.shift(1)
        .rolling(50, min_periods=50).mean()
    )

    x["sma50"] = x.close.rolling(50, min_periods=50).mean()
    x["sma200"] = x.close.rolling(200, min_periods=200).mean()

    x["high252_prior"] = (
        x.high.shift(1).rolling(252, min_periods=252).max()
    )

    x["perf30"] = x.close.pct_change(30)
    x["perf60"] = x.close.pct_change(60)

    prev_close = x.close.shift(1)

    tr = pd.concat([
        x.high - x.low,
        (x.high - prev_close).abs(),
        (x.low - prev_close).abs()
    ], axis=1).max(axis=1)

    x["atr14_prior"] = (
        tr.shift(1).rolling(14, min_periods=14).mean()
    )

    x["atr5_prior"] = (
        tr.shift(1).rolling(5, min_periods=5).mean()
    )

    day_range = x.high - x.low

    x["close_range_pct"] = np.where(
        day_range > 0,
        (x.close - x.low) / day_range,
        np.nan
    )

    x["candle_body"] = (x.close - x.open).abs()

    x["upper_wick"] = (
        x.high - x[["open", "close"]].max(axis=1)
    )

    x["lower_wick"] = (
        x[["open", "close"]].min(axis=1) - x.low
    )

    deliv_surge = (
        x.delivery_vol_clean
        / x.delivery_sma20_prior.replace(0, np.nan)
    )

    abs_return_pct = (
        x.close.pct_change().abs().mul(100).clip(lower=0.25)
    )

    x["daily_absorption"] = deliv_surge / abs_return_pct

    direction = np.sign(x.close.diff())

    x["delivery_obv"] = (
        direction * x.delivery_vol_clean.fillna(0)
    ).cumsum()

    return x


def load_nifty_benchmark():
    for filename in ("nifty.json", "nifty50.json"):
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            continue

        values = {}

        for rec in raw:
            if isinstance(rec, dict) and rec.get("time"):
                close = to_num(rec.get("close"))

                if np.isfinite(close) and close > 0:
                    values[str(rec["time"])[:10]] = close

        if len(values) >= 100:
            s = pd.Series(values, dtype=float).sort_index()

            result = pd.DataFrame({"nifty_close": s})

            result["nifty_perf60"] = (
                result.nifty_close.pct_change(60)
            )

            result["nifty_sma50"] = (
                result.nifty_close.rolling(50, min_periods=50).mean()
            )

            result["nifty_above_sma50"] = (
                result.nifty_close >= result.nifty_sma50
            ).astype(float)

            return result

    print(
        "WARNING: Nifty benchmark unavailable. "
        "RS and Nifty regime fields will be blank."
    )

    return pd.DataFrame(columns=[
        "nifty_close",
        "nifty_perf60",
        "nifty_sma50",
        "nifty_above_sma50"
    ])


def eval_trade(highs, lows, closes, dates, entry_idx, entry, stop):
    """Evaluate exact-price entry and fixed +1.5R/-1R levels."""

    if entry_idx < 0 or entry_idx >= len(closes):
        return {
            "outcome_25d": "NO_ENTRY_BAR",
            "realized_r": np.nan
        }

    if not (
        np.isfinite(entry)
        and np.isfinite(stop)
        and entry > stop > 0
    ):
        return {
            "outcome_25d": "INVALID_RISK",
            "realized_r": np.nan
        }

    risk = entry - stop
    risk_pct = risk / entry
    target = entry + 1.5 * risk

    base = {
        "entry": entry,
        "stop": stop,
        "target_1_5r": target,
        "risk_per_share": risk,
        "risk_pct": risk_pct,
        "entry_date": dates[entry_idx],
    }

    if not MIN_RISK_PCT <= risk_pct <= MAX_RISK_PCT:
        return {
            **base,
            "outcome_25d": "INVALID_RISK",
            "realized_r": np.nan
        }

    if not lows[entry_idx] <= entry <= highs[entry_idx]:
        return {
            **base,
            "outcome_25d": "UNFILLED_ENTRY",
            "realized_r": np.nan
        }

    # Daily OHLC cannot establish intraday order on entry day.
    if lows[entry_idx] <= stop or highs[entry_idx] >= target:
        return {
            **base,
            "outcome_25d": "AMBIGUOUS_ENTRY_DAY",
            "realized_r": np.nan,
            "exit_date": dates[entry_idx],
            "sessions_to_exit": 0,
            "mfe_r": np.nan,
            "mae_r": np.nan
        }

    start = entry_idx + 1

    if start + H25 > len(closes):
        return {
            **base,
            "outcome_25d": "INCOMPLETE_25D",
            "realized_r": np.nan,
            "exit_date": None,
            "sessions_to_exit": np.nan,
            "mfe_r": np.nan,
            "mae_r": np.nan
        }

    mfe, mae = 0.0, 0.0
    outcome = None
    realized = np.nan
    exit_idx = None
    exit_price = np.nan
    ambiguous_bar = False

    for j in range(start, start + H25):
        mfe = max(mfe, (highs[j] - entry) / risk)
        mae = min(mae, (lows[j] - entry) / risk)

        stop_hit = lows[j] <= stop
        target_hit = highs[j] >= target

        if stop_hit and target_hit:
            outcome = "AMBIGUOUS"
            ambiguous_bar = True
            exit_idx = j
            break

        if stop_hit:
            outcome = "STOP_FIRST"
            realized = -1.0
            exit_idx = j
            exit_price = stop
            break

        if target_hit:
            outcome = "SUCCESS_25D"
            realized = 1.5
            exit_idx = j
            exit_price = target
            break

    if outcome is None:
        exit_idx = start + H25 - 1
        exit_price = closes[exit_idx]
        realized = (exit_price - entry) / risk
        outcome = "TIMEOUT_25D"

    result = {
        **base,
        "outcome_25d": outcome,
        "realized_r": realized,
        "exit_date": dates[exit_idx],
        "exit_price": exit_price,
        "sessions_to_exit": exit_idx - entry_idx,
        "mfe_r": mfe,
        "mae_r": mae,
        "ambiguous_bar_in_mfe_mae": int(ambiguous_bar),
    }

    if outcome == "TIMEOUT_25D":
        end60 = start + H60

        if end60 > len(closes):
            result["outcome_60d"] = "INCOMPLETE_60D"
        else:
            outcome60 = "TIMEOUT_60D"

            for j in range(start + H25, end60):
                sh = lows[j] <= stop
                th = highs[j] >= target

                if sh and th:
                    outcome60 = "AMBIGUOUS_60D"
                    break

                if sh:
                    outcome60 = "STOP_60D"
                    break

                if th:
                    outcome60 = "SUCCESS_60D"
                    break

            result["outcome_60d"] = outcome60
    else:
        result["outcome_60d"] = outcome

    return result


def safe_div(a, b):
    return (
        a / b
        if np.isfinite(a) and np.isfinite(b) and b != 0
        else np.nan
    )


def build_event_features(
    symbol, df, i, nifty, current_nifty750, historical_top750
):
    r = df.iloc[i]
    date = r.date

    prior_n = nifty.loc[date] if date in nifty.index else None

    nifty60 = (
        prior_n.nifty_perf60
        if prior_n is not None else np.nan
    )

    nifty_above = (
        prior_n.nifty_above_sma50
        if prior_n is not None else np.nan
    )

    # Select shortest qualifying shelf only.
    shelf = None

    for length in range(7, 26):
        base = df.iloc[i-length:i]
        sh = base.high.max()
        sl = base.low.min()

        if sl > 0 and (sh - sl) / sl <= 0.22 and r.close > sh:
            shelf = (
                length,
                float(sh),
                float(sl),
                float((sh-sl)/sl),
                base
            )
            break

    if shelf is None:
        return None

    base_len, pivot, base_low, spread, base_df = shelf

    atr = r.atr14_prior

    if not np.isfinite(atr) or atr <= 0:
        return None

    if not np.isfinite(r.vol_sma20_prior) or r.vol_sma20_prior <= 0:
        return None

    if not np.isfinite(r.turnover50_prior_cr):
        return None

    min_turnover = (
        5.0 if current_nifty750 or historical_top750 else 2.0
    )

    if r.turnover50_prior_cr < min_turnover:
        return None

    if not np.isfinite(r.sma200) or r.close < 0.90 * r.sma200:
        return None

    if (
        r.close <= r.open
        or r.close <= pivot
        or r.volume < 1.05 * r.vol_sma20_prior
    ):
        return None

    vol_surge = safe_div(r.volume, r.vol_sma20_prior)

    deliv_surge = safe_div(
        r.delivery_vol_clean,
        r.delivery_sma20_prior
    )

    delivery_skew = np.nan
    delivery_cov_base = base_df.delivery_observed.mean()

    up_delivery = 0.0
    down_delivery = 0.0

    for k in range(1, len(base_df)):
        row = base_df.iloc[k]
        prev = base_df.iloc[k-1]

        if (
            row.delivery_observed == 1
            and np.isfinite(row.delivery_vol_clean)
        ):
            if row.close >= prev.close:
                up_delivery += row.delivery_vol_clean
            else:
                down_delivery += row.delivery_vol_clean

    if delivery_cov_base >= MIN_DELIVERY_COVERAGE:
        delivery_skew = (
            up_delivery / down_delivery
            if down_delivery > 0
            else (10.0 if up_delivery > 0 else np.nan)
        )

    base_abs = base_df.daily_absorption
    base_abs_cov = base_abs.notna().mean()

    base_absorption = (
        base_abs.mean()
        if base_abs_cov >= MIN_DELIVERY_COVERAGE
        else np.nan
    )

    base_vol_avg = base_df.volume.mean()

    vdu3 = (
        safe_div(df.iloc[i-3:i].volume.min(), r.vol_sma20_prior)
        if i >= 3 else np.nan
    )

    vdu5 = (
        safe_div(df.iloc[i-5:i].volume.min(), r.vol_sma20_prior)
        if i >= 5 else np.nan
    )

    vdu5avg = (
        safe_div(df.iloc[i-5:i].volume.mean(), r.vol_sma20_prior)
        if i >= 5 else np.nan
    )

    rs60 = (
        r.perf60 - nifty60
        if np.isfinite(nifty60) and np.isfinite(r.perf60)
        else np.nan
    )

    norm_rs = (
        safe_div(rs60, atr / r.close)
        if np.isfinite(rs60) and r.close > 0
        else np.nan
    )

    dist52 = (
        safe_div(r.close, r.high252_prior)
        if np.isfinite(r.high252_prior)
        else np.nan
    )

    atr_contraction = safe_div(r.atr5_prior, atr)
    body_atr = safe_div(r.candle_body, atr)
    upper_wick_atr = safe_div(r.upper_wick, atr)
    lower_wick_atr = safe_div(r.lower_wick, atr)

    breakout_gap_pct = (
        safe_div(
            r.open - df.iloc[i-1].close,
            df.iloc[i-1].close
        ) * 100
        if i > 0 else np.nan
    )

    features = {
        "event_id": f"{symbol}_{date}",
        "symbol": symbol,
        "signal_date": date,

        "is_current_nifty750": int(current_nifty750),
        "is_historical_top750_liquidity_proxy": int(historical_top750),

        "signal_close": r.close,
        "signal_open": r.open,
        "signal_high": r.high,
        "signal_low": r.low,
        "signal_volume": r.volume,

        "base_len": base_len,
        "base_high": pivot,
        "base_low": base_low,
        "base_width_pct": spread * 100,
        "base_width_atr": safe_div(pivot-base_low, atr),
        "base_volume_avg": base_vol_avg,
        "base_volume_vs_prior20": safe_div(
            base_vol_avg, r.vol_sma20_prior
        ),

        "base_delivery_coverage": delivery_cov_base * 100,
        "base_delivery_up_down_skew": delivery_skew,
        "base_absorption": base_absorption,
        "base_absorption_coverage": base_abs_cov * 100,

        "volume_surge": vol_surge,
        "vdu_3d_min_ratio": vdu3,
        "vdu_5d_min_ratio": vdu5,
        "vdu_5d_avg_ratio": vdu5avg,

        "delivery_surge": deliv_surge,
        "delivery_pct_signal": r.delivery_pct_clean,
        "delivery_pct_mean50_prior": r.delivery_pct_mean50_prior,
        "delivery_pct_z": r.delivery_pct_z,
        "delivery_coverage20_prior": r.delivery_coverage20_prior * 100,
        "delivery_coverage50_prior": r.delivery_coverage50_prior * 100,

        "perf30_pct": (
            r.perf30 * 100 if np.isfinite(r.perf30) else np.nan
        ),
        "perf60_pct": (
            r.perf60 * 100 if np.isfinite(r.perf60) else np.nan
        ),
        "nifty_perf60_pct": (
            nifty60 * 100 if np.isfinite(nifty60) else np.nan
        ),
        "rs60_difference_pct_points": (
            rs60 * 100 if np.isfinite(rs60) else np.nan
        ),
        "norm_rs": norm_rs,

        "close_vs_sma50_pct": (
            safe_div(r.close, r.sma50) * 100
            if np.isfinite(r.sma50) else np.nan
        ),
        "close_vs_sma200_pct": (
            safe_div(r.close, r.sma200) * 100
            if np.isfinite(r.sma200) else np.nan
        ),
        "dist_52w_high_ratio": dist52,

        "atr14_pct_close": safe_div(atr, r.close) * 100,
        "atr_contraction_5_to_14": atr_contraction,

        "breakout_close_range": r.close_range_pct,
        "breakout_body_atr": body_atr,
        "breakout_upper_wick_atr": upper_wick_atr,
        "breakout_lower_wick_atr": lower_wick_atr,

        "breakout_gap_pct_descriptive_only": breakout_gap_pct,

        "nifty_above_sma50": nifty_above,
        "turnover50_prior_cr": r.turnover50_prior_cr,
        "delivery_observed_on_signal": int(r.delivery_observed),
    }

    return features, pivot, base_low, atr


def add_entry_record(
    rows, features, method, entry_idx, entry, stop,
    highs, lows, closes, dates
):
    result = eval_trade(
        highs, lows, closes, dates,
        entry_idx, entry, stop
    )

    row = dict(features)
    row.update(result)

    row["entry_method"] = method

    row["entry_session_offset_from_signal"] = (
        entry_idx - features["signal_index"]
    )

    row["entry_time_feature_cutoff"] = (
        "NEXT_SESSION" if method == "A1"
        else "SIGNAL_SESSION_INTRADAY"
    )

    # B/C enter intraday on signal day.
    # Signal-day close, final volume, delivery and candle features
    # are therefore post-entry and must not be used as predictors.
    row["signal_day_features_available_at_entry"] = int(
        method == "A1"
    )

    row["post_entry_feature_warning"] = (
        "" if method == "A1"
        else
        "Signal-day close/volume/delivery/candle features are post-entry; "
        "exclude as predictors for B/C."
    )

    row["signal_index"] = features["signal_index"]

    rows.append(row)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    current_nifty750 = load_nifty750_symbols()
    nifty = load_nifty_benchmark()

    stock_files = [
        p for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
        if is_stock_json(p)
    ]

    if not stock_files:
        raise RuntimeError(
            "No stock JSON files found in data/. Check repository structure."
        )

    stock_audit = []
    duplicate_audit = []
    split_audit = []

    stocks = {}
    breadth_rows = []
    turnover_rows = []

    print(f"Found {len(stock_files)} candidate stock JSON files.")

    for count, path in enumerate(stock_files, 1):
        symbol = os.path.splitext(os.path.basename(path))[0].upper()

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            cleaned = clean_stock(
                raw, symbol,
                stock_audit, duplicate_audit, split_audit
            )

            if cleaned is None:
                continue

            df = add_indicators(cleaned)
            stocks[symbol] = df

            subset = df.dropna(
                subset=["sma50", "turnover50_prior_cr"]
            )

            for rr in subset[
                ["date", "close", "sma50", "turnover50_prior_cr"]
            ].itertuples(index=False):

                breadth_rows.append((
                    rr.date,
                    symbol,
                    int(rr.close >= rr.sma50)
                ))

                turnover_rows.append((
                    rr.date,
                    symbol,
                    rr.turnover50_prior_cr
                ))

        except Exception as exc:
            stock_audit.append({
                "symbol": symbol,
                "status": f"ERROR_{str(exc)[:120]}",
                "rows": 0
            })

        if count % 100 == 0:
            print(f"Read {count}/{len(stock_files)} stock files...")

    if not stocks:
        raise RuntimeError(
            "No usable stock files. Check the input data and audit."
        )

    # Historical top-750 turnover ranking is a proxy,
    # NOT historical index membership.
    historical_top750_by_date = {}

    if turnover_rows:
        tdf = pd.DataFrame(
            turnover_rows,
            columns=["date", "symbol", "turnover"]
        )

        tdf["rank"] = tdf.groupby("date").turnover.rank(
            ascending=False, method="min"
        )

        for date, group in tdf[tdf["rank"] <= 750].groupby("date"):
            historical_top750_by_date[date] = set(group.symbol)

    breadth_all = {}
    breadth_n750 = {}

    if breadth_rows:
        bdf = pd.DataFrame(
            breadth_rows,
            columns=["date", "symbol", "above50"]
        )

        breadth_all = (
            bdf.groupby("date").above50.mean() * 100
        ).to_dict()

        bn = bdf[bdf.symbol.isin(current_nifty750)]

        if not bn.empty:
            breadth_n750 = (
                bn.groupby("date").above50.mean() * 100
            ).to_dict()

    trade_rows = []

    for symbol, df in stocks.items():
        df = df.reset_index(drop=True)

        dates = df.date.to_numpy()
        highs = df.high.to_numpy(float)
        lows = df.low.to_numpy(float)
        closes = df.close.to_numpy(float)

        # Reserve next-session A1 entry plus 25 future sessions.
        for i in range(252, len(df) - 26):
            signal_date = dates[i]

            if signal_date < START_DATE:
                continue

            current_flag = symbol in current_nifty750

            historical_flag = (
                symbol in historical_top750_by_date.get(signal_date, set())
            )

            built = build_event_features(
                symbol, df, i, nifty,
                current_flag, historical_flag
            )

            if built is None:
                continue

            features, pivot, base_low, atr = built

            features["signal_index"] = i

            features["market_breadth_all_pct"] = (
                breadth_all.get(signal_date, np.nan)
            )

            features["market_breadth_current_nifty750_pct"] = (
                breadth_n750.get(signal_date, np.nan)
            )

            # A1: exact pivot entry on the next trading session.
            stop_a = round(min(lows[i], base_low) * 0.995, 2)

            add_entry_record(
                trade_rows, features, "A1",
                i + 1, round(pivot, 2), stop_a,
                highs, lows, closes, dates
            )

            # B/C: only when preceding candle meets cheat-bar definition.
            prev = df.iloc[i - 1]
            prev_range = prev.high - prev.low

            cheat = (
                prev_range <= 0.75 * atr
                and prev.high >= base_low + 0.65 * (pivot - base_low)
                and prev.high < pivot
            )

            if cheat:
                entry_b = round(prev.high + 0.05, 2)
                stop_b = round(prev.low * 0.995, 2)

                add_entry_record(
                    trade_rows, features, "B",
                    i, entry_b, stop_b,
                    highs, lows, closes, dates
                )

                stop_c = round(entry_b - 1.5 * atr, 2)

                add_entry_record(
                    trade_rows, features, "C",
                    i, entry_b, stop_c,
                    highs, lows, closes, dates
                )

    trades = pd.DataFrame(trade_rows)

    if trades.empty:
        raise RuntimeError(
            "No breakout records generated. Check filters and data."
        )

    preferred = [
        "event_id", "symbol", "signal_date", "entry_method",
        "outcome_25d", "realized_r", "entry", "stop",
        "target_1_5r", "risk_pct", "entry_date", "exit_date",
        "sessions_to_exit", "mfe_r", "mae_r", "outcome_60d"
    ]

    cols = preferred + [
        c for c in trades.columns if c not in preferred
    ]

    trades = trades[cols].sort_values(
        ["signal_date", "symbol", "entry_method"]
    ).reset_index(drop=True)

    trades.to_csv(
        os.path.join(DATA_DIR, "breakout_trade_dataset.csv"),
        index=False
    )

    pd.DataFrame(stock_audit).to_csv(
        os.path.join(DATA_DIR, "breakout_stock_audit.csv"),
        index=False
    )

    pd.DataFrame(duplicate_audit).to_csv(
        os.path.join(DATA_DIR, "breakout_duplicate_audit.csv"),
        index=False
    )

    pd.DataFrame(split_audit).to_csv(
        os.path.join(DATA_DIR, "breakout_corporate_action_audit.csv"),
        index=False
    )

    summary_lines = [
        f"Stocks included: {len(stocks)}",
        f"Candidate stock files: {len(stock_files)}",
        f"Total trade-entry records: {len(trades)}",
        f"Unique breakout events: {trades.event_id.nunique()}",
        f"Signal date range: {trades.signal_date.min()} to {trades.signal_date.max()}",
        ""
    ]

    for method in ("A1", "B", "C"):
        sub = trades[trades.entry_method == method]

        summary_lines.append(
            f"[{method}] records={len(sub)}"
        )

        if sub.empty:
            continue

        for label, count in sub.outcome_25d.value_counts(
            dropna=False
        ).items():
            summary_lines.append(
                f"  {label}: {count} ({count/len(sub)*100:.1f}%)"
            )

        resolved = sub[
            sub.outcome_25d.isin([
                "SUCCESS_25D",
                "STOP_FIRST",
                "TIMEOUT_25D"
            ])
        ]

        if not resolved.empty:
            wins = int(
                (resolved.outcome_25d == "SUCCESS_25D").sum()
            )

            losses = int(
                (resolved.outcome_25d == "STOP_FIRST").sum()
            )

            pos = resolved.loc[
                resolved.realized_r > 0, "realized_r"
            ].sum()

            neg = abs(
                resolved.loc[
                    resolved.realized_r < 0, "realized_r"
                ].sum()
            )

            pf = pos / neg if neg > 0 else np.nan

            line = (
                f"  Resolved N={len(resolved)}; "
                f"success%={wins/len(resolved)*100:.1f}; "
                f"stop%={losses/len(resolved)*100:.1f}; "
                f"MeanR={resolved.realized_r.mean():+.3f}; "
            )

            line += (
                f"PF={pf:.3f}"
                if np.isfinite(pf)
                else "PF=N/A"
            )

            summary_lines.append(line)

        summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    with open(
        os.path.join(DATA_DIR, "breakout_run_summary.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(summary_text)

    print("\n" + summary_text)

    print("\nOUTPUT FILES CREATED:")

    for name in (
        "breakout_trade_dataset.csv",
        "breakout_stock_audit.csv",
        "breakout_duplicate_audit.csv",
        "breakout_corporate_action_audit.csv",
        "breakout_run_summary.txt"
    ):
        print(" - " + os.path.join(DATA_DIR, name))

    print(
        "\nCode 1 complete. Do not optimise parameters from this output; "
        "Code 2 will analyse the groups."
    )


if __name__ == "__main__":
    main()
