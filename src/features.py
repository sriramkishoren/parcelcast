"""
Feature engineering: lags + calendar features.

Kept intentionally minimal for v1 — the goal is "ML model with feature
importance", not "best ML model possible". Add rolling features, encodings,
and price/event signals if Saturday's modeling block has spare time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Lag features ─────────────────────────────────────────────────────────────

DEFAULT_LAGS = [1, 2, 4, 8, 12, 52]


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    date_col: str = "week_start",
    lags: list[int] = None,
) -> pd.DataFrame:
    """Add lagged target values within each group."""
    df = df.sort_values(group_cols + [date_col]).copy()
    lags = lags or DEFAULT_LAGS
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def add_yoy_growth(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    date_col: str = "week_start",
) -> pd.DataFrame:
    """Year-over-year growth: current vs same week 52 weeks ago."""
    df = df.sort_values(group_cols + [date_col]).copy()
    yoy_lag = df.groupby(group_cols)[target_col].shift(52)
    df["yoy_growth"] = (df[target_col] - yoy_lag) / yoy_lag
    df["yoy_growth"] = df["yoy_growth"].replace([np.inf, -np.inf], np.nan)
    return df


# ─── Calendar features ───────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "week_start") -> pd.DataFrame:
    """
    Calendar features tuned for retail / parcel volume:
      - fiscal week number, month, quarter
      - cyclical encoding
      - retail-event flags: peak season, BTS, holiday weeks
    """
    df = df.copy()
    d = pd.to_datetime(df[date_col])

    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["fiscal_week"] = d.dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df["woy_sin"] = np.sin(2 * np.pi * df["fiscal_week"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["fiscal_week"] / 52)

    # Retail event flags
    df["is_peak"] = df["fiscal_week"].between(46, 52).astype(int)        # Black Friday → Christmas
    df["is_bts"] = df["fiscal_week"].between(30, 35).astype(int)          # Back-to-school
    df["is_holiday"] = df["fiscal_week"].isin([47, 48, 51, 52]).astype(int)

    return df


# ─── Composite ────────────────────────────────────────────────────────────────

def build_feature_set(
    df: pd.DataFrame,
    target_col: str = "packages",
    group_cols: list[str] = None,
    date_col: str = "week_start",
    include_upp: bool = True,
) -> pd.DataFrame:
    """
    One-shot pipeline: lag + YoY + calendar features.
    Optionally includes UPP as exogenous feature.
    """
    group_cols = group_cols or ["region", "channel"]
    df = add_lag_features(df, target_col, group_cols, date_col)
    df = add_yoy_growth(df, target_col, group_cols, date_col)
    df = add_calendar_features(df, date_col)

    if include_upp and "upp" in df.columns:
        # Lagged UPP — captures the 'UPP error inflates package error' insight
        df = df.sort_values(group_cols + [date_col])
        df["upp_lag_1"] = df.groupby(group_cols)["upp"].shift(1)
        df["upp_lag_4"] = df.groupby(group_cols)["upp"].shift(4)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "packages") -> list[str]:
    """Return columns suitable as model features (numeric, non-target, non-ID)."""
    skip = {target_col, "units", "week_start", "date", "wm_yr_wk", "d", "id",
            "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return cols


# ─── Feature documentation ────────────────────────────────────────────────────

FEATURE_DOCUMENTATION = {
    "lag_1":       "Last week's volume — strongest single predictor for short horizons",
    "lag_2":       "2-week-ago volume — captures very-short-term momentum",
    "lag_4":       "4-week-ago volume — captures monthly cycle",
    "lag_8":       "8-week-ago volume — early indicator of trend shifts",
    "lag_12":      "12-week-ago (quarterly) volume",
    "lag_52":      "Same week last year — captures annual seasonality directly",
    "yoy_growth":  "Year-over-year growth rate — captures structural trend changes",
    "fiscal_week": "Fiscal week number (1-52) — categorical seasonality",
    "month":       "Month — coarser seasonality",
    "quarter":     "Quarter — quarterly business cycle",
    "woy_sin":     "Cyclical encoding of week-of-year (sin component)",
    "woy_cos":     "Cyclical encoding of week-of-year (cos component)",
    "is_peak":     "Black Friday through Christmas peak season flag",
    "is_bts":      "Back-to-school season flag",
    "is_holiday":  "Holiday week flag",
    "upp":         "Current UPP — exogenous; key insight that UPP error inflates package forecast error",
    "upp_lag_1":   "Last week's UPP — proxies UPP trend without lookahead",
    "upp_lag_4":   "Monthly-lagged UPP — smoother trend signal",
}


def get_feature_table(features: list[str]) -> pd.DataFrame:
    """Build the feature documentation table for the deck."""
    return pd.DataFrame([
        {"feature": f, "business_justification": FEATURE_DOCUMENTATION.get(f, "—")}
        for f in features
    ])
