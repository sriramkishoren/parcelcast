"""
Evaluation metrics — using the team's exact metrics:

  - Traditional Error: (Sum Forecast - Sum Actual) / Sum Actual * 100
                       Primary metric. Captures aggregate bias direction.
  - WMAPE: Weighted Mean Absolute Percentage Error
           Secondary metric. Captures absolute accuracy.
  - Lag Analysis: forecast accuracy at lag-1, lag-2, lag-3, lag-4 horizons
                  (mirrors WPR report format)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Core metrics ─────────────────────────────────────────────────────────────

def traditional_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Traditional Error = (Sum Forecast - Sum Actual) / Sum Actual * 100

    Positive = over-forecast, Negative = under-forecast.
    Reported as a percentage.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    actual_sum = y_true.sum()
    if actual_sum == 0:
        return np.nan
    return (y_pred.sum() - actual_sum) / actual_sum * 100


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error.

    WMAPE = Sum(|actual - forecast|) / Sum(actual) * 100
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom * 100


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "traditional_error_pct": round(traditional_error(y_true, y_pred), 2),
        "wmape_pct": round(wmape(y_true, y_pred), 2),
    }


# ─── Lag analysis ─────────────────────────────────────────────────────────────

def lag_analysis(
    actuals: pd.Series,
    forecasts_by_lag: dict[int, pd.Series],
) -> pd.DataFrame:
    """
    Build the lag-analysis table:

       Lag | Traditional Error % | WMAPE %
        1  |        ...          |  ...
        2  |        ...          |  ...
        3  |        ...          |  ...
        4  |        ...          |  ...

    `forecasts_by_lag` is a dict mapping lag (in weeks) to a Series of forecasts
    aligned with `actuals`.
    """
    rows = []
    for lag in sorted(forecasts_by_lag.keys()):
        f = forecasts_by_lag[lag]
        # Align forecasts and actuals by index
        aligned = pd.concat([actuals, f], axis=1, keys=["actual", "forecast"]).dropna()
        if len(aligned) == 0:
            continue
        rows.append({
            "lag_weeks": lag,
            "traditional_error_pct": round(traditional_error(aligned["actual"], aligned["forecast"]), 2),
            "wmape_pct": round(wmape(aligned["actual"], aligned["forecast"]), 2),
            "n_obs": len(aligned),
        })
    return pd.DataFrame(rows)


def build_scorecard(
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Returns DataFrame: model | traditional_error_pct | wmape_pct | rank_wmape
    """
    rows = []
    for name, y_pred in predictions_by_model.items():
        m = all_metrics(y_true, y_pred)
        m["model"] = name
        rows.append(m)
    df = pd.DataFrame(rows)[["model", "traditional_error_pct", "wmape_pct"]]
    df["rank_wmape"] = df["wmape_pct"].rank().astype(int)
    return df.sort_values("rank_wmape").reset_index(drop=True)
