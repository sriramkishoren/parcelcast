"""
Business application layer.

This is what makes ParcelCast a 'forecasting product' rather than a 'forecasting
project'. Each function turns model output into a decision artifact stakeholders
actually consume.

All thresholds and assumptions are clearly labeled and documented in the deck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Contract & threshold parameters ──────────────────────────────────────────
# Illustrative; document as assumptions in the deck.

FEDEX_HD_CONTRACT_MIN_MONTHLY_PACKAGES = 19_500_000
FEDEX_HD_CONTRACT_MAX_MONTHLY_PACKAGES = 26_800_000

ONTRAC_TIER_2_WEEKLY_THRESHOLD = 850_000
ONTRAC_ROLLING_WINDOW_WEEKS = 6


# ─── 1. FedEx HD Monthly Contract Monitor ────────────────────────────────────

def fedex_contract_monitor(
    weekly_forecasts: pd.DataFrame,
    forecast_col: str = "fedex_packages",
    date_col: str = "week_start",
    contract_min: int = FEDEX_HD_CONTRACT_MIN_MONTHLY_PACKAGES,
    contract_max: int = FEDEX_HD_CONTRACT_MAX_MONTHLY_PACKAGES,
) -> pd.DataFrame:
    """
    Roll weekly forecasts up to monthly, compare against contract band.

    Returns DataFrame with: month | forecast | min | max | status | buffer | recommended_action
    """
    df = weekly_forecasts.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M")

    monthly = (
        df.groupby("month")[forecast_col]
        .sum()
        .reset_index()
        .rename(columns={forecast_col: "forecast_packages"})
    )

    monthly["contract_min"] = contract_min
    monthly["contract_max"] = contract_max

    def status(row):
        if row["forecast_packages"] < contract_min:
            return "BELOW MINIMUM"
        if row["forecast_packages"] > contract_max:
            return "ABOVE MAXIMUM"
        return "ON TRACK"

    def buffer(row):
        if row["forecast_packages"] < contract_min:
            return contract_min - row["forecast_packages"]
        if row["forecast_packages"] > contract_max:
            return row["forecast_packages"] - contract_max
        return min(
            row["forecast_packages"] - contract_min,
            contract_max - row["forecast_packages"],
        )

    def action(row):
        if row["forecast_packages"] < contract_min:
            return "Shift volume FROM IP/OnTrac TO FedEx (or face min-volume penalty)"
        if row["forecast_packages"] > contract_max:
            return "Shift volume FROM FedEx TO secondary carriers (UPS/IP/OnTrac)"
        return "No action — within contract band"

    monthly["status"] = monthly.apply(status, axis=1)
    monthly["buffer_packages"] = monthly.apply(buffer, axis=1).astype(int)
    monthly["recommended_action"] = monthly.apply(action, axis=1)

    monthly["forecast_packages"] = monthly["forecast_packages"].astype(int)
    monthly["month"] = monthly["month"].astype(str)
    return monthly


# ─── 2. OnTrac Tier 2 Risk Alert ──────────────────────────────────────────────

def ontrac_tier_risk_alert(
    weekly_volume: pd.DataFrame,
    volume_col: str = "ontrac_packages",
    date_col: str = "week_start",
    threshold: int = ONTRAC_TIER_2_WEEKLY_THRESHOLD,
    window: int = ONTRAC_ROLLING_WINDOW_WEEKS,
    forward_weeks: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute rolling avg of OnTrac volume, project forward using simple trend,
    flag the week where Tier 2 threshold is breached.

    Returns (full_history_df, alert_dict).
    """
    df = weekly_volume.copy().sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])

    df["rolling_avg"] = df[volume_col].rolling(window=window, min_periods=1).mean()
    df["over_threshold"] = df["rolling_avg"] > threshold

    # Project forward using last 4-week trend (simple linear)
    if len(df) >= 4:
        recent = df[volume_col].tail(4).values
        trend = (recent[-1] - recent[0]) / 3  # per week
        last_date = df[date_col].iloc[-1]
        last_val = df[volume_col].iloc[-1]

        future_rows = []
        for i in range(1, forward_weeks + 1):
            future_rows.append({
                date_col: last_date + pd.Timedelta(weeks=i),
                volume_col: max(last_val + trend * i, 0),
                "rolling_avg": np.nan,
                "over_threshold": False,
                "is_projection": True,
            })
        df["is_projection"] = False
        future_df = pd.DataFrame(future_rows)
        df = pd.concat([df, future_df], ignore_index=True)
        df["rolling_avg"] = df[volume_col].rolling(window=window, min_periods=1).mean()
        df["over_threshold"] = df["rolling_avg"] > threshold

    # Find first week (in projection or recent past) where threshold is breached
    breach = df[df["over_threshold"]].head(1)
    alert = {
        "threshold": threshold,
        "current_rolling_avg": int(df["rolling_avg"].iloc[len(df) - forward_weeks - 1]) if len(df) > forward_weeks else None,
        "breach_week": str(breach[date_col].iloc[0].date()) if not breach.empty else None,
        "weeks_until_breach": (
            int((breach[date_col].iloc[0] - pd.Timestamp.today()).days / 7)
            if not breach.empty else None
        ),
        "recommended_action": (
            "Shift OnTrac volume to IP or UPS to stay under Tier 2 threshold"
            if not breach.empty else
            "No action — OnTrac volume within Tier 1 band"
        ),
    }

    return df, alert


# ─── 3. Carrier Cost Optimizer ────────────────────────────────────────────────

def carrier_cost_shift_simulator(
    current_allocation: dict[str, int],
    cost_per_package: dict[str, float],
    shift_pct: float = 0.05,
    shift_from: list[str] = None,
    shift_to: str = "IP",
) -> pd.DataFrame:
    """
    Simulate shifting `shift_pct` of volume from `shift_from` carriers to `shift_to`.

    Returns DataFrame: scenario | FedEx | UPS | IP | OnTrac | weekly_cost | savings_vs_current
    """
    shift_from = shift_from or ["FedEx", "UPS"]

    # Current
    current_cost = sum(current_allocation[c] * cost_per_package[c] for c in current_allocation)

    # Proposed
    proposed = dict(current_allocation)
    total_to_shift = sum(int(proposed[c] * shift_pct) for c in shift_from)
    for c in shift_from:
        proposed[c] -= int(proposed[c] * shift_pct)
    proposed[shift_to] += total_to_shift

    proposed_cost = sum(proposed[c] * cost_per_package[c] for c in proposed)

    rows = [
        {
            "scenario": "Current",
            **current_allocation,
            "weekly_cost_$": round(current_cost, 0),
            "savings_$": 0,
        },
        {
            "scenario": f"Shift {int(shift_pct*100)}% from {'+'.join(shift_from)} → {shift_to}",
            **proposed,
            "weekly_cost_$": round(proposed_cost, 0),
            "savings_$": round(current_cost - proposed_cost, 0),
        },
    ]
    df = pd.DataFrame(rows)
    df["annualized_savings_$"] = df["savings_$"] * 52
    return df


# ─── 4. Executive dashboard (helper for the 4-panel chart) ───────────────────

def build_dashboard_data(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    scorecard: pd.DataFrame,
    carrier_allocation: dict[str, int],
) -> dict:
    """
    Bundle the data needed for the 4-panel executive dashboard.
    Returns a dict the notebook can pass straight to plotting code.
    """
    return {
        "forecast": forecast_df,
        "actuals": actuals_df,
        "scorecard": scorecard,
        "carrier_allocation": carrier_allocation,
    }
