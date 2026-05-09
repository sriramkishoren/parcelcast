"""
Data quality utilities: profiling, cleaning decisions, audit log.

Philosophy: every transformation gets recorded. The cleaning audit log is the
artifact that proves "I don't blindly trust data."
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


# ─── Profiling ────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, numeric_cols: list[str] = None) -> pd.DataFrame:
    """
    Return a per-column profile: dtype, missing %, unique count, basic stats.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "unique": s.nunique(dropna=True),
            "n": len(s),
        }
        if numeric_cols is None or col in numeric_cols:
            if pd.api.types.is_numeric_dtype(s):
                row.update({
                    "min": s.min(),
                    "max": s.max(),
                    "mean": round(s.mean(), 2),
                    "std": round(s.std(), 2),
                    "zero_count": int((s == 0).sum()),
                    "zero_pct": round((s == 0).mean() * 100, 2),
                    "negative_count": int((s < 0).sum()),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> dict:
    """IQR-based outlier bounds and counts."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower_bound": lo, "upper_bound": hi,
        "outliers_low": int((s < lo).sum()),
        "outliers_high": int((s > hi).sum()),
        "outlier_pct": round(((s < lo) | (s > hi)).mean() * 100, 2),
    }


# ─── Audit log ────────────────────────────────────────────────────────────────

class CleaningAuditLog:
    """Records every cleaning decision with before/after metrics."""

    def __init__(self):
        self.entries = []

    def record(
        self,
        action: str,
        column: str,
        reason: str,
        before: dict = None,
        after: dict = None,
    ):
        self.entries.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "column": column,
            "reason": reason,
            "before": before or {},
            "after": after or {},
        })

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def __len__(self):
        return len(self.entries)


# ─── Cleaning ops ─────────────────────────────────────────────────────────────

def winsorize(s: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Cap a series at given percentiles. Returns the capped series."""
    lo, hi = s.quantile([lower_pct, upper_pct])
    return s.clip(lower=lo, upper=hi)


def winsorize_with_log(
    df: pd.DataFrame, col: str, audit: CleaningAuditLog,
    lower_pct: float = 0.01, upper_pct: float = 0.99,
    reason: str = "Cap extreme outliers without deleting rows (preserves time series)",
) -> pd.DataFrame:
    """Winsorize a column and log the change."""
    df = df.copy()
    before_stats = {
        "min": df[col].min(), "max": df[col].max(),
        "mean": round(df[col].mean(), 2),
    }
    df[col] = winsorize(df[col], lower_pct, upper_pct)
    after_stats = {
        "min": df[col].min(), "max": df[col].max(),
        "mean": round(df[col].mean(), 2),
    }
    audit.record("winsorize", col, reason, before_stats, after_stats)
    return df


def interpolate_missing(
    df: pd.DataFrame, col: str, audit: CleaningAuditLog,
    group_cols: list[str] = None,
    reason: str = "Linear interpolation preserves temporal patterns; mean fill would destroy them",
) -> pd.DataFrame:
    """Interpolate missing values within groups (e.g., per FC)."""
    df = df.copy()
    before_missing = int(df[col].isna().sum())

    if group_cols:
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.interpolate(method="linear"))
    else:
        df[col] = df[col].interpolate(method="linear")

    after_missing = int(df[col].isna().sum())
    audit.record(
        "interpolate", col, reason,
        {"missing": before_missing}, {"missing": after_missing},
    )
    return df


def normalize_shares(
    df: pd.DataFrame, share_col: str, group_cols: list[str], audit: CleaningAuditLog,
    reason: str = "Ensure carrier shares sum to exactly 1.0 per group",
) -> pd.DataFrame:
    """Renormalize a share column to sum to 1.0 within each group."""
    df = df.copy()
    sums_before = df.groupby(group_cols)[share_col].sum()
    bad_before = int((~np.isclose(sums_before, 1.0, atol=1e-6)).sum())

    df[share_col] = df.groupby(group_cols)[share_col].transform(lambda x: x / x.sum())

    sums_after = df.groupby(group_cols)[share_col].sum()
    bad_after = int((~np.isclose(sums_after, 1.0, atol=1e-6)).sum())

    audit.record(
        "normalize_shares", share_col, reason,
        {"groups_failing_check": bad_before}, {"groups_failing_check": bad_after},
    )
    return df


# ─── Post-cleaning validation ─────────────────────────────────────────────────

def run_validation_suite(df: pd.DataFrame) -> pd.DataFrame:
    """Run a battery of automated checks; return pass/fail per check."""
    checks = []

    def check(name, condition, detail=""):
        checks.append({"check": name, "pass": bool(condition), "detail": detail})

    # 1. No missing values in key columns
    if "packages" in df.columns:
        check("packages: no missing", df["packages"].isna().sum() == 0,
              f"{df['packages'].isna().sum()} missing")

    # 2. No negative volumes
    if "packages" in df.columns:
        check("packages: no negatives", (df["packages"] < 0).sum() == 0,
              f"{(df['packages'] < 0).sum()} negative")

    # 3. Carrier shares sum to 1.0 per group (if applicable)
    if "carrier_share" in df.columns:
        sums = df.groupby(["region", "channel", "week_start"])["carrier_share"].sum()
        check("carrier shares sum to 1.0",
              np.allclose(sums, 1.0, atol=1e-6),
              f"{(~np.isclose(sums, 1.0, atol=1e-6)).sum()} groups fail")

    # 4. Carrier packages sum equals total packages per group
    if "carrier_packages" in df.columns and "packages" in df.columns:
        agg = df.groupby(["region", "channel", "week_start"]).agg(
            total_pkgs=("packages", "first"),
            sum_carrier=("carrier_packages", "sum"),
        )
        check("carrier_packages reconcile to total",
              np.allclose(agg["total_pkgs"], agg["sum_carrier"], rtol=1e-4),
              "")

    # 5. Date range is continuous
    if "week_start" in df.columns:
        weeks = pd.to_datetime(df["week_start"].unique()).sort_values()
        gaps = (weeks[1:] - weeks[:-1]).days
        check("weeks: no gaps", (gaps == 7).all() if len(gaps) else True, "")

    # 6. UPP is in expected range
    if "upp" in df.columns:
        check("upp: in plausible range",
              df["upp"].between(1.0, 3.0).all(),
              f"{(~df['upp'].between(1.0, 3.0)).sum()} out of range")

    # 7. Region/channel coverage
    if "region" in df.columns and "channel" in df.columns:
        n_regions = df["region"].nunique()
        n_channels = df["channel"].nunique()
        check("hierarchy coverage",
              n_regions == 3 and n_channels == 2,
              f"regions={n_regions}, channels={n_channels}")

    # 8. No duplicate rows on key
    if {"region", "channel", "carrier", "week_start"}.issubset(df.columns):
        dups = df.duplicated(subset=["region", "channel", "carrier", "week_start"]).sum()
        check("no duplicate (region, channel, carrier, week)", dups == 0, f"{dups} dups")

    return pd.DataFrame(checks)
