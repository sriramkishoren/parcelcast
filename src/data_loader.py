"""
Load M5 data and map M5 hierarchy → ParcelCast (parcel-forecasting) hierarchy.

The mapping is the core "domain reframing" of the project:
  Store    → FC (Fulfillment Center)
  State    → Shipping Region
  Category → Channel (1P / MP)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Domain mapping configuration ─────────────────────────────────────────────

# Categories → Channel (the creative reframe)
CATEGORY_TO_CHANNEL = {
    "FOODS": "1P",
    "HOUSEHOLD": "1P",
    "HOBBIES": "MP",  # Treat hobbies as 3rd-party Marketplace volume
}

# State → Region (1:1 mapping, just terminology)
STATE_TO_REGION = {
    "CA": "WEST",
    "TX": "SOUTH",
    "WI": "MIDWEST",
}


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_m5_raw(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the three M5 CSVs."""
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    return {"sales": sales, "calendar": calendar, "prices": prices}


def reshape_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Convert M5 sales from wide (item × day-columns) to long format.

    Input shape: ~30,490 rows × 1,947 cols (6 ID cols + 1,941 day cols)
    Output shape: ~58M rows
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long = sales.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )
    # Memory optimization: same values, smaller dtypes. The melt explodes ~30K
    # unique ids across 1,941 days; without categoricals the id columns alone
    # consume several GB and force downstream operations into swap.
    for c in id_cols + ["d"]:
        long[c] = long[c].astype("category")
    long["units"] = pd.to_numeric(long["units"], downcast="unsigned")
    return long


def join_calendar(long_sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach real dates and event metadata."""
    cal_keep = ["d", "date", "wm_yr_wk", "weekday", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    out = long_sales.merge(calendar[cal_keep], on="d", how="left")
    out["date"] = pd.to_datetime(out["date"])
    return out


# ─── Hierarchy mapping ────────────────────────────────────────────────────────

def map_to_parcel_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the M5 → ParcelCast renaming.

    Adds columns:
      - fc_id (was store_id)
      - region (was state_id, with WEST/SOUTH/MIDWEST naming)
      - channel (derived from cat_id)
    """
    df = df.copy()
    df["fc_id"] = df["store_id"]
    df["region"] = df["state_id"].map(STATE_TO_REGION)
    df["channel"] = df["cat_id"].map(CATEGORY_TO_CHANNEL)

    if df["channel"].isna().any():
        unmapped = df[df["channel"].isna()]["cat_id"].unique()
        raise ValueError(f"Unmapped categories found: {unmapped}")

    return df


# ─── Weekly aggregation ───────────────────────────────────────────────────────

def aggregate_to_weekly(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Aggregate daily units to fiscal weeks (Saturday-to-Friday).

    `by` is the grouping spec, e.g. ['region', 'channel'].
    """
    df = df.copy()
    # M5's calendar already has wm_yr_wk — use it directly for fiscal week alignment
    if "wm_yr_wk" in df.columns:
        # Drop partial fiscal weeks. M5's evaluation set ends mid-week, so the
        # final wm_yr_wk has only 1-2 days of data — aggregating it produces a
        # phantom "week" with ~1/4 the volume that breaks downstream forecasting.
        days_per_week = df.groupby("wm_yr_wk")["date"].nunique()
        complete_weeks = days_per_week[days_per_week == 7].index
        df = df[df["wm_yr_wk"].isin(complete_weeks)]

        # Map wm_yr_wk to its Saturday start date for plotting
        wk_dates = (
            df.groupby("wm_yr_wk")["date"]
            .min()
            .rename("week_start")
            .reset_index()
        )
        df = df.merge(wk_dates, on="wm_yr_wk", how="left")
    else:
        # Fallback: pandas weekly bin
        df["week_start"] = df["date"].dt.to_period("W").dt.start_time

    grouped = (
        df.groupby(by + ["wm_yr_wk", "week_start"], as_index=False)["units"]
        .sum()
        .sort_values(by + ["week_start"])
    )
    return grouped


# ─── Convenience ──────────────────────────────────────────────────────────────

def load_and_prepare(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """End-to-end: load → reshape → join calendar → map hierarchy."""
    raw = load_m5_raw(data_dir)
    long = reshape_to_long(raw["sales"])
    long = join_calendar(long, raw["calendar"])
    long = map_to_parcel_hierarchy(long)
    return long
