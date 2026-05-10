"""
Parcel-domain transformations:
  1. Units → Packages via UPP (Units Per Package)
  2. Packages → Per-carrier volumes via allocation shares

UPP is treated as a TIME-VARYING parameter for 1P (declining trend per WPR
report observation), and a STATIC parameter for MP. This is the project's
key insight: UPP forecast error compounds into package forecast error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ─── UPP configuration ────────────────────────────────────────────────────────
# Illustrative values; document as assumptions in the deck.

UPP_BASELINE = {
    "1P": 2.10,   # Higher = more units bundled per package
    "MP": 1.33,  # Marketplace sellers ship more single-item packages
}

# 1P UPP is declining ~3% per year (matches WPR narrative of channel mix shift)
UPP_1P_ANNUAL_DECLINE = 0.03

# ─── Carrier configuration ────────────────────────────────────────────────────
# Network-level allocation shares. Real-world variation by region is layered on top.

CARRIER_BASE_SHARES = {
    "FedEx": 0.55,
    "UPS": 0.15,
    "IP": 0.20,    # Internal/in-house first-party last-mile carrier
    "OnTrac": 0.10,
}

# Regional adjustments (multipliers applied to base shares, then renormalized)
REGION_CARRIER_ADJUSTMENTS = {
    "WEST":    {"FedEx": 0.90, "UPS": 1.00, "IP": 1.10, "OnTrac": 1.50},  # OnTrac is West-Coast
    "SOUTH":   {"FedEx": 1.05, "UPS": 1.05, "IP": 1.00, "OnTrac": 0.30},
    "MIDWEST": {"FedEx": 1.05, "UPS": 1.10, "IP": 0.95, "OnTrac": 0.40},
}

# Cost per package — illustrative, used for cost-shift simulator
COST_PER_PACKAGE = {
    "FedEx": 8.50,
    "UPS": 9.20,
    "IP": 6.40,
    "OnTrac": 7.10,
}


# ─── UPP conversion ───────────────────────────────────────────────────────────

def get_upp(channel: str, week_start: pd.Timestamp, anchor_date: pd.Timestamp = None) -> float:
    """
    Return UPP for a given channel at a given week.

    1P UPP declines linearly at ~3% per year from the anchor date.
    MP UPP is static.
    """
    if channel == "MP":
        return UPP_BASELINE["MP"]

    if anchor_date is None:
        anchor_date = pd.Timestamp("2011-01-29")  # M5 start date

    years_elapsed = (week_start - anchor_date).days / 365.25
    decline_factor = (1 - UPP_1P_ANNUAL_DECLINE) ** years_elapsed
    return UPP_BASELINE["1P"] * decline_factor


def convert_units_to_packages(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `packages` column = units / UPP.

    Expects columns: channel, week_start, units.
    """
    df = weekly_df.copy()
    if "channel" not in df.columns or "week_start" not in df.columns:
        raise ValueError("DataFrame must have 'channel' and 'week_start' columns")

    df["upp"] = df.apply(lambda r: get_upp(r["channel"], r["week_start"]), axis=1)
    df["packages"] = df["units"] / df["upp"]
    return df


# ─── Carrier allocation ───────────────────────────────────────────────────────

def get_carrier_shares(region: str) -> dict[str, float]:
    """Return regional carrier shares, normalized to sum to 1.0."""
    if region not in REGION_CARRIER_ADJUSTMENTS:
        # Fallback to network-level base shares
        return dict(CARRIER_BASE_SHARES)

    adjustments = REGION_CARRIER_ADJUSTMENTS[region]
    raw = {c: CARRIER_BASE_SHARES[c] * adjustments[c] for c in CARRIER_BASE_SHARES}
    total = sum(raw.values())
    return {c: v / total for c, v in raw.items()}


def split_packages_by_carrier(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the dataframe to one row per (region, channel, week, carrier).

    Output has additional columns: carrier, carrier_share, carrier_packages.
    """
    if "packages" not in weekly_df.columns:
        raise ValueError("Run convert_units_to_packages() first.")

    rows = []
    for _, row in weekly_df.iterrows():
        shares = get_carrier_shares(row["region"])
        for carrier, share in shares.items():
            new_row = row.to_dict()
            new_row["carrier"] = carrier
            new_row["carrier_share"] = share
            new_row["carrier_packages"] = row["packages"] * share
            new_row["carrier_cost"] = new_row["carrier_packages"] * COST_PER_PACKAGE[carrier]
            rows.append(new_row)
    return pd.DataFrame(rows)


# ─── Audit helpers ────────────────────────────────────────────────────────────

def audit_carrier_shares(carrier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify carrier shares sum to 1.0 within each (region, channel, week).
    Returns rows that fail the check.
    """
    grouped = (
        carrier_df.groupby(["region", "channel", "week_start"])["carrier_share"]
        .sum()
        .reset_index()
        .rename(columns={"carrier_share": "share_total"})
    )
    failed = grouped[~np.isclose(grouped["share_total"], 1.0, atol=1e-6)]
    return failed
