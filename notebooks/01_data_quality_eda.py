# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # 📦 ParcelCast — Notebook 1
# ## Data Loading, Quality & Exploration
#
# **Purpose:** Take raw M5 data → reframe as parcel forecasting problem → assess
# quality → clean systematically → understand the time series.
#
# This notebook proves: *I don't blindly trust data. I assess, document, and
# validate before modeling.*

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent))

from src.data_loader import (
    load_and_prepare,
    aggregate_to_weekly,
    CATEGORY_TO_CHANNEL,
    STATE_TO_REGION,
)
from src.parcel_transform import (
    convert_units_to_packages,
    split_packages_by_carrier,
    audit_carrier_shares,
    UPP_BASELINE,
    CARRIER_BASE_SHARES,
)
from src.quality import (
    profile_dataframe,
    detect_outliers_iqr,
    CleaningAuditLog,
    winsorize_with_log,
    interpolate_missing,
    normalize_shares,
    run_validation_suite,
)

sns.set_theme(style="whitegrid", palette="muted")
PRESENTATION_DIR = Path.cwd().parent / "presentation"
PRESENTATION_DIR.mkdir(exist_ok=True)

audit = CleaningAuditLog()

# %% [markdown]
# ## 1. Load M5 → ParcelCast hierarchy
#
# Mapping applied:
# - Store → FC
# - State → Region (CA→WEST, TX→SOUTH, WI→MIDWEST)
# - Category → Channel (FOODS/HOUSEHOLD → 1P, HOBBIES → WFS)

# %%
print("Domain mapping:")
print(f"  States → Regions:       {STATE_TO_REGION}")
print(f"  Categories → Channels:  {CATEGORY_TO_CHANNEL}")
print(f"  UPP baselines:          {UPP_BASELINE}")
print(f"  Carrier base shares:    {CARRIER_BASE_SHARES}")

# %%
long_df = load_and_prepare()
print(f"\nLoaded long-format data: {len(long_df):,} rows")
long_df.head()

# %% [markdown]
# ## 2. Aggregate to weekly (region × channel × week)
# Walmart fiscal weeks (Saturday-to-Friday).

# %%
weekly = aggregate_to_weekly(long_df, by=["region", "channel"])
print(f"Weekly aggregated: {len(weekly):,} rows")
weekly.head()

# %% [markdown]
# ## 3. Convert units → packages via UPP

# %%
weekly = convert_units_to_packages(weekly)
weekly.head()

# %%
# Show the UPP trend (1P declining, WFS static) — the project's key insight
CHANNEL_COLORS = {"1P": "steelblue", "WFS": "darkslategray"}
fig, ax = plt.subplots(figsize=(12, 5))
for ch in weekly["channel"].unique():
    sub = weekly[weekly["channel"] == ch].drop_duplicates("week_start")
    ax.plot(sub["week_start"], sub["upp"], label=ch, linewidth=2,
            color=CHANNEL_COLORS.get(ch, "steelblue"))
ax.set_title(
    "UPP (Units Per Package) trend — 1P declining, WFS static\n"
    "Key insight: UPP forecast error compounds into package forecast error",
    fontsize=14, fontweight="bold",
)
ax.set_ylabel("UPP")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PRESENTATION_DIR / "01_upp_trend.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Split packages → per-carrier volumes (with regional adjustment)

# %%
carrier_df = split_packages_by_carrier(weekly)
print(f"Carrier-level rows: {len(carrier_df):,}")
carrier_df.head(10)

# %%
# Audit: do carrier shares sum to 1.0?
share_audit = audit_carrier_shares(carrier_df)
print(f"Rows where shares fail sum-to-1: {len(share_audit)}")

# %% [markdown]
# ## 5. Data Quality Profile

# %%
profile = profile_dataframe(carrier_df)
profile

# %%
# Outlier detection on package volume
outliers = detect_outliers_iqr(carrier_df["carrier_packages"])
print("Carrier-package outlier analysis:")
for k, v in outliers.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 6. Cleaning Decisions — with reasoning

# %% [markdown]
# **Decision 1: Winsorize extreme volume outliers** — cap at 1st/99th percentile.
# Rationale: extreme spikes are likely real (events) but pull model fitting toward
# the tail. Winsorization preserves time-series continuity (no row deletion).

# %%
carrier_df = winsorize_with_log(
    carrier_df, "carrier_packages", audit,
    lower_pct=0.01, upper_pct=0.99,
    reason="Cap extreme outliers; preserves time-series continuity",
)

# %% [markdown]
# **Decision 2: Interpolate missing values** within each (region, channel, carrier)
# group. Rationale: linear interpolation preserves temporal patterns; mean fill
# would destroy them.

# %%
# (M5 has no missing values, but the pipeline supports it)
carrier_df = interpolate_missing(
    carrier_df, "carrier_packages", audit,
    group_cols=["region", "channel", "carrier"],
)

# %% [markdown]
# **Decision 3: Renormalize carrier shares** to sum to exactly 1.0 per group.
# Rationale: floating-point arithmetic and regional adjustments may introduce
# tiny drift; this guarantees consistency.

# %%
carrier_df = normalize_shares(
    carrier_df, "carrier_share",
    group_cols=["region", "channel", "week_start"],
    audit=audit,
)

# %% [markdown]
# **Decision 4: Recalculate `carrier_packages` = packages × carrier_share.**
# Rationale: ensure the derived column is consistent with the (now normalized)
# components.

# %%
carrier_df["carrier_packages"] = carrier_df["packages"] * carrier_df["carrier_share"]
carrier_df["carrier_cost"] = carrier_df["carrier_packages"] * carrier_df.apply(
    lambda r: {"FedEx": 8.5, "UPS": 9.2, "IP": 6.4, "OnTrac": 7.1}[r["carrier"]],
    axis=1,
)

# %% [markdown]
# ### Cleaning audit log (the artifact)

# %%
audit_df = audit.to_df()
audit_df

# %% [markdown]
# ## 7. Time-series understanding

# %% [markdown]
# **Decomposition** — additive trend + yearly seasonality + residual

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# Pick one series for decomposition: total network 1P weekly volume
network_1p = (
    carrier_df[carrier_df["channel"] == "1P"]
    .groupby("week_start")["carrier_packages"]
    .sum()
    .sort_index()
)

decomp = seasonal_decompose(network_1p, period=52, model="multiplicative")
fig = decomp.plot()
fig.set_size_inches(12, 8)
# Recolor decomposition lines to match the project's primary palette
for ax in fig.axes:
    for line in ax.get_lines():
        line.set_color("steelblue")
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Weekly 1P Network Volume — Decomposition (52-week period)",
    y=1.0, fontsize=14, fontweight="bold",
)
fig.tight_layout()
fig.savefig(PRESENTATION_DIR / "02_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Stationarity check** — Augmented Dickey-Fuller test

# %%
from statsmodels.tsa.stattools import adfuller

adf_stat, p_value, *_ = adfuller(network_1p.dropna())
print(f"ADF statistic: {adf_stat:.4f}")
print(f"p-value:       {p_value:.4f}")
print(f"=> {'Stationary' if p_value < 0.05 else 'NON-stationary — differencing needed for ARIMA-style models'}")

# %% [markdown]
# **Correlation analysis** — what drives package volume?

# %%
# Build a simple feature set for correlation
corr_df = carrier_df.copy()
corr_df["fiscal_week"] = pd.to_datetime(corr_df["week_start"]).dt.isocalendar().week.astype(int)
corr_df["month"] = pd.to_datetime(corr_df["week_start"]).dt.month
corr_df["is_peak"] = corr_df["fiscal_week"].between(46, 52).astype(int)

corr_features = ["units", "upp", "packages", "carrier_packages", "carrier_share",
                 "fiscal_week", "month", "is_peak"]
corr_matrix = corr_df[corr_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation matrix — package volume drivers",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PRESENTATION_DIR / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Post-cleaning validation suite

# %%
validation = run_validation_suite(carrier_df)
validation

# %%
n_pass = validation["pass"].sum()
n_total = len(validation)
print(f"Validation: {n_pass}/{n_total} checks passed")

# %% [markdown]
# ## 9. Save artifacts

# %%
DATA_DIR = Path.cwd().parent / "data"

# Network-level weekly volumes (one row per region × channel × carrier × week)
carrier_df.to_parquet(DATA_DIR / "weekly_network_volumes.parquet", index=False)

# Region+channel weekly (no carrier split — used by modeling notebook)
weekly.to_parquet(DATA_DIR / "weekly_region_channel.parquet", index=False)

# Audit + validation
audit_df.to_csv(DATA_DIR / "cleaning_audit_log.csv", index=False)
validation.to_csv(DATA_DIR / "validation_results.csv", index=False)

print(f"Saved {len(carrier_df):,} rows to weekly_network_volumes.parquet")
print(f"Saved {len(weekly):,} rows to weekly_region_channel.parquet")
print(f"Audit: {len(audit_df)} entries")
print(f"Validation: {n_pass}/{n_total} passed")
