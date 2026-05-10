# Data Quality & EDA — Learnings (Notebook 1)

> Consolidated notes from the walkthrough of `notebooks/01_data_quality_eda.py`.
> Written for both **practical understanding** and **interview preparation**.

---

## Table of Contents

1. [The Dataset](#1-the-dataset)
2. [Domain Reframing (M5 → ParcelCast)](#2-domain-reframing-m5--parcelcast)
3. [UPP — Units Per Package](#3-upp--units-per-package)
4. [Data Cleaning & Preprocessing](#4-data-cleaning--preprocessing)
5. [ML / EDA Concepts in This Notebook](#5-ml--eda-concepts-in-this-notebook)
6. [Outlier Treatment — Deep Dive](#6-outlier-treatment--deep-dive)
7. [Python Implementation Patterns](#7-python-implementation-patterns)
8. [Interview Preparation](#8-interview-preparation)
9. [Key Terms Glossary](#9-key-terms-glossary)
10. [Memorable Analogies](#10-memorable-analogies)

---

## 1. The Dataset

### What it represents

**M5 (Walmart's M-Competition #5 dataset)** — daily unit sales of 30,490 SKUs across 10 Walmart stores in 3 US states (CA, TX, WI), spanning ~5.4 years (Jan 2011 – June 2016). Released as 3 CSVs:

| File | What it is |
|---|---|
| `sales_train_evaluation.csv` | The fact table. **Wide format**: one row per SKU, then 1,941 columns `d_1, d_2, …, d_1941` holding unit sales per day. |
| `calendar.csv` | Lookup table mapping each `d_*` to a real date, the Walmart fiscal week (`wm_yr_wk`), the day of week, and event flags (Easter, SuperBowl, SNAP food-stamp days). |
| `sell_prices.csv` | Weekly sell prices per (store, item). Not used in Notebook 1. |

The "wide" format is unusual — most ML data is "long" (one row per observation). Wide is how OLAP/data-warehouse tables sometimes ship time series, and you have to reshape before modeling.

### Features / columns used

After cleaning, the canonical row shape is one row per **(region, channel, carrier, week_start)**, with these columns:

- **Identifiers / dimensions**: `region`, `channel`, `carrier`, `week_start`, `wm_yr_wk`
- **Volume measures**: `units` (raw), `packages` (= units / UPP), `carrier_packages` (= packages × carrier_share)
- **Derived**: `upp`, `carrier_share`, `carrier_cost`

### Target variable

For Notebook 1, there is no model yet — but the eventual forecast target is **`carrier_packages`** (weekly package volume per region × channel × carrier). That's what the forecasting team needs to plan capacity and validate carrier contracts.

### Business problem

Walmart's parcel team needs to predict **how many packages each carrier will ship in each region/channel each week**, so they can:

1. Hold carriers accountable to contractual minimums (e.g., FedEx Home Delivery commitments).
2. Catch over-allocation to capacity-constrained carriers (OnTrac Tier 2 thresholds).
3. Decide where shifting volume between carriers saves money.

A good forecast = millions saved on carrier penalties and unused contracted capacity.

---

## 2. Domain Reframing (M5 → ParcelCast)

The actual M5 data is retail sales. We **re-frame** it as a parcel-forecasting problem because that's the audience's domain:

| M5 concept | ParcelCast concept | Why this works |
|---|---|---|
| Store (CA_1, TX_2, …) | Fulfillment Center (FC) | Both are physical nodes in a network |
| State (CA, TX, WI) | Region (WEST, SOUTH, MIDWEST) | Both are geographic groupings |
| Category FOODS / HOUSEHOLD | Channel **1P** (first-party) | Walmart's own inventory |
| Category HOBBIES | Channel **WFS** (Walmart Fulfillment Services) | Third-party seller volume |
| `Unit Sales` | `Ordered Units` → divided by **UPP** = `Packages` | Sales convert into shipping demand |

This reframing is more than relabeling — it lets us tell a parcel story (carrier allocation, contract compliance, cost shift) using a real, public dataset.

---

## 3. UPP — Units Per Package

### The core formula

```
packages = units / UPP
```

If a customer orders 6 units of FOODS at UPP = 2.10, that ships as 6 / 2.10 ≈ 2.86 packages on average.

### Why this matters

The forecasting team doesn't care about units sold — they care about **how many boxes have to move through the network**. Two customers each buying 1 unit = 2 packages. One customer buying 2 units = 1 package. Same units, very different parcel volume. So UPP is the bridge between the retail signal and the operational signal.

### The values in this project

| Channel | UPP baseline | Behavior |
|---|---|---|
| **1P** (Walmart's own inventory) | **2.10** | **Time-varying** — declines ~3% per year |
| **WFS** (third-party sellers) | **1.33** | **Static** |

Why these specific numbers:

- **1P > WFS** because Walmart bundles items into shared shipments when a customer orders multiple things. Third-party sellers each ship independently.
- **WFS closer to 1.0** because each WFS seller fulfills their own orders, so most packages contain just one item.

### The "key insight" — 1P UPP declines

1P UPP is **declining over time**. Each year, customers order fewer items per package — partly because basket sizes are shrinking, partly because faster fulfillment means smaller, more frequent shipments. The code models this as a 3% annual decline:

```python
years_elapsed = (week_start - anchor_date).days / 365.25
decline_factor = (1 - 0.03) ** years_elapsed
upp = 2.10 * decline_factor
```

A 1P UPP that started at 2.10 in 2011 drops to ~1.80 by 2016. **This matters because UPP forecast error compounds into package forecast error**: if you assume UPP stays flat at 2.10 but it's actually dropped to 1.80, you'll under-forecast packages by about 17%. That gap is the difference between adequate carrier capacity and a service failure during peak season.

---

## 4. Data Cleaning & Preprocessing

### The reshape (wide → long)

The first transform isn't strictly cleaning, but it's the foundation: `pd.melt` collapses 1,941 day-columns into two columns (`d`, `units`). One row of the input (a SKU) becomes 1,941 rows of the output. The DataFrame goes from 30,490 × 1,947 → 59M × 8 rows.

**Memory optimization** — after melt, cast id columns to `category` dtype and downcast `units` to `uint16`. Why? Object/string columns store every value in full; categorical stores each unique value once and uses a small int code per row. For ~30K unique IDs replicated 1,941 times, this collapses memory ~11× (7.84 GB → 0.71 GB). On a low-RAM machine, this is the difference between "runs in 30s" and "swaps for 2+ minutes."

### Missing value handling

M5 happens to have no missing values in the cleaned aggregate, **but the pipeline supports them**:

- **`interpolate_missing`** uses `groupby().transform(interpolate)` — linear interpolation **within each (region, channel, carrier) group**, not globally.
- **Why grouped, not global?** A missing week for FedEx-WEST should be filled from FedEx-WEST's neighboring weeks, not from a UPS-MIDWEST average. Mixing groups would leak information across series.
- **Why linear interpolation, not mean fill?** Mean fill destroys the temporal pattern (turns a missing September week into a flat year-average value, hiding seasonality). Linear interpolation preserves the local trend.

### Duplicate removal

Not done explicitly because the construction of the pipeline (groupby aggregation) makes duplicates impossible — but the **validation suite** confirms it after the fact: check #8 (`no duplicate (region, channel, carrier, week)`) verifies the assumption holds.

### Outlier treatment — quick summary (full deep dive in §6)

Two-step approach:

1. **Detection**: `detect_outliers_iqr` computes IQR (Q3 − Q1), flags values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. Tukey's classic rule.
2. **Treatment**: `winsorize_with_log` caps the bottom 1% and top 1% at the percentile boundaries.

**Why winsorize, not delete?** A foundational rule in this project: **never delete rows**. For time series, deleting a week creates a gap that breaks lag features, seasonality models, and validation checks. Winsorization keeps the row, just clips the magnitude.

### Data transformations in sequence

1. **Time aggregation**: daily → weekly via `wm_yr_wk` (Saturday-to-Friday Walmart fiscal weeks). Done by groupby-sum.
2. **Hierarchy mapping**: `state_id` → `region`, `cat_id` → `channel` via dict-`.map()`.
3. **Units → Packages**: `packages = units / UPP` — and crucially, UPP for the 1P channel is **time-varying** (3% annual decline), while WFS is static.
4. **Packages → Carrier volumes**: `carrier_packages = packages × carrier_share`, where `carrier_share` depends on region (OnTrac is heavily WEST, etc.). One row becomes 4 rows (one per carrier).
5. **Share normalization**: `normalize_shares` divides each share by the group sum so the four carriers add to exactly 1.0 — defending against floating-point drift.

### Feature encoding / scaling

- **Categorical encoding**: applied for memory, not for modeling — pandas `category` dtype (integer codes under the hood). No one-hot encoding yet (that's a modeling concern).
- **Scaling**: not done in Notebook 1. Tree-based models (LightGBM, used later) don't need scaling. Prophet/ARIMA model the raw scale.

### Assumptions & decisions (explicit, documented)

| Assumption | Why it's defensible |
|---|---|
| HOBBIES → WFS, FOODS+HOUSEHOLD → 1P | Documented, stated as illustrative |
| UPP = 2.10 (1P), 1.33 (WFS) | Plausible; matches WW14-style numbers |
| 1P UPP declines 3%/year | Matches the WW14 narrative on channel mix shift |
| Carrier shares 55/15/20/10 + regional adjustments | Illustrative; the *structure* of regional variation matters more than the exact %s |
| OnTrac is West-Coast-heavy | Real-world fact about OnTrac's footprint |
| Winsorize at 1%/99% | Standard mild-winsorization |

### Audit log + validation suite

The two artifacts that turn this from "I cleaned the data" into "I can prove I cleaned the data correctly":

- **`CleaningAuditLog`** — records every transformation with timestamp, action, column, reason, before/after stats. Saved as `data/cleaning_audit_log.csv`.
- **`run_validation_suite`** — 8 automated checks: no missing/negatives, shares sum to 1, carrier packages reconcile, no week gaps, UPP in plausible range, hierarchy coverage, no duplicates. Run *after* cleaning and *must all pass*.

This pairing — **mutate + audit, then independently verify** — is how you make data pipelines trustworthy.

---

## 5. ML / EDA Concepts in This Notebook

Notebook 1 is "data science before the model." The concepts here are EDA, profiling, and time-series characterization. Forecasting models come in Notebook 2.

### 5.1 Data profiling

- **What**: For every column, compute dtype, missing %, unique count, min/max/mean/std/zero count/negative count.
- **Why used**: Before you trust a dataset, you need a one-page summary that catches obvious bugs.
- **How it works**: Loop over columns, branch on numeric vs categorical, compute statistics. Output one row per column.
- **Real-world importance**: Profiling is the cheapest defense against "garbage in, garbage out."
- **Pros**: Cheap, automatable, catches whole categories of bugs.
- **Cons**: Doesn't catch *correctness* bugs (e.g., units off by 1000×). For that, you need validation rules anchored in business meaning.

### 5.2 IQR outlier detection (Tukey's rule)

- **What**: A value is an outlier if it lies outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.
- **Why used**: Distribution-free (no Gaussian assumption), robust to extreme values, standard.
- **How it works**: Sort data, find Q1 (25th pctile) and Q3 (75th), compute IQR = Q3 − Q1. The "fence" is 1.5 IQR beyond each quartile.
- **Real-world importance**: For demand forecasting, outliers are usually either (a) data errors or (b) real events. Detection ≠ removal.
- **Pros**: Robust, no distributional assumption, easy to explain.
- **Cons**: For skewed distributions, the upper fence may be overly aggressive — leading to false positives.

### 5.3 Winsorization

- **What**: Cap (don't drop) values beyond chosen percentiles.
- **Why used**: For time series, **deleting an outlier creates a hole** that breaks lag features and decompositions.
- **How it works**: `s.clip(lower=Q01, upper=Q99)` — values above Q99 become exactly Q99.
- **Real-world importance**: Used heavily in finance, insurance, and any model where extreme tails distort fit but you can't justify removing the data point.
- **Pros**: Preserves row count, reduces influence of extremes, reversible if logged.
- **Cons**: Biases the distribution toward the center; if extremes are real, you lose signal.

### 5.4 Group-wise interpolation

- **What**: Fill missing values by linear interpolation, but only **within natural groups** (here: per region × channel × carrier).
- **Why used**: A missing FedEx-WEST week should be inferred from neighboring FedEx-WEST weeks, not from a UPS-MIDWEST average.
- **How it works**: `df.groupby(group_cols)[col].transform(lambda x: x.interpolate(method="linear"))`. The `transform` is key — it returns a Series the same length as the input.
- **Real-world importance**: The single most common imputation pattern for hierarchical time series.
- **Pros**: Preserves local trend and seasonality.
- **Cons**: For long gaps you're effectively making up data — always flag in an audit log.

### 5.5 Time-series decomposition

- **What**: Split a time series into **trend + seasonality + residual** components. The notebook uses `seasonal_decompose` from statsmodels with `model="multiplicative"` and `period=52` (weekly data, yearly seasonality).
- **Why used**: To **understand** the series before modeling. If trend is dominant → differencing. Strong seasonality → seasonal terms. Random residuals → decomposition captured the structure.
- **How it works conceptually**:
  - Compute trend via a moving average (window = period).
  - Subtract (additive) or divide out (multiplicative) trend → de-trended series.
  - Average the de-trended series at each seasonal index → seasonal component.
  - What's left = residual.
- **Additive vs multiplicative**:
  - **Additive**: `y = trend + seasonal + residual` — when seasonal swings are roughly constant size.
  - **Multiplicative**: `y = trend × seasonal × residual` — when seasonal swings *grow* with the level (retail!).
- **Pros**: Visual, intuitive, identifies model needs.
- **Cons**: Sensitive to chosen period and noise at series boundaries.

### 5.6 Augmented Dickey-Fuller (ADF) test

- **What**: A statistical test for **stationarity** — whether a series's mean and variance are constant over time.
- **Why used**: ARIMA-family models assume stationarity. ADF tells you whether you need to difference first.
- **How it works conceptually**: Tests the null hypothesis "the series has a unit root" (= non-stationary) vs the alternative "stationary." Low p-value (< 0.05) → reject null → stationary.
- **Pros**: Standard, widely understood, easy to interpret p-value.
- **Cons**: Low power on short series; structural breaks can yield misleading p-values.

### 5.7 Correlation matrix

- **What**: Pairwise Pearson correlations across selected features. Visualized as a heatmap.
- **Why used**: Quick scan for (a) features that are nearly redundant, (b) features that have predictive promise, (c) sanity checks (`packages` should correlate ~1.0 with `units`).
- **How it works**: Pearson r = `cov(X, Y) / (σ_X σ_Y)`, ranges from -1 to +1, measures **linear** association.
- **Pros**: Fast, visual, model-agnostic.
- **Cons**: Pearson only catches linear relationships. For nonlinear screening, use Spearman or mutual information.

### 5.8 The audit log + validation suite paradigm

This isn't an algorithm — it's a **discipline pattern** worth knowing by name:

- **Audit log**: every transformation logs `before` and `after` stats so you can reproduce or revert.
- **Validation suite**: a fixed battery of post-condition checks that must pass after every pipeline run.

Search terms to know: **data contracts**, **assertion-driven pipelines**, **Great Expectations**, **dbt tests**.

---

## 6. Outlier Treatment — Deep Dive

### 6.1 What outliers are

An **outlier** is a data point that lies far from the bulk of the other points. "Far" is fuzzy — that's why we have multiple definitions (IQR-based, z-score-based, model-based) and they often disagree.

**Concrete example.** A team's daily orders for two weeks:

```
12, 14, 13, 15, 14, 16, 15, 13, 14, 15, 350, 14, 13, 15
```

Most values are 12–16. The **350** sticks out. Almost certainly either (a) a data error, (b) a real but unusual event, or (c) a different *kind* of day. **Detection is mechanical. Treatment is judgment.**

### 6.2 Why outliers matter

Outliers warp three things:

1. **Summary statistics** — especially mean, standard deviation, and Pearson correlation.
2. **Model fits** — particularly linear regression and any model that minimizes squared error.
3. **Decisions made downstream** — capacity plans, alerts, contracts.

| Statistic | With the 350 | Without the 350 |
|---|---|---|
| Mean | **38.07** | **14.08** |
| Median | 14 | 14 |
| Std dev | 89.6 | 1.16 |

The mean **triples** because of one bad value. The median doesn't budge. **The mean is non-robust, the median is robust.**

### 6.3 How outliers hurt different models

| Model family | Effect of outliers |
|---|---|
| Linear regression / OLS | Severe — squared error pulls the fitted line toward outliers |
| Logistic regression | Less severe (sigmoid bounds the loss) |
| K-nearest neighbors | Outliers become "ghost neighbors" |
| K-means clustering | Cluster centroids drift toward outliers |
| Neural networks | Gradients spike on outlier batches; often need gradient clipping |
| Decision trees / Random Forest / **LightGBM** | **Highly robust** — splits depend on order, not magnitude |
| Naive Bayes | Affected through the assumed Gaussian |

**Implication for this project**: LightGBM (Notebook 2) is robust to outliers — but Prophet and the moving-average baseline aren't. So winsorization in Notebook 1 protects the non-tree models.

### 6.4 Quartiles (Q1, Q2, Q3)

Sort the data, then split into 4 equal-sized chunks:

- **Q1** (25th percentile): 25% of values are below it
- **Q2** (50th percentile, the median): 50% below
- **Q3** (75th percentile): 75% below

**Worked example.** Twelve sorted values:

```
2, 4, 5, 7, 8, 9, 10, 12, 14, 16, 18, 25
```

- Q1 ≈ **6** (between 5 and 7)
- Q2 ≈ **9.5** (between 9 and 10)
- Q3 ≈ **15** (between 14 and 16)

### 6.5 IQR (Interquartile Range)

```
IQR = Q3 − Q1
```

For our example: IQR = 15 − 6 = **9**.

The IQR is **the spread of the middle 50% of the data**. It ignores the extremes by construction, so it's a robust measure of dispersion.

### 6.6 Lower bound and upper bound (Tukey's rule)

```
lower bound = Q1 − 1.5 × IQR
upper bound = Q3 + 1.5 × IQR
```

For our example: `lower = 6 − 1.5×9 = −7.5`, `upper = 15 + 1.5×9 = 28.5`.

Anything outside `[−7.5, 28.5]` is flagged. In our 12 values, nothing crosses — even the 25 is inside. Tukey's rule is **deliberately conservative**.

**Why 1.5?** Under a normal distribution, IQR ≈ 1.35σ, so 1.5×IQR ≈ 2σ on each side of the quartiles → fences at roughly ±2.7σ from the mean → covers about 99.3% of normal data.

You can adjust k:

- `k = 1.5` → "mild" outliers (default)
- `k = 3.0` → "extreme" outliers (more conservative)

### 6.7 Visual intuition (boxplot)

A box plot is just a visual IQR rule:

```
              Q1     Q2     Q3
              │      │      │
   ─ ─ ─ ─ ─ ┌──────┬───────┐ ─ ─ ─ ─ ─    .  ←  outlier
              │      │       │
   ─ ─ ─ ─ ─ └──────┴───────┘ ─ ─ ─ ─ ─
   ↑                              ↑
   lower fence                    upper fence
   (Q1 − 1.5·IQR)                (Q3 + 1.5·IQR)
```

### 6.8 Z-Score method

```
z = (x − μ) / σ
```

Where μ = mean, σ = standard deviation. Z tells you how many standard deviations a point is from the mean.

**Common rules**:

- |z| > 2 → unusual (~5% of normal data)
- |z| > 3 → outlier (~0.3% of normal data)

**Worked example.** Values: `[5, 6, 7, 8, 9, 10, 11, 12, 13, 50]`

- μ = 13.1, σ ≈ 13.0
- z(50) = (50 − 13.1) / 13.0 ≈ **2.84** — close to flagged but not quite at 3.

The outlier itself dragged the mean and σ up — so it *barely* doesn't get flagged because it's hiding in the inflated σ it created. This is called **swamping** / **masking**.

The robust version uses median and MAD:

```
modified z = 0.6745 × (x − median) / MAD
```

For the same data: median = 9.5, MAD ≈ 2.5 → modified z(50) ≈ **10.9**. Now the outlier screams.

**When z-score is appropriate**:

- ✅ Data is approximately **normally distributed**
- ❌ Data is skewed (most counts, prices, durations)
- ❌ Sample size is small (σ is unstable)

For most real-world business data — which is skewed — **IQR is usually safer than z-score**.

### 6.9 Winsorization (deep dive)

**What it is**: Cap values beyond chosen percentiles instead of removing them.

```
x_winsorized = clip(x, lower = pct(x, 1%), upper = pct(x, 99%))
```

**Worked example.** Values: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 100]`. Apply 10%/90% winsorization.

- 10th percentile ≈ **1.9**, 90th percentile ≈ **18.1**
- After winsorization: `[1.9, 2, 3, 4, 5, 6, 7, 8, 9, 18.1]`
- The 100 became 18.1 — capped, not deleted.

**Pros / Cons**:

| ✅ Pros | ❌ Cons |
|---|---|
| Preserves row count | Biases distribution toward the center |
| Reduces influence of extremes on mean/std | Throws away genuine information at the tails |
| Reversible if you log original values | Choice of percentile is arbitrary |
| Works on any distribution | Won't help if outliers are *inside* the cap range |

### 6.10 Trimming (removing outliers)

**What it is**: Drop the rows where the value is an outlier.

```python
clean = df[(df["x"] >= lower) & (df["x"] <= upper)]
```

**When trimming is appropriate**:

- ✅ Cross-sectional data (each row is independent)
- ✅ You have plenty of data and dropping <1% won't bias estimates
- ✅ The outlier is **provably an error** (negative age, prices in wrong currency)
- ❌ Time series — leaves gaps that break lag/decomposition
- ❌ Hierarchical data where row counts matter for joins
- ❌ Small datasets

**Pros / Cons**:

| ✅ Pros | ❌ Cons |
|---|---|
| Simplest to implement | Loses data |
| Statistics you compute downstream are clean | Can introduce **selection bias** |
| Removes provably-bad data entirely | Breaks time-series alignment |

### 6.11 Capping / Flooring (rule-based)

Same family as winsorization, but with **business-rule** thresholds instead of percentile-derived ones.

```python
df["age"] = df["age"].clip(lower=0, upper=120)        # business rule
df["return_rate"] = df["return_rate"].clip(0.0, 1.0)  # mathematical bound
```

**Use when there's an obvious physical or business constraint** (ages can't be negative, percentages can't exceed 100%, prices can't be below cost). It's not really "outlier treatment" — it's enforcing a contract.

The difference vs winsorization:

- **Winsorization** → cap at percentiles derived from the data (data-driven)
- **Capping/flooring** → cap at hand-picked thresholds (rule-driven)

In practice you often do both: cap at hard business bounds first, then winsorize the residual extremes.

### 6.12 When to keep vs remove outliers

| Situation | Recommended action | Why |
|---|---|---|
| Provable data error (negative price, future date) | **Remove or fix** | Not real data; keeping it pollutes everything |
| Real but rare event (Black Friday, viral product) | **Keep** (or model separately) | Removing tells the model "Black Friday won't happen" → catastrophic at next Black Friday |
| Skewed business metric where extremes are normal (income, claim size) | **Keep + use robust models** | Distribution is genuinely heavy-tailed |
| Time series with rare spikes | **Winsorize** | Preserve continuity, reduce influence on mean-based models |
| Cross-sectional with rare outliers and lots of data | **Trim** if confirmed errors, **keep** if real | Both cheap; pick based on cause |
| Small dataset | **Keep, model robustly** | You can't afford to throw away rows |
| Not sure whether they're errors or real | **Keep, log, flag** | Build a feature `is_outlier=True` and let the model decide |

**The single most important rule**: never delete data silently. If you must remove or cap, log it.

---

## 7. Python Implementation Patterns

### With pandas only

```python
import pandas as pd

# IQR-based detection
q1, q3 = df["x"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
outlier_mask = (df["x"] < lower) | (df["x"] > upper)

# Trimming
clean = df[~outlier_mask]

# Winsorization (percentile-based)
lo, hi = df["x"].quantile([0.01, 0.99])
df["x_wins"] = df["x"].clip(lower=lo, upper=hi)

# Hard capping (rule-based)
df["age"] = df["age"].clip(lower=0, upper=120)

# Z-score detection
from scipy.stats import zscore
df["z"] = zscore(df["x"])
outlier_mask = df["z"].abs() > 3
```

### With scipy

```python
from scipy.stats.mstats import winsorize
arr = winsorize(df["x"].to_numpy(), limits=[0.01, 0.01])
```

`limits=[0.01, 0.01]` = winsorize bottom 1% and top 1%.

### With scikit-learn (model-based detection)

For multivariate outliers — rows where individual columns look fine but the *combination* is suspicious:

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=42)
df["is_outlier"] = iso.fit_predict(df[feature_cols]) == -1

from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
df["is_outlier"] = lof.fit_predict(df[feature_cols]) == -1
```

- **Isolation Forest** isolates points by random splits — outliers get isolated faster, after fewer splits.
- **LOF** compares each point's local density to its neighbors' — if a point has way fewer neighbors than its neighbors do, it's an outlier.

### Production pattern (mirrors this project)

```python
def detect_outliers_iqr(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "lower": q1 - k*iqr,
        "upper": q3 + k*iqr,
        "n_low":  int((s < q1 - k*iqr).sum()),
        "n_high": int((s > q3 + k*iqr).sum()),
    }

def winsorize_with_log(df, col, audit, lo_pct=0.01, hi_pct=0.99, reason=""):
    lo, hi = df[col].quantile([lo_pct, hi_pct])
    before = df[col].agg(["min", "max", "mean"]).to_dict()
    df = df.copy()
    df[col] = df[col].clip(lo, hi)
    after = df[col].agg(["min", "max", "mean"]).to_dict()
    audit.record("winsorize", col, reason, before, after)
    return df
```

Detect → log → transform → log again. Reproducible.

---

## 8. Interview Preparation

### 8.1 The 90-second project story

> "I built a parcel-volume forecasting prototype on Walmart's M5 dataset. The first notebook is end-to-end data quality: I load 5+ years of daily SKU-level sales, melt from wide to long format, aggregate to weekly fiscal periods aligned with Walmart's calendar, and then apply a domain transformation — units divided by Units-Per-Package (UPP) gives me parcel volume, and a regional carrier-share matrix splits that volume across FedEx, UPS, OnTrac, and an internal carrier. I treat 1P UPP as time-varying with a 3% annual decline because that matches a real observation from Walmart's WW14 weekly reporting.
>
> The cleaning is conservative — I never delete rows because that breaks the time-series continuity needed downstream. I winsorize extreme volumes at the 1st and 99th percentile, interpolate any missing values within (region, channel, carrier) groups so I don't leak information across series, and renormalize carrier shares so they sum exactly to 1.0. Every transformation writes to an audit log, and after cleaning I run an 8-check validation suite — no missing, no negatives, shares sum to 1, no week gaps, hierarchy coverage, no duplicates — and they all pass.
>
> EDA-wise, I do a multiplicative seasonal decomposition with a 52-week period to confirm yearly seasonality, run an ADF test to check stationarity for ARIMA-style modeling, and a correlation heatmap as a feature-triage step. The output of this notebook is two parquet files (network and region-channel-week) plus the audit log and validation results — those become the input to the modeling notebook."

### 8.2 Project-level Q&A

**Q: Why didn't you just drop rows with missing or outlier values?**
A: Because this is time-series data, and gaps break everything downstream — lag features, seasonality decompositions, week-over-week comparisons. Dropping is convenient for cross-sectional ML but destructive here. Winsorization caps the magnitude without losing the row, and interpolation fills gaps using local context within the same series. Both transformations get logged so they're reversible if a reviewer disagrees.

**Q: Why categorical dtypes for the id columns?**
A: Pure memory and downstream speed. After melting, the DataFrame has 59M rows with seven object-typed id columns — that's about 7.8 GB and forces low-RAM machines into swap. Casting to `category` collapses ~30K unique IDs to integer codes once and stores small references everywhere else. Memory drops to about 700 MB. No behavior change — categoricals support equality, merge, groupby, and `.map()` identically.

**Q: Walk me through how you'd validate that your cleaning didn't introduce bugs.**
A: Two layers. First, the audit log captures before/after stats for every transformation. Second, an automated validation suite runs 8 invariants after cleaning — no missing, no negatives, carrier shares sum to 1.0 per group, sum of carrier packages equals total packages per group, no week gaps, UPP is in the plausible 1.0–3.0 range, full hierarchy coverage, and no duplicate keys. The suite runs at the end of the cleaning notebook and is designed to fail loudly.

**Q: Multiplicative vs additive seasonal decomposition — when do you use which?**
A: Additive when seasonal swings are roughly constant in absolute size, multiplicative when they scale with the level of the series. Retail almost always wants multiplicative — a 20% Q4 lift means dramatically more units in absolute terms as the business grows. If I plotted residuals from an additive decomposition and saw their variance fanning out over time, that's a tell that I should be using multiplicative instead.

**Q: What does the ADF test tell you and why does it matter?**
A: It tests whether the series is stationary — constant mean and variance over time. The null hypothesis is "non-stationary, has a unit root." A p-value below 0.05 lets me reject the null. Important because Prophet and LightGBM are robust to non-stationarity, but classical ARIMA isn't.

**Q: Why fiscal weeks (Saturday-to-Friday) instead of ISO weeks?**
A: Because the audience operates on Walmart's fiscal calendar and reports weekly volumes that way. Using ISO weeks would mean my numbers don't reconcile against their dashboards.

**Q: What's Traditional Error and why use it instead of MAPE?**
A: Traditional Error = (Sum Forecast − Sum Actual) / Sum Actual × 100. It tells you whether you're over- or under-forecasting in aggregate. Standard MAPE explodes when actuals are small or zero, and it hides the *direction* of error. For capacity planning where over- and under-forecasting have asymmetric costs, the directional signal matters more.

### 8.3 Outlier-specific Q&A

**Q: What's an outlier and how do you decide if a value is one?**
A: An outlier is a value that's far from the bulk of the data — but "far" is defined by the rule you choose. Two common rules: the IQR rule (anything outside Q1−1.5·IQR or Q3+1.5·IQR) and the z-score rule (|z| > 3). I usually start with IQR because it doesn't assume normality, which most real business data violates. For multi-feature data where individual columns look fine but the combination is suspicious, I'd use Isolation Forest or LOF.

**Q: Why use IQR instead of z-score?**
A: IQR is robust — it uses quartiles, so the outliers themselves don't move the boundaries. Z-score uses mean and standard deviation, which both get inflated by the very outliers you're trying to detect, so an outlier can hide in its own σ. This is called masking. IQR also doesn't assume the data is normal.

**Q: When would you remove outliers vs keep them?**
A: Remove only when I can prove they're errors. Keep when they're real but rare events because removing them would tell the model "this never happens." For time series I'll usually winsorize instead of removing because deleting rows breaks lag features. And for hierarchical data I'll always log whatever I do.

**Q: What's winsorization and when do you use it?**
A: Winsorization caps values at chosen percentiles instead of removing them. For a 1%/99% winsorization, anything below the 1st percentile becomes exactly the 1st-percentile value, anything above the 99th becomes the 99th. The row stays in the dataset. I use it when the rows themselves matter — time series where deletion creates gaps, or hierarchical data where row counts feed downstream joins.

**Q: How do outliers affect different model families?**
A: Linear regression, k-NN, k-means, and neural networks are all sensitive — they minimize squared error or use Euclidean distance. Tree-based models like Random Forest and LightGBM are robust because they only care about the *order* of values, not their magnitude.

**Q: How would you detect outliers in a high-dimensional dataset where each individual column looks fine?**
A: Univariate methods like IQR and z-score won't catch combination outliers. I'd use Isolation Forest, which randomly splits the feature space — outliers get isolated quickly. Or Local Outlier Factor, which compares each point's local density to its neighbors'.

**Q: What's the difference between capping and winsorization?**
A: Mechanically they're the same operation — `clip(lower, upper)`. The difference is where the bounds come from. Winsorization uses **percentiles from the data**. Capping uses **business rules**.

**Q: Walk me through how you'd handle a column that's 5% missing and 2% outliers.**
A: First profile to understand whether the missing and outlier are independent or correlated. For missing values I'd use group-wise interpolation if it's time-series data, or a model-based imputer like KNNImputer or IterativeImputer if it's cross-sectional. For outliers I'd winsorize at 1%/99% rather than delete — and I'd log both transformations.

**Q: What does "robust statistic" mean?**
A: A statistic is robust if it doesn't change much when you add or remove a small number of extreme values. Median is robust because it depends on the middle position, not the magnitudes. Mean is non-robust. The whole field of robust statistics is built around designing estimators that perform well even when assumptions like normality are violated.

**Q: You detect outliers but you're not sure whether they're errors or real. What do you do?**
A: Don't delete them silently. I'd log them, build a feature like `is_outlier=True`, and let the model use that signal. If the model performs better with the flag, the outliers contained real signal. The worst thing I can do is silently drop them.

### 8.4 60-second outlier summary

> "An outlier is a value far from the rest of the data. The two simplest detection methods are IQR — anything outside Q1 minus 1.5 IQR or Q3 plus 1.5 IQR — and z-score, anything more than 3 standard deviations from the mean. IQR is preferred for skewed or non-normal data because it uses quartiles that aren't affected by extremes themselves. For treatment, I think of three options: remove, winsorize, or keep. Remove only for confirmed errors. Winsorize when the row matters — time series, hierarchical data — because it caps the magnitude without losing the row. Keep when the outlier is real signal, and pair it with a model that handles extremes well, like LightGBM. The single rule I follow is to never modify data silently — every cleaning decision goes into an audit log."

### 8.5 Three principles to anchor everything

1. **Robust beats sensitive when you don't know the distribution.** Median and IQR > mean and σ for most messy real-world data.
2. **The treatment depends on the cause.** Errors → remove. Rare-but-real → keep. Mixed → flag and let the model decide.
3. **Audit everything.** A reversible decision is a safe decision. A silent one is a bug waiting to happen.

---

## 9. Key Terms Glossary

| Term | One-line definition |
|---|---|
| Wide vs long format | Wide = one row per entity, columns are observations. Long = one row per (entity, observation). |
| `pd.melt` | Pandas operation that converts wide → long. |
| Categorical dtype | pandas dtype storing unique values once + integer codes per row. Saves memory, speeds equality. |
| Hierarchical time series | Series organized into a tree (network → region → channel → carrier). |
| Fiscal week | Business calendar week (Walmart: Sat–Fri); often differs from ISO week. |
| UPP (Units Per Package) | Conversion ratio: packages = units / UPP. |
| Outlier | A data point far from the bulk of the data (definition depends on the rule). |
| Quartile (Q1, Q2, Q3) | The 25th, 50th, 75th percentile cutpoints when sorting data. |
| IQR (Interquartile Range) | Q3 − Q1; the spread of the middle 50% of data. |
| Tukey's fence | `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` — the IQR outlier rule. |
| Z-score | `(x − μ) / σ` — standard deviations from the mean. |
| Modified z-score | Robust version using median and MAD instead of mean and σ. |
| MAD (Median Absolute Deviation) | Median of `|x − median(x)|`. Robust spread measure. |
| Masking / Swamping | Outliers inflating σ such that they hide in their own statistics. |
| Winsorization | Cap values at chosen percentiles instead of removing them. |
| Trimming | Removing outlier rows entirely. |
| Capping / Flooring | Clipping at hand-picked business-rule thresholds. |
| Linear interpolation | Estimate missing value as a straight line between neighbors. |
| `groupby().transform()` | Apply a function within groups, return result aligned to the original index. |
| Stationarity | Time series with constant mean/variance over time. |
| ADF test | Augmented Dickey-Fuller — tests for unit root (non-stationarity). |
| Seasonal decomposition | Split series into trend + seasonality + residual. |
| Additive vs multiplicative | Whether components combine by + (constant amplitude) or × (proportional). |
| Pearson correlation | Linear association between two variables, [-1, 1]. |
| Robust statistic | Resists distortion by extreme values (median, IQR, MAD). |
| Isolation Forest | Random-split tree ensemble for multivariate outlier detection. |
| Local Outlier Factor (LOF) | Density-based multivariate outlier detection. |
| Audit log | Record of every transformation with before/after metrics. |
| Validation suite | Automated post-conditions a cleaned dataset must satisfy. |
| Data contracts | Formal expectations between data producer and consumer. |
| Reproducibility | Same input + same code → same output, every time. |
| Traditional Error | (Sum Forecast − Sum Actual) / Sum Actual × 100. Walmart's primary metric. |
| WMAPE | Sum(|err|) / Sum(actual). Walmart's secondary metric. |

---

## 10. Memorable Analogies

- **Wide → long**: Wide format is a class roster (one row per kid, columns per assignment). Long format is a gradebook (one row per "kid did assignment X on date Y"). Most ML expects the gradebook.
- **Categorical dtype**: Like a hotel guest list. Object/string = writing each guest's full name on every meal receipt. Categorical = assigning each guest a room number once and writing the room number on receipts.
- **Winsorize vs delete**: A photo where one pixel is blown out. Deleting = cutting a hole in the photo. Winsorizing = darkening that one pixel to the brightest "normal" value. The picture stays whole.
- **Group-wise interpolation**: Filling in a missing weather reading for Seattle by averaging Seattle's other days, not by averaging Seattle and Phoenix together.
- **Seasonal decomposition**: Like splitting your salary into "base pay" (trend) + "annual bonus pattern" (seasonality) + "unusual one-off comp" (residual). Each component reveals a different aspect.
- **ADF test**: Like asking "is the river's water level basically the same year-round?" — if yes, it's stationary; if it's been steadily rising, it's not.
- **Audit log + validation suite**: A surgeon's count of sponges before and after surgery. The audit log says "I used 14 sponges." The validation suite says "Count again — yes, 14 came out." Together they prevent bad outcomes.
- **Mean vs median for outliers**: Imagine 10 people in a room earning $50K each, then Bill Gates walks in. The mean income jumps to billions; the median barely moves. Mean = swayed by extremes. Median = robust.
- **IQR vs z-score**: IQR is like asking "is this person taller than most people in the middle of the line?" — answer doesn't depend on the giants at the back. Z-score is like asking "how far from average?" — but the giants at the back already shifted the average.
- **Masking effect**: A loud whisper near a microphone — it's so loud it triggers the auto-volume to lower, so the whisper itself sounds normal in the recording. The outlier hides in the dampening it caused.
- **Winsorization**: Like a salary cap in sports — the highest-paid players still play, but their reported salary is clipped at the cap. Real existence preserved, extreme magnitude tamed.
- **The whole notebook**: You're a chef tasting and labeling every ingredient before cooking starts. Notebook 2 is the cooking. If something tastes wrong in Notebook 2, you trust Notebook 1's labels and look at the recipe, not the ingredients.

---

*Last updated: 2026-05-09*