# Modeling & Forecasting — Learnings (Notebook 2)

> Consolidated notes from the walkthrough of `notebooks/02_modeling.py`.
> Written for both **practical understanding** and **interview preparation**.

---

## Table of Contents

1. [The Overall Objective](#1-the-overall-objective)
2. [The Three Models](#2-the-three-models)
3. [Statistical vs ML Models](#3-statistical-vs-ml-models)
4. [Forecasting & Evaluation Concepts](#4-forecasting--evaluation-concepts)
5. [WMAPE — Deep Dive](#5-wmape--deep-dive)
6. [Time-Series Concepts](#6-time-series-concepts)
7. [Feature Engineering for Forecasting](#7-feature-engineering-for-forecasting)
8. [Train/Test Split & Backtesting](#8-traintest-split--backtesting)
9. [Code Flow / Implementation Logic](#9-code-flow--implementation-logic)
10. [Interview Preparation](#10-interview-preparation)
11. [Key Terms Glossary](#11-key-terms-glossary)
12. [Memorable Analogies](#12-memorable-analogies)

---

## 1. The Overall Objective

### What business problem we're solving

A major retailer's parcel-forecasting team needs to know: **"How many packages will move through the network next week, the week after, and the four weeks after that — broken down by region, channel, and carrier?"**

This forecast directly drives:

| Decision | Why the forecast matters |
|---|---|
| **Carrier capacity contracts** | Forecast 1M for FedEx HD but only 700K materialize → wasted contracted capacity. Forecast 700K and 1M arrive → service falls over. |
| **Tier compliance (OnTrac, etc.)** | Pre-empt over-allocation before SLAs break. |
| **Cost optimization** | "If I shift 5% from FedEx to IP, what's the cost delta?" |
| **Workforce planning at FCs** | Package volume = labor hours. Off by 10% = wrong staffing. |

A 1 percentage point accuracy improvement at network scale = millions of dollars per quarter.

### What "forecasting" / "modeling" means here

**Forecasting** = predicting future values of a time-ordered series, using its history (and optionally other signals) as input.

```
INPUT:  weekly package volume from 2011-01-29 through some cutoff date
OUTPUT: predicted package volume for the next 12 weeks (the test set)
```

**Modeling** = the broader activity: choosing a mathematical structure (exponential smoothing? gradient boosting? neural net?), fitting it to history, using the fitted object to generate predictions.

### Why companies need forecasting

Three orthogonal reasons:

1. **Capacity decisions are made before demand is observed.** Carrier contracts in October for January's volume. Seasonal labor in September.
2. **Asymmetric costs.** Over- vs under-stocking have different cost structures. A forecast lets you optimize against the asymmetry.
3. **Coordination.** Marketing, finance, ops, supply chain all plan around one shared number — the forecast.

### What we're predicting

In Notebook 2: **target = `packages`** — weekly package volume aggregated to network level. Notebook 3 will extend to per-region-channel and per-carrier breakdowns.

A single numeric value per week for 12 weeks (a "point forecast"), with Prophet additionally producing an 80% confidence interval.

### Why forecasting is critical for retail

Retail is **inherently seasonal, event-driven, and capital-intensive**:

- Q4 peak is 2–3× off-peak in package volume
- Black Friday alone moves 30%+ of holiday demand
- Warehouses, contracts, staff are committed weeks-to-months ahead

A 1% improvement in WMAPE on a $1B parcel spend is real money.

---

## 2. The Three Models

### Why three, not one

| Model | Role | "What it represents" |
|---|---|---|
| **MovingAverageBaseline (4-wk MA)** | Floor / sanity check | "What if you didn't model anything fancy at all?" |
| **Prophet** | Statistical forecasting model | The standard time-series-decomposition approach |
| **LightGBM** | Machine learning model | The modern global, feature-driven approach |

**Project rule**: every more-complex model must beat the baseline by a meaningful margin to justify its existence. Without that rule, you can hide poor-performing complex models behind impressive-sounding names.

### 2.1 The 4-Week Moving Average baseline

```python
forecast = mean(last_4_weeks_of_actuals)
```

That's the entire model. Outputs the same number for every future week.

**Why it exists**:

1. **Correctness floor.** If your fancy model can't beat this, you have a bug.
2. **Realistic effort bar.** Some problems don't need ML. Maybe ship the baseline.
3. **Honest comparison.** "1.55% WMAPE" means nothing without "3.96%" to anchor against.

**How it works conceptually**: assumes the next 4 weeks look like the previous 4 weeks. No trend, no seasonality, no events. Smooths single-week noise but reacts slowly to genuine shifts.

| ✅ Pros | ❌ Cons |
|---|---|
| Trivial (one line) | Cannot capture trend, seasonality, events |
| Zero hyperparameters | Wrong by definition during peak season |
| Always interpretable | Reacts to genuine shifts only after a 4-week lag |

### 2.2 Prophet

**What it is**: a forecasting library released by Facebook/Meta in 2017. A **decomposable additive (or multiplicative) model**:

```
y(t) = trend(t) × (1 + seasonality(t) + holidays(t)) × ε
```

**How it works conceptually** — three components, fit jointly:

1. **Trend**: piecewise linear function with automatic changepoints (where slope changes).
2. **Seasonality**: Fourier series (sums of sines and cosines) approximating yearly patterns.
3. **Holidays**: discrete bumps at specified dates. We pass `US_RETAIL_HOLIDAYS` — Black Friday, Cyber Monday, Christmas, Memorial Day, July 4th, Labor Day, BTS_peak — with `lower_window=-3, upper_window=3` (effect spans 3 days before through 3 days after).

Fit using Bayesian inference (Stan under the hood). That's why Prophet gives uncertainty intervals "for free."

**Why multiplicative seasonality?** Retail seasonal swings **scale with the level**. A 20% Black Friday lift in 2016 moves more absolute units than the same percentage lift in 2011. Additive seasonality assumes the *absolute* swing is constant — wrong for growing businesses.

**Why `weekly_seasonality=False`?** Our data is already weekly. Within-week patterns don't exist when each observation IS one week.

| ✅ Pros | ❌ Cons |
|---|---|
| Built-in seasonality, holidays, changepoints | Slow to fit (Bayesian sampling) |
| Sensible defaults | Limited to additive/multiplicative decomposition |
| Native uncertainty intervals | Can't easily add exogenous features without hacks |
| Robust to missing dates and outliers | Per-series — no information sharing |
| Interpretable named components | Not always more accurate than simpler methods |

In our scorecard, Prophet ranked **3rd** by WMAPE (5.88%) — beaten by both LightGBM and the simple baseline.

### 2.3 LightGBM

**What it is**: Light Gradient Boosting Machine — Microsoft's gradient-boosted decision tree library. Frequently state-of-the-art for tabular data.

**How gradient boosting works conceptually**:

1. Start with a stupid first guess: the mean. Compute residuals (actual − guess).
2. Train a small decision tree to predict those residuals from the features. Add its predictions to the previous prediction with a small learning rate.
3. Compute new residuals. Train another small tree on those. Add.
4. Repeat 500 times (`n_estimators=500`).

Each tree is small (`num_leaves=31`) and learns a tiny bit. Together they form a powerful additive model that fits complex non-linear interactions.

**Why "global" model?** Trains **one model across all (region, channel) series simultaneously**. Alternative: 6 separate Prophet models, one per region-channel combo. Global wins:

1. **Cross-series learning** — a back-to-school bump in WEST/1P helps predict MIDWEST/1P.
2. **More data** — 6 series × 270 weeks ≈ 1,600 training rows vs ~270 per local model.
3. **Easier feature engineering** — lag/calendar/UPP features all in the same DataFrame.

**Tradeoff**: model has to learn region differences via `region` as a feature, instead of being trained separately by construction.

**Features it uses (~16)**:

- **Lags**: `lag_1, lag_2, lag_4, lag_8, lag_12, lag_52`
- **YoY growth**: `yoy_growth`
- **Calendar**: `year, month, quarter, fiscal_week`
- **Cyclical encoding**: `woy_sin, woy_cos`
- **Retail flags**: `is_peak, is_bts, is_holiday`
- **UPP signals**: `upp_lag_1, upp_lag_4`

| ✅ Pros | ❌ Cons |
|---|---|
| Often state-of-the-art for tabular | Doesn't extrapolate beyond training distribution |
| Handles non-linear interactions | Requires explicit feature engineering |
| Robust to outliers and missing values | Not naturally probabilistic |
| Fast to train and serve | Many hyperparameters to tune |
| Provides feature importance | Risk of leakage if features computed wrong |

Ranked **1st** at 1.55% WMAPE.

---

## 3. Statistical vs ML Models

| | Statistical (Prophet, ARIMA, ETS) | ML (LightGBM, XGBoost, neural nets) |
|---|---|---|
| **Mental model** | "Decompose into named components" | "Learn a function from features to target" |
| **Inputs** | Just the time series | A feature matrix (lags, calendar, exogenous) |
| **Seasonality captured by** | Built into model structure | Engineered as features (`fiscal_week`, `lag_52`) |
| **Multiple series** | One model per series | One global model across all |
| **Uncertainty** | Native (Bayesian or analytic intervals) | Requires extra work (quantile regression, conformal) |
| **Interpretability** | Components are named | Feature importance + SHAP |
| **When it shines** | Few series, clean seasonality, interpretable | Many series, rich exogenous features, pure accuracy |

Best forecasting teams use both — they're complementary, not competitive.

---

## 4. Forecasting & Evaluation Concepts

### 4.1 Forecast vs Actuals

The simplest sanity check: plot the predicted line on top of the actual line. [05_forecast_vs_actuals.png](../presentation/05_forecast_vs_actuals.png):

- **Black** (actuals) — the truth
- **Steel blue** (LightGBM) — tracks tightly
- **Orange** (Prophet) — biased low, with shaded 80% CI
- **Dashed dark slate** (baseline) — flat line

**Why this plot is the most-looked-at in any forecasting project**: numbers in a scorecard hide failure modes. A model with 5% WMAPE could be slightly off every week (good) or spot-on for 9 weeks then catastrophic on 3 (bad). The eye catches that pattern instantly. WMAPE doesn't.

### 4.2 Model scorecard

A small DataFrame comparing all models on the same test set on the same metrics:

| Model | Traditional Error % | WMAPE % | Rank |
|---|---:|---:|---:|
| LightGBM | −0.64 | 1.55 | 1 |
| Baseline (4wk MA) | −1.86 | 3.96 | 2 |
| Prophet | −5.56 | 5.88 | 3 |

**Why a scorecard, not just one number?**

1. Reproducibility — anyone can re-run and verify the ranking.
2. Conversation — stakeholders argue about which metric matters more.
3. Honesty — if your favorite model wins on only one metric, the scorecard makes that visible.

### 4.3 Accuracy metrics — the full landscape

| Metric | Formula | What it captures | When to use |
|---|---|---|---|
| **MAE** | `mean(|actual − pred|)` | Avg absolute error in original units | Errors equally costly regardless of size |
| **MSE** | `mean((actual − pred)²)` | Squared error (penalizes big misses) | Large errors disproportionately bad |
| **RMSE** | `sqrt(MSE)` | Same as MSE in original units | Same as MSE, easier to interpret |
| **MAPE** | `mean(|err| / |actual|) × 100` | Avg % error per point | Cross-series comparison; **breaks on zero actuals** |
| **WMAPE** | `sum(|err|) / sum(|actual|) × 100` | Weighted % error | **Robust to small/zero actuals**; this project's secondary |
| **Traditional Error** | `(sum(pred) − sum(actual)) / sum(actual) × 100` | Aggregate **bias** (direction matters) | Asymmetric costs; this project's primary |
| **MASE** | MAE / MAE_naive | Skill vs naive forecast | Cross-series, scale-robust |

### 4.4 Why this project uses Traditional Error + WMAPE

- **Traditional Error**: directional bias. Over-forecast = wasted capacity ($). Under-forecast = service failures (customer pain). **Direction matters.**
- **WMAPE**: point-by-point accuracy. Bounded, doesn't explode on small actuals, naturally weighted by volume.

Together: bias + accuracy. A model with 0% TE and 50% WMAPE = right on average but noisy. A model with 10% TE and 1% WMAPE = consistently off by 10% but predictable.

**Avoid plain MAPE**: blows up when actuals near zero.

### 4.5 Lag analysis

You don't always have the latest data when you need to forecast. Today is Week 30, data lake is 1 week behind → forecasting Week 31 from Week 29 = **lag-2 forecast**. Some processes need lag-4 forecasts (long planning cycles).

Our lag analysis:

| Lag | Traditional Error % | WMAPE % |
|---|---:|---:|
| 1 week | −4.04 | 4.88 |
| 2 weeks | −4.54 | 5.17 |
| 3 weeks | −4.82 | 5.24 |
| 4 weeks | −4.97 | 5.32 |

**Reading**: accuracy degrades monotonically with horizon (~0.44pt WMAPE worse from lag-1 to lag-4). Expected and matches WPR reporting format.

---

## 5. WMAPE — Deep Dive

### What it stands for

**Weighted Mean Absolute Percentage Error**. A forecast accuracy metric that tells you the total error as a percentage of the total actual volume.

### The formula

```
WMAPE = Sum(|actual - forecast|) / Sum(|actual|) × 100
```

Two sums, then a ratio, then a percent.

### Worked example

| Week | Actual | Forecast | \|Error\| |
|---|---:|---:|---:|
| 1 | 100 | 90  | 10 |
| 2 | 120 | 130 | 10 |
| 3 | 80  | 95  | 15 |
| 4 | 200 | 180 | 20 |
| **Sum** | **500** | — | **55** |

```
WMAPE = 55 / 500 × 100 = 11%
```

You missed by 55 packages out of 500 actual = 11% off in aggregate.

### Why it's "weighted"

Mathematically equivalent to a **volume-weighted average of per-point percentage errors**:

```
WMAPE = Σ( (|err_i|/actual_i) × (actual_i/Σactual_i) )
              ↑                       ↑
        point's % error    point's volume weight
```

Big weeks count more than quiet weeks. A 50% miss on a 5-package week barely moves WMAPE; a 50% miss on a 1,000-package week dominates it. **For business forecasting, this is exactly the right behavior.**

### Why WMAPE instead of MAPE

Standard **MAPE** = `mean(|err_i| / |actual_i|) × 100`

Two big problems:

1. **Blows up on small/zero actuals.** Actual=5, forecast=50 → 900% error on one point. One bad small-volume week wrecks the average.
2. **Treats every observation equally.** A 30% error on the busiest week of the year counts the same as a 30% error on a sleepy week.

WMAPE fixes both: denominator is the *total* (large, stable, can't blow up), and bigger weeks naturally count more.

### Interpreting values

| WMAPE | What it means in retail forecasting |
|---|---|
| **< 5%** | Excellent — close to noise floor of the data |
| **5–10%** | Good — typical for mature forecasting systems |
| **10–20%** | OK — defensible for hard problems |
| **> 20%** | Poor — usually a model or feature problem |

LightGBM at **1.55%** is in the "excellent" band.

### Why WMAPE pairs with Traditional Error

| Metric | Tells you | Sign sensitivity |
|---|---|---|
| **WMAPE** | How **noisy** are you? | No — uses absolute value |
| **Traditional Error** | How **biased** are you? | Yes — positive=over, negative=under |

**Bias and noise are independent failure modes**. A model can be unbiased but noisy, biased but predictable, or some mix.

### In code ([src/evaluation.py:33-43](../src/evaluation.py#L33-L43))

```python
def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom * 100
```

---

## 6. Time-Series Concepts

### Stationarity

A series is **stationary** if its statistical properties (mean, variance) don't change over time. Most retail series are *not* stationary — they trend up and have seasonality. Tested with ADF in Notebook 1.

### Trend

The long-term direction. The retailer's parcel volume trends up over time as e-commerce grows. Prophet models this as piecewise linear; LightGBM captures it implicitly via `lag_52`.

### Seasonality

Repeating patterns at fixed periods. **Yearly** seasonality dominates retail (Q4 peak, Q1 trough). **Weekly** seasonality is irrelevant here because we aggregated to weekly.

### Cyclic patterns

Patterns at non-fixed periods (business cycles, multi-year fluctuations). Distinct from seasonality because the period varies. We don't model these explicitly.

### Holidays / events

Discrete spikes at known dates. Prophet handles via `US_RETAIL_HOLIDAYS`. LightGBM handles via `is_holiday`, `is_peak`, `is_bts` flags.

---

## 7. Feature Engineering for Forecasting

Most ML models need a tabular feature matrix. The art is turning a time series into one.

### Lag features

```python
df["lag_1"] = df.groupby(["region", "channel"])["packages"].shift(1)
```

`lag_1` for each row is the value from the previous row **within the same region-channel group**. The `groupby` is critical — without it you'd shift across groups (catastrophic leakage).

Lag set: `[1, 2, 4, 8, 12, 52]` — short-term momentum (1, 2), monthly cycle (4), trend shifts (8, 12), yearly seasonality (52).

### YoY growth

```python
yoy_growth = (current_value - value_52_weeks_ago) / value_52_weeks_ago
```

Captures structural growth/decline that lag_52 alone doesn't.

### Calendar features

`year`, `month`, `quarter`, `fiscal_week` — categorical/ordinal time markers.

### Cyclical encoding

```python
woy_sin = sin(2π × week / 52)
woy_cos = cos(2π × week / 52)
```

**Why?** Because `fiscal_week=52` and `fiscal_week=1` are semantically adjacent (consecutive weeks), but a model treats `52` and `1` as numerically far apart. The sin/cos pair gives the model two features where week 52 and week 1 are very close in 2D space. This is how you tell a model "December 31 and January 1 are next to each other."

Same trick applies to hour-of-day, day-of-week, month — anything that cycles.

### Event flags

`is_peak`, `is_bts`, `is_holiday` — binary indicators for retail seasons.

### Lagged exogenous features

`upp_lag_1` and `upp_lag_4` — past UPP as a feature. **Lagged** (not current) to avoid leakage: at forecast time, you don't know the current week's UPP yet.

---

## 8. Train/Test Split & Backtesting

### Critical rule: no random splits for time series

Random splits give the model future information to predict the past — leakage.

**The right way**: chronological split. Train on first ~95% of weeks, test on last 12.

```python
train = network.iloc[:-TEST_WEEKS]
test = network.iloc[-TEST_WEEKS:]
```

Mirrors production behavior: you only know the past at the moment of prediction.

### Why beginners get this wrong

`sklearn.model_selection.train_test_split(shuffle=True)` is the default. For time series, use `shuffle=False` or build the split manually.

### Backtesting

**Backtesting** = simulating "what would the model have predicted if I'd run it last week, last month, last quarter?" by retraining on historical cutoffs.

Our `rolling_lag_forecast` function does exactly this:

```python
for i in range(n_total - n_test_weeks, n_total):
    train_end = i - lag + 1
    sub_train = series_df.iloc[:train_end]
    m = ProphetModel().fit(sub_train, ...)
    pred = m.predict(lag)[-1]
```

For each test week `i`, retrain on data ending `lag` weeks earlier, predict the final point. Gives **horizon-aware accuracy**.

### Why backtesting beats single train/test split

1. Reveals horizon dependency.
2. Averages out the "lucky test week" problem.
3. Mirrors production behavior more faithfully.

**Downside**: expensive — you retrain the model many times.

### Common pitfalls beginners make

- **Random shuffle on time series** — leakage.
- **Computing features over the full dataset before splitting** — leakage if any aggregate spans both.
- **Using current-week UPP as a feature** — leakage. Use `upp_lag_1`.
- **Reporting MAPE on data that includes zeros** — division by zero gives infinity.
- **Skipping the baseline** — you can't claim your model is "good" without one.
- **One-shot evaluation on a single test set** — use rolling backtests for horizon-aware metrics.

---

## 9. Code Flow / Implementation Logic

### Block A: Imports + load cleaned data

Loads `weekly_region_channel.parquet` (output of Notebook 1). Already cleaned, validated, has `upp` and `packages`.

### Block B: Aggregate to network level

Sum `packages` across (region, channel) per week → one number per week.

```python
network = (
    weekly.groupby("week_start", as_index=False)["packages"].sum()
    .sort_values("week_start")
)
```

### Block C: Train/test split

```python
TEST_WEEKS = 12
train = network.iloc[:-TEST_WEEKS]
test = network.iloc[-TEST_WEEKS:]
```

### Block D: Build feature matrix for LightGBM

`build_feature_set()` from `src/features.py` adds lag, calendar, UPP, YoY features. `get_feature_columns()` returns numeric non-target columns.

### Block E: Fit each model

```python
baseline = MovingAverageBaseline(window=4).fit(train)
prophet  = ProphetModel().fit(train)
lgbm     = LightGBMForecaster().fit(train_features, feature_cols)
```

### Block F: Generate predictions

12 predicted weeks per model. LightGBM predicts on the actual feature rows; Prophet predicts forward 12 steps; baseline returns the same number 12 times.

### Block G: Build the scorecard

`build_scorecard()` computes Traditional Error and WMAPE per model on the test set, ranks by WMAPE.

### Block H: Forecast vs Actuals chart

The four-line overlay plot.

### Block I: Feature importance

LightGBM's `feature_importance(importance_type="gain")` — total reduction in loss attributable to each feature.

### Block J: Lag analysis

Loop over lags 1–4, retrain Prophet at each cutoff, score, save WPR-format CSV.

### Block K: Save artifacts

`model_scorecard.csv`, `lag_analysis.csv`, `network_forecast.parquet`, three PNGs.

---

## 10. Interview Preparation

### 10.1 The 90-second story

> "Notebook 2 is the modeling layer of the parcel-forecasting prototype. I take the cleaned weekly package volumes from Notebook 1 — about 270 weeks aggregated to the network level — and I train three models: a 4-week moving-average baseline, Prophet with multiplicative seasonality and US retail holidays like Black Friday and Cyber Monday, and a global LightGBM model that uses lag features at 1, 2, 4, 8, 12, and 52 weeks plus calendar features, year-over-year growth, and lagged UPP as exogenous signals.
>
> The split is chronological — last 12 weeks held out — because random shuffles on time series cause leakage. I evaluate using the team's exact metrics: Traditional Error for aggregate bias and WMAPE for point-by-point accuracy. LightGBM wins at 1.55% WMAPE, beating the baseline by 2.4 points and Prophet by 4.3 points. Prophet is structurally pessimistic on this series.
>
> I also build a lag analysis that mirrors the WPR reporting format — for each forecast horizon from 1 to 4 weeks, I retrain Prophet at the corresponding historical cutoff and measure accuracy. The pattern is monotonic degradation, which is the expected and defensible behavior."

### 10.2 Q&A

**Q: Why do you use three models instead of one?**
A: Each plays a different role. The 4-week moving average is a floor that every more-complex model has to clear, otherwise complexity isn't earning its keep. Prophet is the standard time-series-decomposition approach that gives interpretable trend, seasonality, and holiday components, plus uncertainty intervals out of the box. LightGBM is a global ML model that can use exogenous features Prophet structurally can't, like lag features and the project's UPP signal. Comparing them on the same test set with the same metrics tells the team what kind of approach actually works on this data.

**Q: Why multiplicative seasonality in Prophet?**
A: Because retail seasonal swings scale with the level of the series. A 20% Black Friday lift in 2016 moves more absolute units than the same percentage lift in 2011. Additive seasonality assumes the absolute swing is constant, which is wrong for any growing business. The diagnostic is to look at residuals — if their variance fans out over time, you should be using multiplicative.

**Q: Why is LightGBM a "global" model and what does that buy you?**
A: A global model is one trained across all series simultaneously, with series identifiers as features. The alternative is one model per series. Global wins three things: cross-series patterns get learned (a back-to-school bump in WEST helps predict MIDWEST), more training data per fit, and easier deployment. The tradeoff is the model has to learn that regions are different by using region as a feature, instead of being trained separately by construction.

**Q: Walk me through your features.**
A: Six lag features at 1, 2, 4, 8, 12, and 52 weeks — covering short-term momentum, monthly cycle, trend shifts, and yearly seasonality. A year-over-year growth ratio. Calendar features for year, month, quarter, fiscal week. Cyclical encoding of week-of-year as sin and cos so the model knows week 52 and week 1 are adjacent. Three retail event flags for peak season, back-to-school, and holiday weeks. And two lagged UPP features which are exogenous signals — I use lagged not current UPP to avoid leakage.

**Q: Why cyclical encoding instead of just using week number?**
A: Because the model treats numbers as having ordinal distance, but week 52 and week 1 are conceptually adjacent. With sin and cos encoding, the two features wrap around naturally — week 52 and week 1 are next to each other in the 2D embedded space. Same trick applies to hour-of-day, day-of-week, month, anything that cycles.

**Q: Why didn't you use a random train/test split?**
A: Time-series data has temporal dependencies. Random shuffling means the model sees future observations during training and predicts the past — that's leakage and inflates accuracy estimates. The right approach is chronological: train on the first N weeks, test on the last K weeks. That mirrors how the model would actually be used in production.

**Q: What's WMAPE and why is it better than MAPE for this problem?**
A: WMAPE is sum-of-absolute-errors divided by sum-of-actuals, expressed as a percentage. Standard MAPE averages percentage errors per point, which has two problems: it explodes when actuals are small or zero, and it weights every point equally regardless of business importance. WMAPE is bounded, robust to small actuals, and naturally weights by volume — a busy week's error matters more than a quiet week's.

**Q: What's Traditional Error and why is it the primary metric?**
A: Traditional Error is sum of forecast minus sum of actual, divided by sum of actual, as a percentage. It captures aggregate directional bias — positive means systematic over-forecast, negative means systematic under-forecast. The team uses it as primary because over-forecasting and under-forecasting have asymmetric costs in parcel ops: over-forecast means paying for unused contracted capacity, under-forecast means service failures. WMAPE tells you how noisy you are; Traditional Error tells you whether you're consistently leaning the wrong way.

**Q: Your feature importance shows lag_8 as #1 — does that surprise you?**
A: It surprised me at first because intuitively I'd expect lag_1 to dominate. The reason lag_8 ranks high is that this is a relatively short test set on a series with strong seasonal structure, and the boosting algorithm finds that lag_8 partitions the data well at multiple split points. The bigger story is which features have near-zero importance: most calendar features are essentially unused, because lag_52 already captures yearly seasonality. The model converges on the lags a forecaster would intuitively use.

**Q: What's lag analysis and why does the team care about it?**
A: Lag analysis measures forecast accuracy at different planning horizons. If the data lake is one week behind, you can only do lag-1 forecasts; some processes need 4-week lookaheads. So I report Traditional Error and WMAPE separately for lag-1 through lag-4 forecasts, generated by retraining the model at each historical cutoff. The expected pattern is monotonic accuracy degradation as the horizon extends — exactly what mine shows. That's the same format the team's existing WPR report uses.

**Q: When would you choose Prophet over LightGBM, even if Prophet has worse accuracy?**
A: Three situations. First, when interpretability matters more than accuracy — Prophet decomposes into named components. Second, when uncertainty intervals matter — Prophet provides them natively. Third, when you have very little data — Prophet's strong priors on trend and seasonality help in low-data regimes where LightGBM would overfit.

**Q: What would you do differently if you had more time?**
A: Three things. Hyperparameter tuning on LightGBM via Bayesian search. Expand backtesting beyond Prophet to all models with proper time-series CV. And blend the models — a weighted average of LightGBM and Prophet often beats either alone since they make different kinds of errors.

**Q: How does LightGBM avoid overfitting on a small dataset?**
A: Three regularization mechanisms in the params. `min_data_in_leaf=20` requires at least 20 samples in each leaf, preventing memorization. `feature_fraction=0.9` and `bagging_fraction=0.9` randomly subsample features and rows for each tree. And the small `learning_rate=0.05` combined with 500 estimators means each tree contributes only a tiny correction.

**Q: What's data leakage in time-series ML and how do you guard against it?**
A: Leakage is when training-time information includes data unavailable at prediction time. In time series, the most common forms are: random train/test splits, using non-lagged exogenous features, and computing features over the full dataset before splitting. Guards: chronological splits, lagged versions of any exogenous variable, compute aggregates only on the training portion of each fold.

### 10.3 Follow-up questions

**Q: How would you productionize this?**
A: Containerize the training script, schedule it weekly via Airflow or Argo, write predictions to a feature store. Add feature validation so a schema change upstream fails loudly. Wire WMAPE and Traditional Error into a monitoring dashboard with alerts when either crosses a threshold for two consecutive weeks. Save model artifacts and feature snapshots so any prediction can be reproduced.

**Q: How would you handle a sudden demand shift like COVID?**
A: Three layers. Detection: monitor the prediction-actual gap; if it spikes beyond historical bounds for two consecutive weeks, flag a regime shift. Adaptation: shorten the training window, or add a regime indicator feature. Override: provide an analyst-facing override mechanism for black-swan periods.

**Q: Is a 12-week test set enough?**
A: It's a small sample for stable metric estimates. The lag analysis partially mitigates this by retraining at multiple cutoffs. For more robust assessment I'd implement walk-forward validation across the full historical period — pick 20 different cutoffs, retrain at each, average the metrics. For this prototype the 12-week test was a deliberate scope decision.

**Q: Why didn't you use a neural network?**
A: For tabular data of this size — a few thousand rows — gradient boosting almost always matches or beats neural nets while being faster to train and more interpretable. Neural nets become competitive at much larger scales or with natural sequence structure benefiting from Temporal Fusion Transformer-style architectures.

**Q: How do you know your features aren't leaking?**
A: Three checks. Every feature is either a lag of the target or an attribute of the prediction date itself. Exogenous features like UPP are explicitly lagged. The train/test split is strictly chronological, and feature computation respects group boundaries via groupby-shift, so a value from the test period can't accidentally appear as a lag in the train period.

**Q: What's the most important thing you learned?**
A: That the metric you choose shapes the conversation more than the model you choose. The team's choice of Traditional Error as primary is what makes this project relevant to their decisions — over- vs under-forecast is what costs money in parcel ops. Picking generic metrics like RMSE would technically work but would miss the bias signal entirely. Forecasting is as much about "what does success mean to the people consuming the forecast" as it is about modeling.

---

## 11. Key Terms Glossary

| Term | One-line definition |
|---|---|
| Forecasting | Predicting future values of a time-ordered series |
| Point forecast | Single predicted value (vs probabilistic forecast giving a distribution) |
| Confidence interval | Range that contains the truth with stated probability (Prophet's 80% CI) |
| Trend | Long-term direction of a series |
| Seasonality | Repeating pattern at a fixed period |
| Cyclic pattern | Repeating pattern at a non-fixed period |
| Stationarity | Statistical properties don't change over time |
| Decomposition | Splitting a series into trend, seasonality, residual components |
| Additive vs multiplicative seasonality | Whether seasonal swings are constant size or scale with level |
| Lag feature | Past value of the target used as a predictor |
| Cyclical encoding | Sin/cos transform so cyclic features wrap around correctly |
| Exogenous feature | An external signal used as a predictor (UPP, weather, prices) |
| Global model | One model trained across all series with identifiers as features |
| Local model | One model per series |
| Train/test split | Holding out data for unbiased evaluation; chronological for time series |
| Backtesting | Simulating model performance at historical cutoffs |
| Walk-forward validation | Backtesting with expanding training windows |
| Horizon | How many steps ahead you're forecasting |
| Lag analysis | Reporting accuracy at multiple forecast horizons |
| Baseline | The simplest defensible model; the floor every other model must beat |
| Moving Average | Forecast = mean of last N observations |
| Prophet | Facebook/Meta's decomposable additive forecasting library |
| LightGBM | Microsoft's gradient-boosted decision tree library |
| Gradient boosting | Sequentially fit small trees to residuals |
| Feature importance | How much each feature contributed to model accuracy |
| MAE / RMSE / MAPE / WMAPE | Different error metrics with different sensitivities |
| Traditional Error | Aggregate signed bias; team's primary metric |
| Bias vs variance (forecasts) | Systematic offset vs noisy scatter |
| Data leakage | Training-time access to data unavailable at prediction time |
| Multiplicative model | Components combine by multiplication; appropriate for proportional growth |
| Changepoint | Point in time where a trend slope changes (Prophet detects automatically) |
| Fourier series | Sum of sines and cosines used to model periodic functions (Prophet seasonality) |
| Bayesian inference | Statistical method that updates beliefs given data; underlies Prophet |

---

## 12. Memorable Analogies

- **Baseline model**: The pace car at a NASCAR race. Doesn't win, but every contender has to lap it to be taken seriously.
- **Prophet**: An experienced retail analyst who decomposes "we sold X" into "base trend + holiday boost + Q4 spike + noise." Names the parts.
- **LightGBM**: 500 specialists, each correcting tiny mistakes the previous 499 missed. No single specialist is smart, but together they're powerful.
- **Lag features**: A weather forecaster looking at yesterday's temperature, last week's, and the same day last year. Past observations as predictors of the future.
- **Cyclical encoding**: A clock has 12 numbers but they wrap around — 12 and 1 are next to each other. Sin/cos lets the model see that adjacency.
- **Global vs local model**: A single doctor who's seen 10,000 patients vs 50 doctors who've each seen 200. The single doctor recognizes patterns across populations.
- **Multiplicative seasonality**: A 10% raise on $50K is $5K, but a 10% raise on $500K is $50K. The percentage stays the same; the absolute amount scales.
- **Train/test split for time series**: You can't predict last week using next week's data. Even by accident. Order matters.
- **Backtesting**: Pretending to be in 2015 and asking "if I'd built this model then, how would it have done in 2016?" Repeated for many start dates.
- **Forecast horizon**: A weather forecast for tomorrow is more accurate than one for next week. Same for any prediction — confidence drops with distance.
- **WMAPE vs MAPE**: WMAPE asks "by what % did you miss the year's headcount total?" MAPE asks "average % miss per event." If one quiet event has 5 expected and 50 attend, MAPE says 900% on that one — WMAPE puts it in perspective.
- **Traditional Error**: A scale that always reads 2 lbs heavy. Your weight is biased by 2 lbs every time. That's a bias problem, not a noise problem.
- **WMAPE specifically**: Estimating how many people show up to events all year. WMAPE asks "over the whole year, by what percentage did you miss the total headcount?" Not "average error per event" — for capacity planning, the year-total accuracy is what books your venues, your catering, your staff.
- **Gradient boosting**: Each new tree is like a critic listening to the previous trees and saying "you missed THIS pattern" — and then trying to predict only that residual. After 500 critics weighing in (each gently), the ensemble sees patterns no single tree could catch.
- **Confidence intervals**: A weather app saying "tomorrow's high: 72°F, range 68–76." The point estimate is 72; the band tells you how confident the model is. Prophet's 80% CI means 80% of the time, the truth should fall inside the orange band.
- **The whole notebook**: A bake-off. Three contestants (baseline, Prophet, LightGBM) cook from the same ingredients (cleaned data from Notebook 1) using different recipes. Same judges (Traditional Error, WMAPE) score them. The scorecard is the leaderboard. The lag analysis is the "but how do they do under different conditions?" follow-up round.

---

*Last updated: 2026-05-09*