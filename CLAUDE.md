# ParcelCast — Claude Code Context

> Read this first. This file is the source of truth for working on this project.

## What this project is

A weekend portfolio project demonstrating end-to-end parcel volume forecasting on the public M5 retail dataset. Built as a portfolio project for retail/logistics analytics roles.

**Audience:** Sam (Director of Parcel Forecasting at a major retailer). They know the WPR reporting format, use Traditional Error and WMAPE as primary/secondary metrics, and care about FedEx HD contract compliance and OnTrac Tier 2 thresholds.

**Time budget:** One weekend. Saturday for build, Sunday for deck + polish. Reproducibility and narrative matter more than model accuracy gains beyond a strong baseline.

## The domain mapping (critical context)

This project re-frames M5 retail data as a parcel forecasting problem:

| M5 | → ParcelCast |
|---|---|
| Unit Sales | Ordered Units |
| Store (CA_1, TX_2, …) | FC (Fulfillment Center) |
| State (CA, TX, WI) | Region (WEST, SOUTH, MIDWEST) |
| FOODS, HOUSEHOLD | Channel: 1P |
| HOBBIES | Channel: MP (3P Marketplace seller) |
| Units ÷ UPP | Package Volume |

UPP (Units Per Package) values: 1P ≈ 2.10 declining 3%/yr, MP ≈ 1.33 static. The declining-1P-UPP insight matches the real WPR report observation and is the project's signature analytical detail. Don't lose it.

Carrier allocation: FedEx ~55%, UPS ~15%, IP ~20%, OnTrac ~10%, with regional adjustments (OnTrac is West-Coast-heavy, etc.). All numbers are illustrative assumptions — clearly label as such.

## Project structure

```
parcelcast/
├── README.md                      # Public-facing project description
├── CLAUDE.md                      # This file
├── SATURDAY_PLAN.md              # Hour-by-hour playbook
├── requirements.txt
├── data/                          # M5 raw + intermediate parquet (gitignored)
├── src/                           # All importable modules
│   ├── data_loader.py             # M5 load, reshape, hierarchy mapping
│   ├── parcel_transform.py        # UPP conversion, carrier split
│   ├── quality.py                 # Profiling, cleaning, audit log, validation
│   ├── features.py                # Lags + calendar features
│   ├── models.py                  # MovingAverageBaseline, ProphetModel, LightGBMForecaster
│   ├── evaluation.py              # Traditional Error, WMAPE, lag analysis, scorecard
│   └── business.py                # FedEx monitor, OnTrac alert, cost simulator
├── notebooks/
│   ├── 01_data_quality_eda.py     # Loads → cleans → validates (run as jupytext)
│   ├── 02_modeling.py             # Baseline + Prophet + LightGBM, scorecard
│   └── 03_business_applications.py # FedEx, OnTrac, cost shift, dashboard
└── presentation/                  # Saved chart PNGs for the deck
```

## Conventions used in this project

### Code style
- Python 3.10+ syntax (uses `dict[str, ...]` type hints)
- Modules export functions/classes; notebooks orchestrate
- Notebooks are stored as jupytext `.py` files (percent format) so they diff cleanly in git. Convert with `jupytext --to ipynb notebooks/*.py`
- All chart-saving paths route through `Path.cwd().parent / "presentation"`
- All data artifacts go to `Path.cwd().parent / "data"` as parquet (or CSV when human-readable)

### Metrics
- **Traditional Error** = (Sum Forecast − Sum Actual) / Sum Actual × 100. Primary metric.
- **WMAPE** = Sum(|actual − forecast|) / Sum(actual) × 100. Secondary.
- Do NOT use generic MAPE or RMSE in any visible output. Use the team's metrics.

### Naming
- `week_start` = the Saturday of the retailer's fiscal week (Sat–Fri)
- `region` ∈ {WEST, SOUTH, MIDWEST}
- `channel` ∈ {1P, MP}
- `carrier` ∈ {FedEx, UPS, IP, OnTrac}
- Use `packages` for total package volume, `carrier_packages` for per-carrier

### Cleaning philosophy
- **Never delete rows** — winsorize outliers, interpolate missing
- Every transformation gets logged via `CleaningAuditLog`
- Recalculate derived columns from components after any cleaning step
- Run `run_validation_suite()` at the end of cleaning; aim for 100% pass

### Modeling
- Always include the 4-week MA baseline. Every model must beat it to justify its complexity.
- Prophet uses `seasonality_mode='multiplicative'` and the predefined `US_RETAIL_HOLIDAYS` table in `src/models.py`
- LightGBM is a global model — one model trained across all (region, channel) series with group identifiers as features

## How to use Claude Code on this project

### Good prompts for this project

- "Run notebook 01 end-to-end and report the row counts at each major step. Don't modify anything."
- "Notebook 02 is failing at the LightGBM cell with [paste error]. Read src/features.py and src/models.py, identify the cause, propose a fix, then apply it after I confirm."
- "Add a new chart to notebook 03 showing carrier-level forecast error broken down by region. Save it as presentation/10_carrier_error_by_region.png. Use the existing seaborn style."
- "Convert all three notebook .py files to .ipynb using jupytext."

### Things to NOT do
- Don't let Claude Code rewrite the metrics in `src/evaluation.py` — Traditional Error and WMAPE are non-negotiable
- Don't let it "improve" the domain mapping. The mapping is intentional and tied to the WPR report.
- Don't let it switch from Prophet to a different library without confirming. Prophet's holidays are pre-configured and align with the WPR narrative.
- Don't let it auto-format huge swaths of files unprompted. Diffs should be small and reviewable.

### Acceptance criteria
A change is "done" when:
1. The relevant notebook runs top-to-bottom without errors
2. Charts are saved to `presentation/` with the existing filename pattern
3. The validation suite still passes (when applicable)
4. Git diff is reviewable in under 2 minutes

## Time-boxing rules

If Claude Code is debugging the same issue for more than 15 minutes:
1. Commit current state
2. Either accept a simpler fallback (drop the offending feature) or pause and think before continuing
3. Refer to SATURDAY_PLAN.md "Cut list" — there's a documented fallback for every block

## What "good" looks like for the final deliverable

By Sunday evening, this repo should have:
- 3 reproducible notebooks (run end-to-end without errors)
- 9+ saved charts in `presentation/`
- Cleaning audit log CSV
- Validation suite output (all passing)
- Model scorecard CSV
- Lag analysis CSV (mirrors WPR format)
- A clean README with screenshots
- An 8-slide deck (built separately, dropped into `presentation/parcelcast_deck.pdf`)

If any of these are missing Sunday at 4 PM, refer to the cut list.
