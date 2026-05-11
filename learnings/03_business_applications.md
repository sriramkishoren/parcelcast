# Business Applications — Learnings (Notebook 3)

> Consolidated notes from the walkthrough of `notebooks/03_business_applications.py`.
> Written for both **practical understanding** and **interview preparation**.

---

## Table of Contents

1. [Overall Purpose](#1-overall-purpose)
2. [Outputs from Previous Notebooks](#2-outputs-from-previous-notebooks)
3. [Forecast Horizon / Forecasted Weeks](#3-forecast-horizon--forecasted-weeks)
4. [Forecast and Historical Carrier Volumes](#4-forecast-and-historical-carrier-volumes)
5. [Per-Carrier Package Forecasts](#5-per-carrier-package-forecasts)
6. [FedEx HD Monthly Contract Monitor](#6-fedex-hd-monthly-contract-monitor)
7. [OnTrac Tier 2 Risk Alert](#7-ontrac-tier-2-risk-alert)
8. [Carrier Cost-Shift Simulator](#8-carrier-cost-shift-simulator)
9. [Executive 4-Panel Dashboard](#9-executive-4-panel-dashboard)
10. [Code Flow / Implementation](#10-code-flow--implementation)
11. [Interview Preparation](#11-interview-preparation)
12. [Key Terms Glossary](#12-key-terms-glossary)
13. [Memorable Analogies](#13-memorable-analogies)

---

## 1. Overall Purpose

### What Notebook 3 is trying to achieve

In one sentence: **turn the forecast into a decision**.

Notebook 1 cleaned the data. Notebook 2 produced a forecast number. Notebook 3 takes that number and answers four operational questions:

1. **"Will we honor our FedEx contract next month?"** → contract monitor
2. **"Are we at risk of breaching OnTrac's Tier 2 cap?"** → alert system
3. **"What does it save if we shift 5% of volume between carriers?"** → cost simulator
4. **"Show me the whole picture in one slide."** → executive dashboard

A forecast that just sits in a parquet file isn't useful. A forecast that fires alerts, monitors contracts, and quantifies trade-offs is a **product**.

### Forecasting visualization vs business monitoring vs decision-making?

Primarily **business monitoring + decision-making**, with visualization as the wrapper:

| Component | Type |
|---|---|
| FedEx Contract Monitor | Compliance monitoring |
| OnTrac Tier 2 Alert | Risk alert |
| Carrier Cost-Shift Simulator | Decision-making (what-if) |
| Executive Dashboard | Visualization |

The deeper pattern: **every component takes a numeric forecast and produces a recommended action**. That number → decision conversion is what makes this a forecasting *product* rather than a forecasting *project*.

### How Notebook 3 connects to Notebooks 1 & 2

```
Notebook 1                 Notebook 2                   Notebook 3
─────────                  ─────────                    ─────────
Raw M5 CSVs                Cleaned weekly volumes       network_forecast.parquet  ──┐
   ↓                       (from N1)                                                ↓
Reshape + clean                ↓                        weekly_network_volumes      ↓
   ↓                       Train 3 models               .parquet (from N1)  ───────→Business apps
weekly_network_volumes     Generate 12-week forecast    model_scorecard.csv         ↓
weekly_region_channel      → network_forecast.parquet   (from N2)  ────────────────→Dashboard
.parquet                                                                            ↓
                                                                             [decisions, alerts, $$$]
```

Notebook 3 is **purely a consumer** — reads three artifacts produced upstream, applies business logic, writes new artifacts (`fedex_contract_status.csv`, `cost_optimization.csv`) plus three PNGs. It doesn't train any models or do data cleaning itself.

---

## 2. Outputs from Previous Notebooks

### Files reused

| File | Source | What it is |
|---|---|---|
| `data/network_forecast.parquet` | Notebook 2 | LightGBM's 12-week network-level package forecast (with low/high CI bands) |
| `data/weekly_network_volumes.parquet` | Notebook 1 | Cleaned historical weekly carrier volumes (one row per region × channel × carrier × week) |
| `data/model_scorecard.csv` | Notebook 2 | Three-model comparison (Traditional Error %, WMAPE %, rank) — used in dashboard panel 3 |

### How they're loaded

```python
forecast = pd.read_parquet(DATA_DIR / "network_forecast.parquet")
carrier_history = pd.read_parquet(DATA_DIR / "weekly_network_volumes.parquet")
scorecard = pd.read_csv(DATA_DIR / "model_scorecard.csv")
```

Three lines, no transformation needed — Notebook 1 and 2 already produced clean, validated artifacts.

### Why this dependency matters in real-world ML pipelines

This pattern is called **pipeline materialization** — each stage writes a versioned artifact downstream stages consume. Benefits:

1. **Re-run any stage independently** without re-running everything upstream.
2. **Reproducibility** — point-in-time artifacts can be rolled back.
3. **Decoupling** — Notebook 3 doesn't care HOW the forecast was made, just that it conforms to the expected schema.
4. **Testability** — each stage tests its inputs and outputs against a fixed contract.

Production analog: dbt models, Airflow tasks reading/writing S3, MLflow artifact tracking.

---

## 3. Forecast Horizon / Forecasted Weeks

### How many weeks were forecasted?

**12 weeks** (~3 months). Set in Notebook 2 via `TEST_WEEKS = 12` and carried forward when generating the production forecast.

### Was forecasting done in Notebook 2?

Yes. Notebook 2:

1. **Trained** three models on the first ~258 weeks
2. **Evaluated** on a 12-week chronological test set → `model_scorecard.csv`
3. **Generated** a forward-looking 12-week forecast (using LightGBM, the winner) → `network_forecast.parquet`

Notebook 3 only consumes step 3's output.

### How to choose forecast horizon in real-world projects

Three constraints:

| Constraint | What it means |
|---|---|
| **Decision lead time** | Carrier contracts: monthly. Workforce hiring: 4–8 weeks. Forecast must match the longest lead-time decision. |
| **Accuracy decay** | Forecasts get worse the further out (lag analysis: WMAPE 4.88% → 5.32% from lag-1 to lag-4). |
| **Data availability for retraining** | Weekly retrains → "next 12 weeks" rolls forward each week. |

For parcel forecasting at a major retailer, **12 weeks** is the sweet spot.

### Why weekly (not daily/monthly)?

| Granularity | Use case | Tradeoff |
|---|---|---|
| **Daily** | Same-day operations | Noisier; harder seasonality |
| **Weekly** ✅ | Tactical planning | Smooths daily noise; matches WPR reporting |
| **Monthly** | Strategic planning | Loses too much resolution |

Weekly also aligns with the retailer's fiscal-week calendar — matching the audience's grain is half the battle.

---

## 4. Forecast and Historical Carrier Volumes

### Where the forecast was calculated

`network_forecast.parquet` is generated **at the end of Notebook 2**:

1. Refit LightGBM on **all** training data (no held-out split — fresh model)
2. Generate predictions for the next 12 weeks at network level
3. Save with columns: `week_start`, `packages_forecast`, `packages_low`, `packages_high`

### How historical and forecasted values combine

```
weekly_network_volumes.parquet (~270 weeks of actuals)
                ↓
        last 52 weeks plotted as solid black line  ──┐
                                                     │
network_forecast.parquet (12 future weeks)            ├─→ Panel 1 of dashboard
                ↓                                     │
        12 weeks plotted as solid blue line     ──────┘
        with shaded CI band
```

Same `week_start` axis. **History ends, forecast begins** — continuity point on the x-axis.

### Why dashboards need both

1. **Context** — a forecast number alone is meaningless. "We'll do 170K next week" needs "we did 165K last week" as anchor.
2. **Trust** — viewers eyeball whether the forecast plausibly continues the historical trend.
3. **Storytelling** — "what's *changing*?" requires showing both before and after.

Common rookie mistake: showing only the forecast on a fresh chart with no history.

---

## 5. Per-Carrier Package Forecasts

### How they're derived

**Two-step top-down**:

**Step 1**: LightGBM forecasts the **network total** (one number per week × 12 weeks).

**Step 2**: split using fixed carrier shares:

```python
for carrier, share in CARRIER_BASE_SHARES.items():
    forecast_wide[f"{carrier.lower()}_packages"] = forecast["packages_forecast"] * share
```

With shares `{FedEx: 0.55, UPS: 0.15, IP: 0.20, OnTrac: 0.10}`, a 100K week becomes 55K + 15K + 20K + 10K.

### Top-down vs bottom-up

| Approach | Pros | Cons |
|---|---|---|
| **Top-down** (what we do) | One model. Smoother aggregate signal. Allocation easy to update. | Static shares can't learn carrier-specific dynamics. |
| **Bottom-up** | Each carrier learns its own pattern. | 4× the models. Carrier sums may not match network total. |
| **Hybrid (reconciliation)** | Best of both — forecast all levels, reconcile. | Complex; needs `hts`, `nixtla`, etc. |

For a prototype, top-down is the right choice — defensible and lets you swap allocators later.

### Business / allocation logic

`CARRIER_BASE_SHARES` in [src/parcel_transform.py](../src/parcel_transform.py) — illustrative network-wide percentages. Notebook 1 applies regional adjustments; Notebook 3 uses simpler network-wide shares because the contract monitor and alerts operate at the network level.

### How a real logistics system would do this

In production:

1. **Allocation shares wouldn't be static** — they'd come from carrier capacity contracts plus a routing optimizer.
2. **Allocation could be week-of-week dynamic** — peak shifts, route closures, weather events all change the optimal split.
3. **Per-carrier forecasts would be reconciled** with the network forecast.

The weekend version skips all that — "swap in the production allocator and the rest of the pipeline still works."

---

## 6. FedEx HD Monthly Contract Monitor

### Where the thresholds came from

```python
FEDEX_HD_CONTRACT_MIN_MONTHLY_PACKAGES = 320_000
FEDEX_HD_CONTRACT_MAX_MONTHLY_PACKAGES = 420_000
```

These are **business contract thresholds**, not statistical limits. In real life they come from the legal contract the retailer signs with FedEx.

For this prototype they're **rescaled** from the real-world WPR numbers (~19.5M to ~26.8M monthly) down to the M5-derived dataset's volume scale, so the alerts actually flip status across months.

### Business threshold vs statistical threshold

| Type | Source | Purpose |
|---|---|---|
| **Business threshold** (FedEx contract) | Hand-set by humans based on contracts/policy | Compliance, decision-making |
| **Statistical threshold** (e.g., 3σ alert) | Derived from data distribution | Anomaly detection |

A statistical threshold says "this is unusual." A business threshold says "this violates an agreement."

### Why FedEx HD specifically

1. **Contract structure** — FedEx HD is the anchor carrier (55% of network), strictest min/max bands.
2. **Penalty exposure** — missing the FedEx min has direct financial consequences.
3. **Story focus** — one carrier illustrates the pattern; building monitors for all four would dilute the deck narrative.

The monitor function in [src/business.py:27-83](../src/business.py#L27-L83) is generic — works for any carrier.

### The business problem this solves

Contracts have **financial consequences**:

- **Below MIN** → "shortfall penalty" (paying for unused capacity)
- **Above MAX** → service degradation, SLA penalties
- **In band** → no action

Without a forward-looking monitor, the team learns about a contract miss at the end of the month. With a 12-week forecast running through the contract band, they know **8 weeks ahead** that April is going to bust the ceiling.

This is forecasting's highest ROI use case: **buying lead time on a costly mistake**.

### How stakeholders use this

Sam opens the dashboard and sees:

- "April 2016: ABOVE MAXIMUM by 26,401 packages"
- "Recommended action: Shift volume FROM FedEx TO UPS/IP/OnTrac"

Her response: "Brian, push the April rebalance to next week's carrier ops review."

The monitor doesn't make decisions — it surfaces the right ones at the right time.

---

## 7. OnTrac Tier 2 Risk Alert

### What "Tier 2" means

Carrier capacity contracts are typically **tiered**:

- **Tier 1** — normal operating capacity
- **Tier 2** — surge capacity (costs more or higher-priority handling)
- **Tier 3** — extreme overflow with significant SLA penalties

OnTrac's Tier 2 threshold isn't a contract violation — it's a **cost or service trigger**. You want to avoid it.

### How the risk is calculated

[src/business.py:88-147](../src/business.py#L88-L147):

1. Take historical weekly OnTrac volume series
2. Compute **rolling 6-week average** to smooth single-week spikes
3. Flag weeks where rolling avg > threshold
4. **Project 4 weeks forward** using simple linear trend on last 4 weeks
5. Recompute rolling avg with projection — flag any future breach
6. Return: current rolling avg, breach week (if any), recommended action

Rolling avg (not raw) because Tier 2 is sustained capacity utilization, not one-off bursts.

### Why OnTrac specifically

- **Regional carrier** (West Coast heavy) — hard capacity ceilings
- **Tier 2 trigger is asymmetric** — costs jump permanently for the period
- **Smallest carrier** in the mix (10%) — most likely to be over-allocated during regional spikes

Other carriers don't have the same Tier 2 cost structure or are less capacity-constrained.

### Forecasting variance vs operational threshold vs business rule?

It's an **operational threshold** (Tier 2 capacity number) implemented as a **business rule** (when rolling avg > threshold, alert). Forecasting variance comes in indirectly through the 4-week projection.

### Properties of mature alert systems

1. **Tunable threshold** — lives in config, not code
2. **Smoothing** — rolling avg, EWMA, or hysteresis prevent flapping
3. **Forward projection** — alerts fire **before** the breach
4. **Recommended action** — tells recipient what to do, not just that something happened

Anti-pattern: alerts that fire constantly (alert fatigue), alerts only after damage, alerts with no associated action.

---

## 8. Carrier Cost-Shift Simulator

### Purpose

Answers: **"What would it cost (or save) if we re-routed some volume between carriers?"**

The forecast tells you *what will happen*. The simulator tells you *what could happen if you intervene*.

### How we decided to create this

Flows from the contract monitor: if April is forecasted ABOVE MAX, **the natural follow-up question** is "where do we move the excess?" Without the simulator you can flag the problem but not quantify the fix.

This is the design instinct that separates a "forecast" from a "forecast product": **every alert should have a paired decision tool**.

### How it works conceptually

```
INPUT:   current carrier allocation (weekly volumes)
         cost per package per carrier
         shift_pct (how much to move)
         shift_from (which carriers to take from)
         shift_to (where to send it)

LOGIC:   compute current weekly cost
         compute proposed allocation after the shift
         compute proposed weekly cost
         delta = current - proposed
         annualize delta × 52

OUTPUT:  scenario table
```

For our run:

| Scenario | FedEx | IP | OnTrac | UPS | Weekly Cost | Savings | Annualized |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | 99,058 | 37,366 | 15,011 | 28,566 | $1,450,521 | — | — |
| Shift 5% from FedEx+UPS → IP | 94,106 | 43,746 | 15,011 | 27,138 | $1,436,123 | $14,398 | $748,696 |

### Inputs and assumptions

| Input | Source | Documented as illustrative? |
|---|---|---|
| Current allocation | Latest week's actuals from `weekly_network_volumes.parquet` | No — real values from data |
| Cost per package per carrier | `COST_PER_PACKAGE` (FedEx $8.50, UPS $9.20, IP $6.40, OnTrac $7.10) | **Yes** — flagged in the deck |
| Shift % | Hand-set parameter (5%) | Configurable |
| Shift directions | "Take from FedEx+UPS, send to IP" | Hand-chosen scenario |

**The big assumption**: carriers absorb the shift at the same per-package cost. In reality there are capacity limits and per-package costs that scale non-linearly.

### Technical implementation

[src/business.py:152-194](../src/business.py#L152-L194):

```python
def carrier_cost_shift_simulator(current_allocation, cost_per_package,
                                 shift_pct=0.05, shift_from=None, shift_to="IP"):
    current_cost = sum(current_allocation[c] * cost_per_package[c] for c in current_allocation)
    proposed = dict(current_allocation)
    total_to_shift = sum(int(proposed[c] * shift_pct) for c in shift_from)
    for c in shift_from:
        proposed[c] -= int(proposed[c] * shift_pct)
    proposed[shift_to] += total_to_shift
    proposed_cost = sum(proposed[c] * cost_per_package[c] for c in proposed)
    # ...build scenario table, multiply by 52 for annual
```

Deliberately simple — pure arithmetic, no optimization, no constraints. A real-world version would add capacity constraints, per-region cost differences, and a linear-programming optimizer.

### How operations / finance teams use this

**Operations team** (carrier-ops manager): "Sam flagged April will bust the FedEx ceiling. Simulator says shifting 5% to IP saves $14K/week. Let me model 7%, 10%, 12% to find the right amount."

**Finance team** (carrier-spend analyst): "What's the annualized impact of carrier-mix changes proposed for next quarter?" Builds a budget review table.

The simulator is a **what-if tool** — enables conversations that wouldn't happen otherwise because the math was too tedious.

---

## 9. Executive 4-Panel Dashboard

### Why a single image vs multiple charts

Executives don't open dashboards. They open slide decks and emails. A 4-panel image **fits in one slide**.

The 4 panels mirror the 4 questions an executive will ask:

| Panel | Question it answers |
|---|---|
| 1: Volume trend + forecast | "What does the next quarter look like?" |
| 2: Carrier share pie | "How is volume distributed right now?" |
| 3: Model scorecard | "Can I trust the forecast?" |
| 4: FedEx contract status | "What action do I need to take?" |

### Panel 1: which model's forecast?

**LightGBM only**, even though three models were trained.

**Why one, not all three?**

- **Cognitive load** — executive wants the answer, not a model bake-off
- **Confidence** — committing to a recommendation, not "we couldn't decide"
- **Scorecard handles trust** — Panel 3 separately shows accuracy

### How LightGBM was determined to be best

| Model | Traditional Error % | WMAPE % | Rank |
|---|---:|---:|---:|
| **LightGBM** | −0.64 | **1.55** | **1** |
| Baseline (4wk MA) | −1.86 | 3.96 | 2 |
| Prophet | −5.56 | 5.88 | 3 |

LightGBM ranks first by WMAPE **and** has smallest absolute Traditional Error. Wins on both axes.

The decision is encoded in Notebook 2 (`network_forecast.parquet` uses LightGBM specifically). Notebook 3 trusts that decision.

### Why FedEx contract status (not other carriers) in Panel 4

Same logic as §6 — FedEx is the anchor carrier with largest financial exposure. Carrier share pie in Panel 2 already gives broader context. A more sophisticated dashboard would have a "carrier alerts" panel showing all carriers with active alerts.

### Audience and panel-by-panel interpretation

**Audience**: Sam (Director, Parcel Forecasting) and her staff. Not data scientists.

| Panel | What an executive should think |
|---|---|
| Top-left (volume trend) | "Forecast continues the historical trend smoothly. CI band is tight — model is confident." |
| Top-right (carrier share) | "FedEx still dominates at 55%. OnTrac is small but operationally important." |
| Bottom-left (scorecard) | "LightGBM at 1.55% WMAPE — under 2% is excellent. Beats both alternatives." |
| Bottom-right (FedEx status) | "March is fine. April will bust the ceiling. Decide on a rebalance now." |

Together: "the forecast is trustworthy, here's what's coming, and here's the action to take."

---

## 10. Code Flow / Implementation

### 7-step walkthrough

**Step 1 — Imports**: `src.business` (monitor functions + threshold constants), `src.parcel_transform` (`COST_PER_PACKAGE`, `CARRIER_BASE_SHARES`).

**Step 2 — Load forecast and history**: two parquet reads (`network_forecast.parquet`, `weekly_network_volumes.parquet`).

**Step 3 — Project per-carrier forecasts**: multiply network forecast by carrier shares → `forecast_wide` with `fedex_packages`, `ups_packages`, etc.

**Step 4 — FedEx contract monitor**: pass `forecast_wide["fedex_packages"]` into `fedex_contract_monitor`. Groups by month, sums weekly forecasts, drops partial months (<4 weeks), compares to thresholds, returns status DataFrame. Save CSV. Render bar chart.

**Step 5 — OnTrac alert**: build OnTrac historical series. Pass into `ontrac_tier_risk_alert` → rolling avg, 4-week projection, breach detection. Print alert dict. Render line chart with risk zone.

**Step 6 — Cost simulator**: get latest week's volumes. Pass into `carrier_cost_shift_simulator` with cost dict and 5% shift. Returns scenario table. Save CSV.

**Step 7 — Executive dashboard**: 4-panel matplotlib figure. Add suptitle, footer, save at dpi=150.

### Common beginner mistakes

- **Forecast saved without carrier breakdown** — Notebook 3 has nothing to monitor
- **Hardcoded thresholds in the notebook** — should be in `src/business.py` for tunability
- **Alerts printed but not saved** — no way to monitor the monitor over time
- **Dashboard built once, never re-renderable** — should be a function of input data
- **Showing all models in the executive view** — confusion, not clarity
- **Real-world thresholds with prototype data** — alerts always fire (or never fire); meaningless

---

## 11. Interview Preparation

### 11.1 The 90-second end-to-end story

> "ParcelCast is a parcel-volume forecasting prototype I built on the public M5 retail dataset. Three notebooks, one pipeline.
>
> Notebook 1 cleans 5+ years of daily SKU sales — wide-to-long melt, fiscal-week aggregation, a domain transform that converts retail units to package volume via Units-Per-Package, a carrier-share split into FedEx, UPS, IP, and OnTrac. Every cleaning step writes to an audit log, and I run an 8-check validation suite at the end.
>
> Notebook 2 trains three models — a 4-week moving-average baseline, Prophet with multiplicative seasonality and US retail holidays, and a global LightGBM with lag, calendar, and lagged UPP features. LightGBM wins at 1.55% WMAPE on a 12-week chronological test. I also run a lag analysis that mirrors the team's existing WPR reporting format.
>
> Notebook 3 is the business application layer — it takes the forecast and produces decisions. A FedEx contract monitor flags April as ABOVE the contract maximum by ~26K packages. An OnTrac Tier 2 alert fires because the rolling 6-week avg just crossed the threshold. A cost-shift simulator quantifies that moving 5% of volume from FedEx and UPS to IP saves about $14K per week, $749K annualized. Everything bundles into a single 4-panel executive dashboard.
>
> The point isn't that the model is sophisticated — it's that the pipeline turns a forecast into something a director actually uses. That's the difference between a forecasting *project* and a forecasting *product*."

### 11.2 Q&A

**Q: Walk me through the data flow across your three notebooks.**
A: Notebook 1 reads three M5 CSVs and writes two parquet files plus the audit log. Notebook 2 reads those parquets, trains models, and writes the model scorecard, the lag analysis CSV, and a network forecast parquet. Notebook 3 reads the forecast parquet, the historical volumes parquet, and the model scorecard, then writes contract status and cost optimization CSVs plus the dashboard PNG. Each notebook is independent — that's pipeline materialization, the same pattern production ML systems use.

**Q: Why split the network forecast across carriers using static shares instead of forecasting each carrier separately?**
A: Top-down forecasting with a single model on the aggregate plus static allocation shares. Three reasons. One, the aggregate signal is smoother and easier to forecast. Two, one model is much easier to maintain. Three, the allocation logic is decoupled — I can update carrier shares without retraining. The tradeoff is the model can't learn carrier-specific dynamics. For a prototype, that's the right call; in production I'd consider hierarchical reconciliation.

**Q: Where do the FedEx contract thresholds come from?**
A: Business contract values from the carrier agreement, not statistical limits. In real-world parcel ops at a major retailer they'd be the actual contract minimum and maximum per month. For the prototype I rescaled them from the realistic ~19.5M/26.8M monthly range down to the M5-derived dataset's volume scale — about 320K to 420K — so the alerts actually flip status across months and tell a story. I documented the rescale in the source so anyone reviewing knows to swap in production values when deploying.

**Q: Why is the OnTrac Tier 2 alert based on a rolling average instead of raw weekly volume?**
A: Because Tier 2 is typically triggered by sustained capacity utilization, not single-week spikes. If I alert on raw weekly, a one-time surge fires the alert — false positive. The 6-week rolling average smooths that out. Same idea as why server alerts use 5-minute averages of CPU rather than instantaneous samples. The downside is the alert is laggy: by the time the rolling avg crosses, we've already had several elevated weeks. So I also project four weeks forward using a simple linear trend, which gives lead time before the breach actually happens.

**Q: Walk me through the cost simulator's logic.**
A: Take the current carrier allocation as input — actual weekly volumes per carrier. Compute the current weekly cost as sum of volume times cost-per-package. Build a proposed allocation by removing 5% from each "shift from" carrier and adding the total to the "shift to" carrier. Compute the proposed cost the same way. Difference is weekly savings. Multiply by 52 for annual. The big simplifying assumption is that the receiving carrier can absorb the shift at the same per-package cost — in reality there are capacity limits and non-linear pricing. The simulator gives a first-order estimate, not a binding quote.

**Q: Why does the executive dashboard show only the LightGBM forecast in Panel 1, not all three models?**
A: Cognitive load. An executive opening this dashboard isn't doing a model bake-off — they want the answer. Showing three lines invites "wait, which one do I trust?" Showing one line, with the model scorecard separately in Panel 3, says "here's the recommendation, here's the proof it's the best." For technical audiences I'd show all three; for executive consumption, I commit to the winning model.

**Q: How would you extend this to forecast at the per-carrier-per-region level instead of network total?**
A: Two paths. Top-down: keep the network model, then split by both carrier and regional shares — that's a 4×3 = 12-way split per week. Easy to implement, but allocation gets stale. Bottom-up: train a global LightGBM with `(region, channel, carrier)` as group features so each combination gets implicitly forecasted. The features stay the same, the target becomes per-combination volume. I'd start with bottom-up because LightGBM handles it naturally, then reconcile with the network forecast using a hierarchical method like MinT if I cared about totals matching exactly.

**Q: How would you operationalize this whole pipeline?**
A: Schedule the three notebooks weekly via Airflow or Prefect. Notebook 1 runs first against the latest data extract, writes parquets to a versioned S3 location. Notebook 2 reads them, trains, writes new model artifacts and forecasts. Notebook 3 reads everything and writes the dashboard PNG plus alert CSVs. Push alerts to Slack. Wire WMAPE and Traditional Error into a monitoring dashboard with regression alerts. Version model artifacts so any forecast can be reproduced six months later.

**Q: What's a forecasting product vs a forecasting project?**
A: A project ends with a model and a metric — "we trained LightGBM, here's its WMAPE." A product ends with a decision — "the model says volume will exceed FedEx's contract ceiling in April; here's how much it costs and here's how to fix it." The technical work is similar; the difference is in the wrapper. The product version requires monitoring, alerts, simulators, and a dashboard built around a stakeholder's specific decisions.

**Q: How do you decide which forecast horizon to use?**
A: Three constraints. Decision lead time — how far ahead does the business actually need to commit? Accuracy decay — how does WMAPE grow with horizon? My lag analysis showed degradation from 4.88% at lag-1 to 5.32% at lag-4, which is graceful, so 12 weeks is comfortable. And retraining cadence — if I retrain weekly, the practical horizon is rolling rather than fixed. For carrier ops at a major retailer, 12 weeks is the right pick because monthly contract decisions are the binding constraint.

**Q: A stakeholder says they don't trust the forecast. How do you respond?**
A: Two-step. First understand: is it a metric concern, a directional concern, or a single-week concern? Each has a different response. For metric concerns, walk through the held-out test results and the lag analysis. For directional concerns, decompose the forecast into its drivers. For single-week concerns, retrain with the latest week's actual and see if the fit improves. The wrong response is "trust me, the model is good." Always show the receipts.

### 11.3 Follow-up questions

**Q: What if the carrier shares change over time?**
A: My current implementation uses static shares — that's a known simplification. The fix is to make shares time-varying — fit them weekly from recent actuals (5-week rolling avg of share), or model them explicitly as a function of season/region. For the prototype, static is fine because the forecast horizon is short (12 weeks). For an annual forecast I'd absolutely time-vary them.

**Q: How would you handle an alert that fires constantly?**
A: First, investigate whether the alert is correct — maybe the threshold is genuinely being exceeded. If false positive, three options: raise the threshold, add hysteresis (fire at threshold + 10% but only clear at threshold − 10%), or require N consecutive breach periods before firing. Whatever I do, I'd document the change in the audit log.

**Q: Could you add SHAP values to explain LightGBM's per-week predictions?**
A: Yes — SHAP would tell us which features pushed each prediction up vs down. For the executive dashboard that's overkill, but for an analyst drill-down it'd be powerful. Two caveats: SHAP for time-series can mislead because lags share information, and the global feature importance I already compute tells most of the same story.

**Q: How would you measure whether this dashboard is actually used?**
A: Three layers of telemetry. One, instrument the dashboard generation — track who's viewing the PNG. Two, instrument the alert outputs — when contract status flips to ABOVE MAX, track whether a follow-up action happened in the carrier-ops system within 7 days. Three, qualitative feedback — quarterly check-in: "Has the dashboard influenced any decisions this quarter?" If the answer is no, the dashboard isn't working regardless of how clean the code is.

**Q: What would you do differently if you were building this for production?**
A: Five things. Replace static carrier shares with a learned allocation. Add hierarchical reconciliation. Add per-carrier monitoring (not just FedEx). Wire alerts into Slack/email rather than CSVs. Add data quality monitoring upstream — if `carrier_history` schema changes, the dashboard breaks silently. And add A/B testing infrastructure to compare a new model against production before swapping.

**Q: What's the most important lesson from building this end-to-end?**
A: That the metric you optimize and the model you build matter less than the decision the output supports. I spent the most time on Notebook 3 — the business layer — even though the modeling work in Notebook 2 was technically harder. The reason is that Notebook 3 is where the pipeline becomes useful to a human. Without the contract monitor, the cost simulator, and the dashboard, the LightGBM model is just a parquet file. With them, it's a tool that changes how a director runs her week.

---

## 12. Key Terms Glossary

| Term | One-line definition |
|---|---|
| Pipeline materialization | Each pipeline stage writes a versioned artifact downstream stages consume |
| Top-down vs bottom-up forecasting | Forecast aggregate then split, vs forecast leaves and aggregate |
| Hierarchical reconciliation | Adjusting forecasts at multiple levels so they sum consistently |
| Allocation share | Static or dynamic % of total volume going to each carrier |
| Forecast horizon | Number of future periods being predicted |
| Decision lead time | How far ahead a business decision must be committed |
| Confidence interval (CI) band | Range that contains the true value with stated probability |
| Business threshold | Hand-set limit from contracts/policy |
| Statistical threshold | Limit derived from data distribution (e.g., 3σ) |
| Tiered carrier capacity | Carriers price/SLA different volume bands differently (Tier 1, 2, 3) |
| Rolling average | Mean over a sliding window of N periods |
| EWMA | Exponentially-weighted moving average — recent points count more |
| Forward projection | Extrapolating a trend N periods into the future |
| Alert hysteresis | Threshold structure that prevents flapping (separate fire-vs-clear thresholds) |
| Alert fatigue | When too many alerts cause recipients to ignore them |
| What-if scenario analysis | Computing the cost/benefit of a hypothetical action |
| Cost-per-unit | Variable cost associated with each unit shipped/produced |
| Annualized figure | Single-period number scaled to a full year (× 52, × 12, × 365) |
| Top-line vs bottom-line | Revenue/volume vs profit/cost figures |
| Suptitle | Matplotlib figure-level title (vs axis-level title) |
| Footer/branding | Text added at the bottom for context, source, date |
| Dashboard | Single-screen summary of multiple metrics for at-a-glance decisions |
| Decision support tool | Software that quantifies trade-offs to help humans decide |
| Linear programming (LP) | Optimization with linear constraints (used in real cost optimizers) |
| Capacity constraint | Hard limit on how much a system can absorb |
| Simulation | Computing a counterfactual outcome under hypothetical assumptions |
| Forecast product vs forecast project | Forecast wrapped in monitors/alerts/simulators that drive decisions, vs forecast as standalone artifact |

---

## 13. Memorable Analogies

- **Pipeline materialization**: A factory assembly line where each station leaves its output on a conveyor belt. Downstream stations only need the part, not the upstream machinery.
- **Top-down vs bottom-up forecasting**: Top-down is forecasting the company's revenue and dividing by department. Bottom-up is forecasting each department and summing. Both approaches exist for budgets too.
- **Allocation share**: Splitting a pizza. The pizza is the network total. The slices are the carrier shares.
- **Forecast horizon**: A weather forecast for tomorrow is reliable. For next month, less so. Where the line is depends on what decision you're making — picnics need 1-day, planting crops needs 3-month.
- **Business threshold vs statistical threshold**: A speed limit (business) vs the speed at which your car starts vibrating unusually (statistical). Different rules, different responses.
- **Tier 2 capacity**: Like a phone plan with overage charges. You're fine inside your data cap. Cross it, and the per-MB cost jumps.
- **Rolling average for alerts**: A car dashboard light that turns on if the engine has been overheating for 5 minutes, not the moment it hits the threshold. Sustained signal beats instantaneous noise.
- **Alert with no recommended action**: A smoke detector that just beeps and doesn't tell you whether to leave the house or change the battery. Half a tool.
- **Cost simulator**: A "what if I refinanced my mortgage at this rate" calculator. Doesn't refinance the loan; just shows the trade-off.
- **Executive dashboard**: A car dashboard. Speedometer, fuel, RPM, warning lights — exactly what the driver needs at 70 mph. Not a maintenance manual.
- **The 4 panels**: The four corners of a Hollywood movie poster — title, hero shot, cast list, release date. Each carries one job; together they sell the movie.
- **Forecast product vs forecast project**: A project is the recipe. A product is the meal on the plate. Same ingredients, very different deliverable.
- **Showing only LightGBM in Panel 1**: A movie review that gives a star rating. Not "here are 5 critics' opinions, you decide." Commit to a recommendation.
- **Pre-shifting volume on an ABOVE MAX alert**: A driver seeing a jam ahead on Waze and rerouting before getting stuck. Forecast = the live traffic data; rerouting = the shifted volume.
- **Pipeline as conveyor belt**: Each notebook is a station. Belt = parquet files. Stations don't need to know each other — only that the part on the belt has the right shape.
- **What separates a project from a product**: Recipes vs restaurants. A recipe describes how to cook. A restaurant feeds someone, accepts payment, handles complaints, and updates the menu next season.

---

*Last updated: 2026-05-10*