# 🤖 Claude Code Prompts — Copy-Paste Playbook

> One prompt per Saturday block. Copy → paste into Claude Code → review the diff → commit. Repeat.

---

## ⚙️ How to use this file

You can drive Claude Code from a terminal **or** straight from the VS Code extension. Pick whichever flow fits the block — most people stay in VS Code for editing-heavy work and switch to the terminal when running notebooks.

### Option A — VS Code extension (recommended for this project)

1. Install the **Claude Code** extension from the VS Code marketplace and sign in.
2. Open the `parcelcast/` folder as the workspace root.
3. Confirm the Python interpreter is `./.venv/bin/python` (Cmd+Shift+P → "Python: Select Interpreter"). The repo's `.vscode/settings.json` pins this by default.
4. Open the Claude panel (Cmd+Esc, or click the Claude icon in the activity bar).
5. For each block below, paste the prompt verbatim into the Claude panel, then **review the proposed diff in the editor BEFORE clicking Accept**.
6. Use the integrated terminal (Ctrl+\`) for git commits and notebook execution — Claude can see the same workspace.
7. If you highlight code in the editor before sending a prompt, that selection is automatically included as context — handy for "fix this function" style asks.

### Option B — Terminal CLI

1. Open one terminal in your project root, run `claude` to start Claude Code.
2. Open another terminal in the project root for git, running notebooks, etc.
3. For each block below, paste the prompt verbatim, then **review what Claude Code proposes BEFORE accepting changes**.

### After every block (both options)

Commit immediately: `git add -A && git commit -m "Block N: <summary>"`. Small, reviewable commits are the whole point — don't let multiple blocks pile up.

**Rule:** If Claude Code is on a wrong track for more than 5 minutes, stop it (Esc in the VS Code panel, or Ctrl+C in the terminal), reset context, and re-prompt with more specifics. Never let it sprawl.

---

## 🌅 Block 0 (Friday Night) — Project sanity check

```
Read CLAUDE.md, README.md, and SATURDAY_PLAN.md. Then verify:
1. All Python files in src/ and notebooks/ parse without syntax errors
2. The data/ directory contains the 3 expected M5 CSVs (sales_train_evaluation.csv, calendar.csv, sell_prices.csv)
3. All requirements from requirements.txt are installed in the active environment

Report findings. Do not modify any files.
```

---

## 🌅 Block 1 (Sat 9:30–12:30) — Notebook 1: Data + Quality + EDA

### Step 1.1 — Convert and run

```
Convert notebooks/01_data_quality_eda.py to notebooks/01_data_quality_eda.ipynb using jupytext, then execute it in place with `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_quality_eda.ipynb` so the outputs land in the source file (this is what gets committed to GitHub for inline rendering). If it fails at any step, stop, report the error, and propose a fix before applying.
```

### Step 1.2 — If load step is slow

```
The M5 load step in notebook 01 is taking >2 minutes on my machine. Profile src/data_loader.py specifically the reshape_to_long function. If it's the melt operation, propose a faster alternative using pd.wide_to_long or a chunked approach. Don't change behavior — only performance.
```

### Step 1.3 — Verify outputs

```
After notebook 01 finishes, verify these files exist and report their row counts:
- data/weekly_network_volumes.parquet
- data/weekly_region_channel.parquet
- data/cleaning_audit_log.csv
- data/validation_results.csv

Also list the PNG files saved to presentation/. Confirm the validation_results CSV shows all checks passing.
```

### Step 1.4 — Polish the EDA charts

```
The 3 charts saved by notebook 01 (UPP trend, decomposition, correlation heatmap) are functional but plain. Improve them:
- Use figure size (12, 5) for time series, (10, 8) for the heatmap
- Title fonts at size 14 bold
- Add subtle gridlines
- Use a consistent color palette (steelblue / darkslategray for primary lines)
- Save as PNG at dpi=150 with bbox_inches='tight'

Don't change the underlying analysis — only the visualization style.
```

---

## 🌞 Block 2 (Sat 1:00–3:30) — Notebook 2: Modeling

### Step 2.1 — Run modeling notebook

```
Run notebooks/02_modeling.ipynb (convert from .py if needed). Report Traditional Error % and WMAPE % for each of the 3 models in the scorecard. If LightGBM fails or takes more than 5 minutes, stop it and report — we have a documented fallback to drop it.
```

### Step 2.2 — If Prophet is slow on lag analysis

```
The lag analysis cell in notebook 02 is fitting Prophet 48 times and taking >10 minutes. Reduce the loop to lags [1, 2, 4] and use only the last 8 test weeks (instead of 12). Update the chart accordingly. The story still works — we're just trimming compute.
```

### Step 2.3 — If LightGBM scores worse than Prophet

```
The LightGBM scorecard shows worse WMAPE than Prophet. This is expected on weekly aggregated data with strong yearly seasonality.

Add a markdown cell after the scorecard explaining:
"On weekly aggregated network-level volume, Prophet's yearly seasonality + US retail holidays component captures most of the signal. Tree-based ML models like LightGBM typically show their value at finer granularity (daily, FC-level, or SKU-level) where complex feature interactions matter more. This is a finding, not a failure — and it directly informs where we'd invest modeling effort next: at the FC × channel × week level."

Frame it as a finding in the deck too.
```

### Step 2.4 — Lag analysis matches WPR format

```
Show me the saved lag_analysis.csv. The format should mirror what WPR reports use: rows = lag (1, 2, 3, 4 weeks), columns = Traditional Error % and WMAPE %. If the format is off, fix it. Add a sentence to the markdown above this cell calling out: "This matches the WPR lag-analysis format and shows accuracy degradation as forecast horizon extends — the same pattern reported there."
```

---

## 🌆 Block 3 (Sat 3:45–6:00) — Notebook 3: Business Apps

### Step 3.1 — Run business apps notebook

```
Run notebooks/03_business_applications.ipynb end-to-end. Report:
1. The FedEx HD contract status for each forecasted month (ON TRACK / BELOW MIN / ABOVE MAX)
2. The OnTrac Tier 2 alert details (current rolling avg, breach week if any)
3. The cost-shift simulator weekly savings number and the annualized number
4. Confirm the 4-panel executive dashboard saved correctly to presentation/09_executive_dashboard.png
```

### Step 3.2 — Tune contract thresholds if numbers look implausible

```
The FedEx contract monitor shows monthly forecasted volumes [paste values]. Compare these to the contract band (19.5M to 26.8M packages/month). If forecasted volumes are far outside this band (e.g., ours are 500K and contract is 20M), the threshold values in src/business.py are not aligned to our M5-derived volume scale.

Adjust the constants FEDEX_HD_CONTRACT_MIN_MONTHLY_PACKAGES and FEDEX_HD_CONTRACT_MAX_MONTHLY_PACKAGES so that:
- The forecasted monthly volumes fall inside the band roughly 70% of the time
- Some months are flagged as BELOW MIN or ABOVE MAX (otherwise the monitor isn't doing anything)
- Update the threshold comments to clarify these are rescaled from real WPR numbers to fit the M5 dataset scale

Same for OnTrac Tier 2 — ONTRAC_TIER_2_WEEKLY_THRESHOLD should be set so the monitor occasionally fires.
```

### Step 3.3 — Polish the executive dashboard

```
Open presentation/09_executive_dashboard.png. The 4-panel layout has these issues to fix:
- Panel 3 (model scorecard table) — make sure column headers wrap correctly
- Panel 4 (FedEx bars) — rotate x-axis labels 30 degrees, not 45
- Add a footer line at the bottom: "ParcelCast | M5 data with parcel-domain reframing | Built [today's date]"
- Increase suptitle font size to 16

Re-save at dpi=150.
```

---

## 🌙 Block 4 (Sat 6:30–7:30) — Reproducibility check

```
Critical reproducibility check. Do the following in order:

1. Delete all files from data/ EXCEPT the 3 raw M5 CSVs. List what got deleted.
2. Delete all PNG files from presentation/.
3. Restart and run all 3 notebooks in order: 01 → 02 → 03.
4. Confirm:
   - All 3 notebooks ran without errors
   - All expected parquet files are back in data/
   - All expected PNG files are back in presentation/
   - The model scorecard CSV is reproducible (same numbers as before)

If anything fails, fix it now. This is the most important check of the weekend — a non-reproducible repo is unprofessional.
```

After Claude Code confirms reproducibility, commit and push:

```bash
git add -A && git commit -m "Sat EOD: all notebooks reproducible, all artifacts saved"
git push
```

---

## 🌅 Block 5 (Sun 12:30–2:00) — Notebook polish

```
Polish all 3 notebooks for presentation. For each notebook:

1. Verify the first markdown cell starts with the project emoji 📦, the notebook title, and a "What this notebook does" 2-3 sentence summary
2. Every code cell must have a markdown cell above it with at minimum a section header
3. Markdown cells should be 1-3 sentences focused on the WHY not the HOW
4. Remove any "scratch" or debugging cells
5. Ensure final cells of each notebook print a clear "Done" summary line listing what was saved

Don't change any logic — only narrative quality. Show me a diff summary before applying.
```

---

## 🌙 Block 6 (Sun 3:30–4:30) — README screenshots + final push

```
Update README.md to add a "Screenshots" section after the TL;DR. Embed these images from presentation/ in markdown:
- 02_decomposition.png as "Time series decomposition"
- 05_forecast_vs_actuals.png as "Forecast vs actuals (12-week test)"
- 06_lag_analysis.png as "Lag analysis (matches WPR format)"
- 09_executive_dashboard.png as "Executive dashboard"

Use relative paths like ![alt](presentation/02_decomposition.png).

Also fill in the "Key Findings" section with actual values from data/model_scorecard.csv, data/lag_analysis.csv, data/fedex_contract_status.csv, and data/cost_optimization.csv.

Show me the diff before applying.
```

After applying:

```bash
git add -A && git commit -m "Final: README with screenshots and findings"
git push
```

---

## 🆘 Recovery prompts

### "Notebook X is broken and I don't know why"

```
Notebook X failed with the following error:

[paste full traceback]

Steps to take:
1. Identify the failing cell and the upstream cause
2. Check whether the issue is in src/ or in the notebook itself
3. Propose the smallest possible fix
4. Wait for my approval before applying

Do not refactor unrelated code.
```

### "I'm running out of time and need to ship"

```
I'm short on time. Apply the cut list from SATURDAY_PLAN.md:
1. Drop LightGBM cells from notebook 02
2. Drop the cost-shift simulator from notebook 03
3. Update the README "What's here" section to reflect what shipped
4. Verify notebooks 01 and 02 still run end-to-end

Commit with message: "Apply weekend cut list — Prophet-only, FedEx + OnTrac monitors only"
```

### "Help me draft the email to Sam"

```
Draft a Monday-morning email to Sam (Director of Parcel Forecasting at a major retailer) sharing this project. Use the template in SATURDAY_PLAN.md as a starting point but personalize based on what actually shipped (check git log, look at README, look at presentation/). Three short paragraphs. No hard sell. Include the GitHub URL.
```

---

## 📝 General Claude Code rules I'm holding myself to

- Review every file change before accepting
- Commit after every block (6 commits minimum across the weekend)
- If a block goes 30% over its time budget, invoke the cut list
- Never let Claude Code change the 3 things in CLAUDE.md "Things to NOT do"
- Always run notebooks myself — don't trust Claude Code's reports of "it ran successfully" without verifying
