# 🗓 ParcelCast Weekend Playbook (with Claude Code)

> Tight, hour-by-hour. Designed to run with Claude Code as your engineering partner.

---

## 🛠 The Claude Code workflow

Two terminals, side-by-side:
- **Terminal A**: `claude` (your engineering partner)
- **Terminal B**: git + `make` commands + spot-checks

Every block = paste a prompt from `PROMPTS.md` → review the diff → run `make check` → commit.

You drive, Claude Code executes. Don't let Claude Code drive — ever.

---

## 🌃 Friday Night (60 min) — Setup

| Step | Time | Command |
|---|---|---|
| 1. Verify Claude Code | 5 min | `claude --version` |
| 2. Clone / scaffold project | 5 min | drop zip, `git init`, first commit |
| 3. Push to GitHub | 5 min | `gh repo create parcelcast --public --source=. --remote=origin --push` |
| 4. Python venv + install | 10 min | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| 5. Verify imports | 2 min | `python -c "import prophet, lightgbm, statsmodels; print('OK')"` |
| 6. Download M5 from Kaggle | 15 min | `kaggle competitions download -c m5-forecasting-accuracy -p data/ && cd data && unzip *.zip` |
| 7. Convert notebooks | 2 min | `make convert` |
| 8. Smoke test Claude Code | 10 min | `claude` → paste **PROMPTS.md Block 0** |
| 9. Commit + push | 5 min | `git add -A && git commit -m "Friday setup" && git push` |

**Why all this Friday?** Prophet install is the #1 weekend killer. Solve it tonight.

---

## ☀️ Saturday — Build (9 hrs with buffer)

### 9:00–9:30 — Setup polish
- Personalize README "Domain Mapping" section in your voice
- Fill in the metadata in CLAUDE.md if anything looks off
- Commit

### 9:30–12:30 — Block 1: Notebook 1 (Data + Quality + EDA)
**Use:** PROMPTS.md Block 1 (steps 1.1, 1.3, 1.4)
**Output:** parquet artifacts, audit log, validation suite passing, 3 charts saved
**Verify:** `make check` shows ✓ for all expected files

If load step is slow, use prompt 1.2.

### 12:30–1:00 — Lunch 🍕

### 1:00–3:30 — Block 2: Notebook 2 (Modeling)
**Use:** PROMPTS.md Block 2
**Output:** scorecard CSV, lag analysis CSV, forecast parquet, 3 charts
**Verify:** `make check` shows ✓ for `model_scorecard.csv` and `lag_analysis.csv`

If LightGBM is broken or worse than Prophet, use prompt 2.3 to frame it as a finding, not a failure.

### 3:30–3:45 — Walk break ☕
Get up. Look out a window. Don't open the laptop.

### 3:45–6:00 — Block 3: Notebook 3 (Business Apps)
**Use:** PROMPTS.md Block 3
**Output:** FedEx status CSV, cost optimization CSV, 4 charts including executive dashboard

**Critical:** the threshold values may need rescaling for M5 data. Use prompt 3.2 if numbers look implausible.

### 6:00–6:30 — Dinner 🍔

### 6:30–7:30 — Block 4: Reproducibility check (most important hour of the weekend)
**Use:** PROMPTS.md Block 4 OR run `make reset` directly

This deletes all generated artifacts and rebuilds from scratch. If anything fails, fix it now.

```bash
make reset
make check
```

If `make check` shows all ✓, commit + push:

```bash
git add -A && git commit -m "Sat EOD: all reproducible"
git push
```

### Saturday end-of-day checkpoint

Run this manually:

```bash
make check
ls presentation/*.png | wc -l   # Should be 9 or more
git log --oneline               # Should show 5+ commits today
```

If all green, **stop**. Sunday is for shipping.

---

## ☀️ Sunday — Ship (6 hrs)

### 9:00–9:30 — Storyboard the deck on paper
**Don't open Keynote/PPT yet.** Sketch all 8 slides. One bold takeaway per slide.

### 9:30–12:00 — Build the deck (use your existing infographic style: light minimalist, Sora + DM Sans, soft pastels, 900px)

Drop saved PNGs from `presentation/` directly. Don't over-design. 18 min/slide.

### 12:00–12:30 — **SHOCK TEST** + lunch 🌮
- Imagine sending the project to Pallavi *right now*
- Pride or panic? Cut, don't add.

### 12:30–2:00 — Block 5: Notebook polish
**Use:** PROMPTS.md Block 5

Each notebook gets clean markdown narrative. No logic changes. Show diff before applying.

### 2:00–3:30 — Practice 3x
- Round 1: identify rough patches
- Round 2: tighten transitions
- Round 3: confident delivery
- Target: 12 minutes (live always runs 25% over)

### 3:30–4:30 — Block 6: README + final push
**Use:** PROMPTS.md Block 6

Embed screenshots in README, fill in actual findings, commit, push.

### 4:30 — Done. Walk away.

Send the email Monday morning, NOT Sunday night. Sunday night sends look frantic.

---

## 🚨 Cut list — invoke ruthlessly if behind

| If by this time you haven't... | Use prompt |
|---|---|
| Sat 12:30 — finished Notebook 1 | "Skip the ADF stationarity test cell in notebook 01. It's not load-bearing for the story. Comment it out and continue." |
| Sat 3:30 — finished modeling | PROMPTS.md → Recovery → "I'm running out of time and need to ship" |
| Sat 6:00 — finished business apps | "Drop the cost-shift simulator cells from notebook 03. Update the dashboard to be 3-panel instead of 4-panel. Update README." |
| Sun 12:00 — slides aren't half done | Merge slides 4 + 5 into one. Ship 7, not 8. |

---

## 🔥 Risk register

| Risk | First line of defense |
|---|---|
| Prophet install fails | Did Friday-night install work? If yes, you're safe. If no, fall back to `statsforecast` (1 line install, similar API). |
| Lag analysis loop is too slow | PROMPTS.md prompt 2.2 — reduce lags + test weeks |
| Threshold values give weird outputs | PROMPTS.md prompt 3.2 — rescale for M5 data |
| LightGBM scores worse than Prophet | PROMPTS.md prompt 2.3 — frame as a finding |
| Notebook crashes mid-run Sunday | You committed every hour Saturday — reset and rerun |
| Claude Code keeps making bad changes | Stop, reset, paste CLAUDE.md again, re-prompt with more specifics |

---

## 📨 What to send Pallavi (Monday morning)

**Subject:** ParcelCast — quick weekend project on parcel volume forecasting

> Hi Pallavi,
>
> Following up on our conversation — I built a small project this weekend to demonstrate how I'd approach the forecasting work.
>
> I used the public M5 Walmart dataset and reframed it as a parcel-forecasting problem: stores → FCs, units → packages via UPP, plus carrier allocation across FedEx / UPS / IP / OnTrac.
>
> The repo walks through:
> 1. Data quality + cleaning, with an audit log of every transformation
> 2. Two models (Prophet + LightGBM) evaluated with Traditional Error and WMAPE — the team's metrics
> 3. A lag-analysis table mirroring the WW14 format
> 4. Three business-application views: FedEx HD contract monitor, OnTrac Tier 2 risk alert, carrier cost-shift simulator
>
> Repo: https://github.com/[username]/parcelcast
> 8-slide deck attached.
>
> Happy to walk you through it whenever works on your end.
>
> Thanks,
> Sriramkishore

---

## 🎤 The ONE sentence to lead the presentation with

> *"Real Walmart M5 data, cleaned with the team's metrics, two models compared, and forecasts mapped to FedEx contract compliance and OnTrac tier monitoring."*

That's it. Don't open with anything longer.

---

You've got this. Trust the prompts. Trust the time-boxes. Ship Sunday at 4:30. 💪
