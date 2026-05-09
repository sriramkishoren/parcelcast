# ParcelCast — common commands
# Run from project root: `make <target>`

.PHONY: help install convert run-01 run-02 run-03 run-all clean reset check

help:
	@echo "ParcelCast targets:"
	@echo "  install     — pip install -r requirements.txt"
	@echo "  convert     — convert .py notebooks to .ipynb (jupytext)"
	@echo "  run-01      — execute notebook 01 (data + quality + EDA)"
	@echo "  run-02      — execute notebook 02 (modeling)"
	@echo "  run-03      — execute notebook 03 (business apps)"
	@echo "  run-all     — execute all 3 notebooks in order"
	@echo "  check       — verify all expected outputs exist"
	@echo "  clean       — delete generated PNGs + intermediate parquet (keep raw M5)"
	@echo "  reset       — clean + run-all (full reproducibility test)"

install:
	pip install -r requirements.txt

convert:
	jupytext --to ipynb notebooks/*.py

run-01:
	jupyter nbconvert --to notebook --execute notebooks/01_data_quality_eda.ipynb --inplace

run-02:
	jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --inplace

run-03:
	jupyter nbconvert --to notebook --execute notebooks/03_business_applications.ipynb --inplace

run-all: run-01 run-02 run-03

check:
	@echo "Checking expected outputs..."
	@test -f data/weekly_network_volumes.parquet && echo "  ✓ weekly_network_volumes.parquet" || echo "  ✗ MISSING: weekly_network_volumes.parquet"
	@test -f data/weekly_region_channel.parquet && echo "  ✓ weekly_region_channel.parquet" || echo "  ✗ MISSING: weekly_region_channel.parquet"
	@test -f data/cleaning_audit_log.csv && echo "  ✓ cleaning_audit_log.csv" || echo "  ✗ MISSING: cleaning_audit_log.csv"
	@test -f data/validation_results.csv && echo "  ✓ validation_results.csv" || echo "  ✗ MISSING: validation_results.csv"
	@test -f data/model_scorecard.csv && echo "  ✓ model_scorecard.csv" || echo "  ✗ MISSING: model_scorecard.csv"
	@test -f data/lag_analysis.csv && echo "  ✓ lag_analysis.csv" || echo "  ✗ MISSING: lag_analysis.csv"
	@test -f data/network_forecast.parquet && echo "  ✓ network_forecast.parquet" || echo "  ✗ MISSING: network_forecast.parquet"
	@test -f data/fedex_contract_status.csv && echo "  ✓ fedex_contract_status.csv" || echo "  ✗ MISSING: fedex_contract_status.csv"
	@test -f data/cost_optimization.csv && echo "  ✓ cost_optimization.csv" || echo "  ✗ MISSING: cost_optimization.csv"
	@echo ""
	@echo "Charts in presentation/:"
	@ls -1 presentation/*.png 2>/dev/null | wc -l | xargs -I {} echo "  {} PNG files"

clean:
	rm -f data/*.parquet data/*.csv
	rm -f presentation/*.png
	rm -f notebooks/.ipynb_checkpoints/*
	@echo "Cleaned. Raw M5 files in data/ are preserved."

reset: clean run-all check
	@echo ""
	@echo "Reproducibility test complete."
