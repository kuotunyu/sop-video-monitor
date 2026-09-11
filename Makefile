# Single entry points (spec 8.3). `make reproduce` is the full data+GPU path;
# `make reproduce-lite` is what CI runs on a clean checkout without data.

.PHONY: reproduce reproduce-lite test lint audit features baseline mstcn extract-psr psr psr-train psr-nested

UV ?= uv
INDUSTREAL ?= data/external/industreal
RUN ?= reports/industreal_dev_v1
FEATURES ?= artifacts/features/industreal/dinov2_vits14_s1

reproduce: audit features baseline reproduce-lite  ## Full development path: needs the local IndustReal copy, the `baseline` group and a GPU (CPU works, slower).

reproduce-lite:  ## Recompute every committed run's metrics from its prediction table, check tables.md, run the unit tests.
	$(UV) run sop-monitor reproduce-lite
	$(UV) run pytest -q

audit:  ## Probe local videos, cross-check labels, inventory HA-ViD public files, refresh data/manifest.json.
	$(UV) run sop-monitor audit-industreal --root $(INDUSTREAL) --out $(RUN)/data_audit.json --manifest data/manifest.json

features:  ## Cache frozen DINOv2 ViT-S/14 frame features for the train and val videos (test is not touched).
	$(UV) run sop-monitor extract-features --root $(INDUSTREAL) --split train --split val --model dinov2_vits14 --stride 1

baseline:  ## Train the linear head on train, select and report on val, write $(RUN)/.
	$(UV) run sop-monitor train-baseline --features $(FEATURES) --labels-dir $(INDUSTREAL)/labels --out $(RUN)

mstcn:  ## Causal vs non-causal MS-TCN++ on the same features, write reports/industreal_dev_v2_mstcn/.
	$(UV) run sop-monitor train-mstcn --features $(FEATURES) --labels-dir $(INDUSTREAL)/labels --out reports/industreal_dev_v2_mstcn

extract-psr:  ## Pull only the PSR label CSVs out of the val recording archives (val_p1.zip, val_p2.zip).
	$(UV) run sop-monitor extract-psr-labels --archive $(INDUSTREAL)/val_p1.zip --archive $(INDUSTREAL)/val_p2.zip --out $(INDUSTREAL)/psr

psr:  ## Leave-one-participant-out step-completion baseline on val, write reports/industreal_dev_v3_psr/.
	$(UV) run sop-monitor train-psr --features $(FEATURES) --psr-dir $(INDUSTREAL)/psr --out reports/industreal_dev_v3_psr

psr-train:  ## Step-completion heads fitted on the 36 train recordings, evaluated on val (needs train_p*.zip labels extracted).
	$(UV) run sop-monitor train-psr --features $(FEATURES) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --out reports/industreal_dev_v5_psr_train

psr-nested:  ## Same as psr-train but decoders chosen on out-of-fold train predictions, seeds 0-2, write reports/industreal_dev_v6_psr_nested/.
	$(UV) run sop-monitor train-psr --features $(FEATURES) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --out reports/industreal_dev_v6_psr_nested

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
