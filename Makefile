# Single entry points (spec 8.3). `make reproduce` is the full data+GPU path;
# `make reproduce-lite` is what CI runs on a clean checkout without data.

.PHONY: reproduce reproduce-lite test lint audit features baseline

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

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
