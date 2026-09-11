# Entry points (spec 8.3). `make reproduce-lite` is what CI runs on a clean checkout without data;
# `make reproduce` is the full data + GPU path in dependency order.

.PHONY: reproduce reproduce-lite test lint verify-splits audit features features-vits extract-psr learn-sop psr psr-latency baseline mstcn psr-lopo psr-train psr-nested

UV ?= uv
INDUSTREAL ?= data/external/industreal
FEATURES ?= artifacts/features/industreal/dinov2_vitb14_s1
FEATURES_VITS ?= artifacts/features/industreal/dinov2_vits14_s1

# ---- CI path (no data, no torch) -------------------------------------------------------------

reproduce-lite:  ## Recompute every committed run from its prediction table, check tables and SOP checks, verify the split hash, run the unit tests.
	$(UV) run sop-monitor reproduce-lite
	$(UV) run sop-monitor verify-splits --directory splits/industreal
	$(UV) run pytest -q

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

verify-splits:
	$(UV) run sop-monitor verify-splits --directory splits/industreal

# ---- Data path: needs the local IndustReal copy and `uv sync --all-extras --group baseline` ----

audit:  ## Probe local videos, cross-check labels, inventory HA-ViD public files, refresh data/manifest.json.
	$(UV) run sop-monitor audit-industreal --root $(INDUSTREAL) --out reports/data_audit.json --manifest data/manifest.json

features:  ## Frozen DINOv2 ViT-B/14 frame features for the train and val videos (current default; test untouched).
	$(UV) run sop-monitor extract-features --root $(INDUSTREAL) --split train --split val --model dinov2_vitb14 --stride 1 --batch-size 128

features-vits:  ## ViT-S/14 features (needed only to reproduce v1-v7).
	$(UV) run sop-monitor extract-features --root $(INDUSTREAL) --split train --split val --model dinov2_vits14 --stride 1

extract-psr:  ## Pull only the PSR label CSVs (+ JPEG name ranges) out of the six recording archives.
	$(UV) run sop-monitor extract-psr-labels --archive $(INDUSTREAL)/val_p1.zip --archive $(INDUSTREAL)/val_p2.zip --archive $(INDUSTREAL)/train_p1.zip --archive $(INDUSTREAL)/train_p2.zip --archive $(INDUSTREAL)/train_p3.zip --archive $(INDUSTREAL)/train_p4.zip --out $(INDUSTREAL)/psr

learn-sop:  ## Learn the step precedence graphs from the train-split PSR labels (sop/industreal/learned_precedence_*.json).
	$(UV) run sop-monitor learn-sop --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --out sop/industreal

psr:  ## Current best PSR configuration (industreal_dev_v11): ViT-B/14, MS-TCN++ + prior/dwell, 30 s budget, nested decoder + epoch selection by decoded F1, seeds 0-2.
	$(UV) run sop-monitor train-psr --features $(FEATURES) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 30 --head mstcn --decoder prior_dwell --mstcn-epoch-grid 20 --mstcn-epoch-grid 40 --mstcn-epoch-grid 60 --mstcn-epoch-grid 80 --mstcn-epoch-criterion decoded_f1 --out $(or $(OUT),reports/industreal_dev_v11_psr_epochsel_f1)
	$(UV) run sop-monitor check-psr-run --run $(or $(OUT),reports/industreal_dev_v11_psr_epochsel_f1) --run-name mstcn_prior_dwell_cap30_s0 --out sop_checks_mstcn_prior_dwell_cap30_s0.json

psr-latency:  ## Head x decoder ablation under 15 s / 30 s budgets (industreal_dev_v7/v8/v9; set FEATURES and OUT).
	$(UV) run sop-monitor train-psr --features $(FEATURES) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --delay-cap 15 --delay-cap 30 --out $(or $(OUT),reports/industreal_dev_v8_psr_vitb)

reproduce: audit features-vits baseline mstcn features extract-psr learn-sop psr reproduce-lite  ## Full path, about 1.5 h on an RTX 4090.

# ---- Offline TAS line (industreal_dev_v1, v2; ViT-S/14 features) ------------------------------

baseline:
	$(UV) run sop-monitor train-baseline --features $(FEATURES_VITS) --labels-dir $(INDUSTREAL)/labels --out reports/industreal_dev_v1

mstcn:
	$(UV) run sop-monitor train-mstcn --features $(FEATURES_VITS) --labels-dir $(INDUSTREAL)/labels --out reports/industreal_dev_v2_mstcn

# ---- Historical PSR protocols, kept so the older reports stay reproducible --------------------

psr-lopo:  ## industreal_dev_v3: leave-one-participant-out inside val, linear head, plain decoder.
	$(UV) run sop-monitor train-psr --features $(FEATURES_VITS) --psr-dir $(INDUSTREAL)/psr --head linear --decoder plain --out reports/industreal_dev_v3_psr

psr-train:  ## industreal_dev_v5: fit on the 36 train recordings, in-sample decoder selection.
	$(UV) run sop-monitor train-psr --features $(FEATURES_VITS) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --out reports/industreal_dev_v5_psr_train

psr-nested:  ## industreal_dev_v6: nested decoder selection, seeds 0-2, no latency budget.
	$(UV) run sop-monitor train-psr --features $(FEATURES_VITS) --psr-dir $(INDUSTREAL)/psr --train-split splits/industreal/train.csv --eval-split splits/industreal/val.csv --selection nested --seed 0 --seed 1 --seed 2 --out reports/industreal_dev_v6_psr_nested
