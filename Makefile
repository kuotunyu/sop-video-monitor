# Entry points (spec 8.3). `make reproduce-lite` is what CI runs on a clean checkout without data;
# `make reproduce` is the full data + GPU path in dependency order.

.PHONY: reproduce reproduce-lite test lint verify-splits audit audit-havid freeze-havid features features-vits extract-psr learn-sop psr psr-latency baseline mstcn psr-lopo psr-train psr-nested features-havid i3d-havid havid-tas havid-tas-v3 havid-tas-i3d learn-havid-sop havid-sop tune-havid-sop havid-sop-v5 havid-sop-v6 havid-tas-seeds havid-seeds-summary tune-havid-sop-sheet havid-sop-v8 havid-tas-lookahead havid-lookahead-summary review-queue review-ui

UV ?= uv
INDUSTREAL ?= data/external/industreal
HAVID ?= data/external/ha-vid
FEATURES ?= artifacts/features/industreal/dinov2_vitb14_s1
FEATURES_VITS ?= artifacts/features/industreal/dinov2_vits14_s1

# ---- CI path (no data, no torch) -------------------------------------------------------------

reproduce-lite:  ## Recompute every committed run from its prediction table, check tables and SOP checks, verify the split hashes, run the unit tests.
	$(UV) run sop-monitor reproduce-lite
	$(UV) run sop-monitor verify-splits --directory splits/industreal
	$(UV) run sop-monitor verify-splits --directory splits/ha-vid
	$(UV) run pytest -q

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

verify-splits:
	$(UV) run sop-monitor verify-splits --directory splits/industreal
	$(UV) run sop-monitor verify-splits --directory splits/ha-vid

# ---- Data path: needs the local IndustReal copy and `uv sync --all-extras --group baseline` ----

audit:  ## Probe local IndustReal videos, cross-check labels, inventory HA-ViD public files, refresh data/manifest.json (hashes every archive, several minutes).
	$(UV) run sop-monitor audit-industreal --root $(INDUSTREAL) --out reports/data_audit.json --manifest data/manifest.json

audit-havid:  ## HA-ViD W1 audit: temporal annotations x official split x probed mp4s -> reports/havid_audit.json (needs the extracted HAViD_rgb).
	$(UV) run sop-monitor audit-havid --temporal $(HAVID)/HAViD_temporalAnnotation.zip --official $(HAVID)/ActionSegmentation_data.zip --rgb-dir $(HAVID)/HAViD_rgb --out reports/havid_audit.json

freeze-havid:  ## Freeze splits/ha-vid from the official subject split (test = official test subjects, val = 6 light train subjects, seed 0).
	$(UV) run sop-monitor freeze-splits --dataset ha-vid --official $(HAVID)/ActionSegmentation_data.zip --temporal $(HAVID)/HAViD_temporalAnnotation.zip --out-dir splits

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

# ---- HA-ViD line: needs the delivered archives (data/README.md) and splits/ha-vid --------------

features-havid:  ## DINOv2 ViT-B/14 frame features for the HA-ViD train + val videos (483 mp4s, three views; test untouched).
	$(UV) run sop-monitor extract-features --dataset ha-vid --split-dir splits/ha-vid --rgb-dir $(HAVID)/HAViD_rgb --out artifacts/features/ha-vid --model dinov2_vitb14 --stride 1 --batch-size 128

i3d-havid:  ## Export the official I3D features of the train + val videos into the same npz cache layout.
	$(UV) run sop-monitor export-official-features --official $(HAVID)/ActionSegmentation_data.zip --split-dir splits/ha-vid --out artifacts/features/ha-vid/i3d_official

havid-tas:  ## havid_dev_v1: per-view MS-TCN++ (causal + offline) and late fusion on DINOv2 ViT-B/14, primitive tasks, both hands.
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --out reports/havid_dev_v1_tas_lh
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand rh --out reports/havid_dev_v1_tas_rh

havid-tas-v3:  ## havid_dev_v3: as v1 (DINOv2 ViT-B/14) but epochs selected by val F1@10 and three parameter-free fusion rules.
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand lh --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_lh
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand rh --selection-metric f1@10 --out reports/havid_dev_v3_tas_f1sel_rh

learn-havid-sop:  ## Learn the per-plate precedence graphs, mandatory steps and duration bounds from splits/ha-vid/train.csv (sop/ha-vid/*.json).
	$(UV) run sop-monitor learn-havid-sop --temporal $(HAVID)/HAViD_temporalAnnotation.zip --split-dir splits/ha-vid --out sop/ha-vid

havid-sop:  ## havid_dev_v4: SOP checks, synthetic violations and native w on val, from the v3 fusion_causal predictions of both hands.
	$(UV) run sop-monitor havid-sop --pred-lh reports/havid_dev_v3_tas_f1sel_lh --pred-rh reports/havid_dev_v3_tas_f1sel_rh --run-name fusion_causal --out reports/havid_dev_v4_sop_synthetic

tune-havid-sop:  ## Leave-one-subject-out grid over (min support, min agreement) and duration quantiles on the train split -> reports/havid_dev_v5_sop_tuned/oof.json.
	$(UV) run sop-monitor tune-havid-sop --temporal $(HAVID)/HAViD_temporalAnnotation.zip --split-dir splits/ha-vid --out reports/havid_dev_v5_sop_tuned/oof.json

havid-sop-v5:  ## havid_dev_v5: the v4 evaluation with the knowledge learned under the settings tune-havid-sop selected (sop/ha-vid/tuned; set SUPPORT, AGREEMENT, LOWQ, HIGHQ).
	$(UV) run sop-monitor learn-havid-sop --temporal $(HAVID)/HAViD_temporalAnnotation.zip --split-dir splits/ha-vid --out sop/ha-vid/tuned --min-support $(or $(SUPPORT),20) --min-agreement $(or $(AGREEMENT),1.0) --low-q $(or $(LOWQ),0.01) --high-q $(or $(HIGHQ),0.99)
	$(UV) run sop-monitor havid-sop --pred-lh reports/havid_dev_v3_tas_f1sel_lh --pred-rh reports/havid_dev_v3_tas_f1sel_rh --run-name fusion_causal --graphs sop/ha-vid/tuned --out reports/havid_dev_v5_sop_tuned

havid-tas-i3d:  ## havid_dev_v2: the same protocol on the official I3D features (control).
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/i3d_official --hand lh --out reports/havid_dev_v2_i3d_lh
	$(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/i3d_official --hand rh --out reports/havid_dev_v2_i3d_rh

reproduce: audit features-vits baseline mstcn features extract-psr learn-sop psr reproduce-lite  ## Full IndustReal path, about 1.5 h on an RTX 4090.

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

havid-sop-v6:  ## havid_dev_v6: v5 knowledge with the predicted segments smoothed (blips shorter than the train 1st-percentile step length, 11 frames, absorbed).
	$(UV) run sop-monitor havid-sop --pred-lh reports/havid_dev_v3_tas_f1sel_lh --pred-rh reports/havid_dev_v3_tas_f1sel_rh --run-name fusion_causal --graphs sop/ha-vid/tuned --min-segment-frames 11 --out reports/havid_dev_v6_sop_smoothed

havid-tas-seeds:  ## Seeds 1 and 2 of the v3 protocol for both hands (havid_dev_v3_tas_f1sel_{lh,rh}_s{1,2}); seed 0 is havid_dev_v3_tas_f1sel_{lh,rh}.
	for seed in 1 2; do for hand in lh rh; do $(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand $$hand --selection-metric f1@10 --seed $$seed --out reports/havid_dev_v3_tas_f1sel_$${hand}_s$$seed || exit 1; done; done

havid-seeds-summary:  ## havid_dev_v7: mean +- std over seeds 0-2 of the v3 protocol, per hand (reports/havid_dev_v7_tas_seeds_{lh,rh}).
	$(UV) run sop-monitor summarise-havid-seeds --run reports/havid_dev_v3_tas_f1sel_lh --run reports/havid_dev_v3_tas_f1sel_lh_s1 --run reports/havid_dev_v3_tas_f1sel_lh_s2 --out reports/havid_dev_v7_tas_seeds_lh
	$(UV) run sop-monitor summarise-havid-seeds --run reports/havid_dev_v3_tas_f1sel_rh --run reports/havid_dev_v3_tas_f1sel_rh_s1 --run reports/havid_dev_v3_tas_f1sel_rh_s2 --out reports/havid_dev_v7_tas_seeds_rh

tune-havid-sop-sheet:  ## Leave-one-subject-out tuning at instruction-sheet granularity (hole index and tool dropped) -> reports/havid_dev_v8_sop_sheet/oof.json.
	$(UV) run sop-monitor tune-havid-sop --temporal $(HAVID)/HAViD_temporalAnnotation.zip --split-dir splits/ha-vid --granularity sheet --out reports/havid_dev_v8_sop_sheet/oof.json

havid-sop-v8:  ## havid_dev_v8: SOP checks at instruction-sheet granularity with the knowledge tuned out of fold (set SUPPORT, AGREEMENT, LOWQ, HIGHQ from oof.json).
	$(UV) run sop-monitor learn-havid-sop --temporal $(HAVID)/HAViD_temporalAnnotation.zip --split-dir splits/ha-vid --granularity sheet --out sop/ha-vid/sheet --min-support $(or $(SUPPORT),20) --min-agreement $(or $(AGREEMENT),1.0) --low-q $(or $(LOWQ),0.01) --high-q $(or $(HIGHQ),0.99)
	$(UV) run sop-monitor havid-sop --pred-lh reports/havid_dev_v3_tas_f1sel_lh --pred-rh reports/havid_dev_v3_tas_f1sel_rh --run-name fusion_causal --graphs sop/ha-vid/sheet --granularity sheet --out reports/havid_dev_v8_sop_sheet

havid-tas-lookahead:  ## havid_dev_v9 inputs: causal networks with 15 / 45 / 90 frames of look-ahead (1 / 3 / 6 s output delay), seeds 0-2, both hands, no offline twins.
	for seed in 0 1 2; do for la in 15 45 90; do for hand in lh rh; do $(UV) run sop-monitor train-havid-tas --features artifacts/features/ha-vid/dinov2_vitb14_s1 --hand $$hand --selection-metric f1@10 --seed $$seed --lookahead-frames $$la --no-offline --out reports/havid_dev_v9_tas_la$${la}_$${hand}_s$$seed || exit 1; done; done; done

havid-lookahead-summary:  ## havid_dev_v9: mean +- std over seeds per look-ahead and hand.
	for la in 15 45 90; do for hand in lh rh; do $(UV) run sop-monitor summarise-havid-seeds --run reports/havid_dev_v9_tas_la$${la}_$${hand}_s0 --run reports/havid_dev_v9_tas_la$${la}_$${hand}_s1 --run reports/havid_dev_v9_tas_la$${la}_$${hand}_s2 --out reports/havid_dev_v9_tas_lookahead_la$${la}_$${hand} || exit 1; done; done

# ---- W5 review (docs/review.md) -------------------------------------------------------------

review-queue:  ## Deviation queue of the v8 SOP run's recogniser output (artifacts/review/, ignored by git).
	$(UV) run sop-monitor build-review-queue --run reports/havid_dev_v8_sop_sheet --source pred --out artifacts/review/v8_pred_queue.jsonl

review-ui:  ## Local review page on http://127.0.0.1:8765/ for the v8 queue (set REVIEWER).
	$(UV) run sop-monitor review-ui --queue artifacts/review/v8_pred_queue.jsonl --decisions artifacts/review/v8_pred_decisions.jsonl --reviewer "$(REVIEWER)"
