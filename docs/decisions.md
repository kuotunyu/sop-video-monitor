# Decisions log

Dated, one entry per decision that is not derivable from the code. ADR-style long-form records
live in `docs/decisions/`; this file is the running index. Newest last.

## 2026-09-03 — datasets and licences

See [`decisions/0001-dataset-and-licences.md`](decisions/0001-dataset-and-licences.md): HA-ViD is
the main dataset (CC BY-NC 4.0, request form), IMPACT the fallback, IndustReal the metric donor
(Apache-2.0). Raw data, features and weights never enter the repository.

## 2026-09-11 — IndustReal is development data until HA-ViD arrives

HA-ViD's public files contain no videos or annotations, so all development happens on IndustReal.
Its numbers are reported as development results on the validation split and never enter an
HA-ViD table; the frozen IndustReal test split (`splits/industreal/`, hashed) is not evaluated
without an explicit decision.

## 2026-09-11 — frame semantics of the IndustReal labels

Measured, not assumed: label frame indices are 0-based JPEG names of the 10 fps stream, segments
are half-open, overlapping action segments resolve to the latest onset. One recording
(`11_assy_0_1`) has JPEG names offset by 29 frames from its mp4 (pixel-verified); the loader
derives and applies the offset from the archive's name range.

## 2026-09-11 — online metrics follow the reference code, with three named deviations

`metrics/online.py` reproduces IndustReal's `PSR/psr_utils.py` (POS with substitute cost 2 and
ground-truth-length normalisation, time-aware system-level F1, delays only for on-time
detections). Deviations, all documented in the module: closest-earlier fallback when no later
candidate exists, no 100-frame substitute when nothing is detected, "at or after" matching for
repeated step ids (the reference's strict "after" makes the ground truth imperfect against itself).

## 2026-09-11 — every level of model selection uses the out-of-fold event metric

Decoder settings (v6), latency budget (v7) and MS-TCN++ training length (v11) are chosen on
out-of-fold predictions of the training recordings, by the decoded F1 the run reports. In-sample
selection (v5) did not transfer across participants; held-out BCE as an epoch criterion (v10) chose
a head that decodes worse. Frame-level proxies are not used for selection.

## 2026-09-11 — a 30 s latency budget is the operating point

The out-of-fold F1-vs-delay fronts (v7–v9) show that precision is bought with seconds; a 30 s
budget costs no F1 and returns 4–6 s. Budgets below ≈ 20 s are not reachable with frozen frame
features; ViT-B/14 lowered the MS-TCN++ floor from 17 s to 12 s.

## 2026-09-12 — DINOv2 ViT-B/14 is the default frame feature

ViT-S → ViT-B moved the whole front (v8); ViT-B → ViT-L helped only the linear head, cost 2.2× the
extraction time and a 1.2 GB checkpoint (v9). ViT-L features are kept on disk but are not the
default.

## 2026-09-12 — procedure knowledge is learned from the train labels, not hand-written

`sop/industreal/learned_precedence_{assy,main}.json` are learned from the 36 train recordings
(edges that hold in every co-occurring recording, min support 3) and used only for checks; the
decoder prior learns which components move and how a recording starts. Three val recordings
violate the learned order in their ground truth; a precision-tuned decoder flags at most one.

## 2026-09-12 — repository layout tidy-up

`reports/data_audit.json` holds the shared on-disk audit (moved out of `industreal_dev_v1/`);
empty placeholder directories from the W0 skeleton were removed; `reports/README.md` is the
navigation table with the status of every run; historical PSR protocols stay reproducible through
`make psr-lopo / psr-train / psr-nested`, the current best through `make psr`.
