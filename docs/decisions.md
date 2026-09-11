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

## 2026-09-12 — HA-ViD has arrived; what was taken and what was left

The authors granted access (password-protected Dropbox folder; link and password stay out of the
repository). Downloaded into `data/external/ha-vid/`: the temporal annotations, the
action-segmentation benchmark folder (official splits, mapping, per-frame ground truth, I3D
features) and the RGB videos. Left on the server: depth, skeletons, the object-detection frames
(110 GB, outside the non-goals), the action-recognition clip datasets and the pretrained
checkpoints. Everything measured about the delivery is in `reports/havid_audit.{json,md}`.

## 2026-09-12 — the HA-ViD frozen split follows the official subject split

`splits/ha-vid/` is frozen with the same hash contract as IndustReal: **test** = the official test
subjects (S02, S04, S05, S19, S22, S24, S26; 41 recordings, 123 videos) so that the paper's
MS-TCN/DTGRM/BCN numbers remain comparable; **val** = 6 subjects drawn with seed 0 from the
official train subjects that have only three annotated recordings (S01, S08, S10, S12, S18, S30),
so the recording-rich subjects stay in **train** (17 subjects, 143 recordings). The one annotated
recording that the official bundles list nowhere (`S02A08I21`, a test subject) is excluded from
every split. The test split is not evaluated without an explicit decision (same rule as
IndustReal); development results are reported on val.

## 2026-09-12 — trigger B of ADR 0001 fires: HA-ViD's native errors are too few for a headline

The released annotations carry the paper's `wrong` label as `w`, primitive-task level only:
67 segments in all, 16 inside the official test subjects (7 of their 41 recordings). That is under
the 20-segment threshold set in ADR 0001, so HA-ViD stays the main dataset, the native error
table will report exact counts with Wilson intervals and be labelled underpowered, and the headline
violation metrics will come from the synthetic order-violation table, marked as synthetic.
