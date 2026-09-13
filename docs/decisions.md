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

## 2026-09-13 — first HA-ViD baselines: per-view MS-TCN++ with late fusion, DINOv2 vs I3D

Protocol: frozen `splits/ha-vid` (train 17 subjects → val 6 subjects), primitive-task labels of
one hand per run, class ids from the official `mapping.txt`, one MS-TCN++ per camera view
(causal and non-causal), epoch selected on val MoF, late fusion = mean of the per-view posteriors
(`reports/havid_dev_v1_tas_{lh,rh}`, `reports/havid_dev_v2_i3d_{lh,rh}`).
Fusion vs best single causal view (F1@10, left / right hand, DINOv2): 29.1 vs 27.2 /
30.3 vs 26.6, inside overlapping intervals. DINOv2 ViT-B/14 vs the authors' I3D features,
fusion_causal F1@10 (left / right): 29.1 vs 28.7 / 30.3 vs 23.0; on the right hand the I3D fusion
also has a lower Edit (21.0 vs 33.5, intervals not overlapping); with a weak side view in the
average (causal Edit 16.2) the I3D fusion falls below its own front view alone (Edit 30.5). Decision: DINOv2 ViT-B/14 is the default frame
feature for the HA-ViD line — equal on the left hand, better on the right-hand causal fusion, and
it covers every annotated frame (the I3D features drop five at each end). Plain posterior
averaging is kept as the fusion baseline but is not assumed safe: a weighted or learned fusion is
the next fusion experiment. The causal-vs-offline gap for the DINOv2 fusion is 14.4 / 11.0 F1@10
points. All of this is a development result on val; the frozen test subjects were not read.

## 2026-09-13 — HA-ViD TAS epochs are chosen by val F1@10; fusion stays parameter-free

v1 chose each network's epoch by val MoF, which favoured `null` and stopped two causal networks at
epoch 5–10. v3 (`reports/havid_dev_v3_tas_f1sel_*`) chooses by val F1@10 and adds two
parameter-free fusion rules next to the mean (normalised geometric mean, confidence-weighted
mean), so nothing about the fusion is tuned on val. Result (fusion_causal F1@10, left / right):
v1 29.1 / 30.3 → v3 32.0 / 30.2, with Edit 28.3 / 33.5 → 35.4 / 32.4; best fusion rule on F1@10:
geometric on both hands (32.2 / 33.3), inside the intervals. Decision: the HA-ViD line selects
epochs by val F1@10 from now on; mean fusion stays the default report row because the three
rules are tied on six subjects, with the geometric mean reported next to it. Causal networks
peak at epochs 10–25 and degrade by epoch 50, so 50 epochs is enough for them.

## 2026-09-13 — the HA-ViD SOP layer learns its procedure knowledge from the train split

The public OWL graphs name steps `PT1`… without HR-SAT codes, so `sop/ha-vid/learned_*.json`
are learned from the 143 train recordings per plate (plate read off the label vocabulary):
precedence edges that hold in every co-occurring recording (min support 3), mandatory steps
(present in ≥ 90 % of a plate's recordings), duration bounds (5th–95th percentile). On val
(`reports/havid_dev_v4_sop_synthetic`): order-violation recall on ground-truth sequences 18 / 18
with 16 / 18 clean recordings flagged (the graph cannot represent the stage-2/3 instruction
variants; support 20 still flags 8 / 18); on the v3 `fusion_causal` sequences 15 / 15 with 17 / 18
flagged; omission is the one check that works on real sequences (17 / 17, 1 / 18 false alarms);
duration flags 14 / 18 clean recordings because a per-step 5–95 % window is not a recording-level
rule. Native `w`: 0 of 12 segments detected and none predicted — underpowered as ADR 0001
trigger B foresaw. Consequence: the synthetic table is the headline violation table (marked
synthetic); the native table stays a count with a Wilson interval; the next SOP work is
per-variant or majority-rule graphs and a k-of-n duration rule, and the recogniser (F1@10 ≈ 30)
is the bottleneck for the deployable numbers.

## 2026-09-13 — SOP knowledge is selected out of fold on train; the majority rule is rejected

`tune-havid-sop` runs leave-one-subject-out over the 17 train subjects for min support ∈ {3, 5,
10, 20} × min agreement ∈ {1.0, 0.95, 0.9, 0.8} and duration quantiles {[5, 95], [2, 98],
[1, 99]} %, with a selection rule fixed before val is read (coverage minus recording-level false
alarms; narrowest duration window with ≤ 5 % step false alarms, else the widest). Result
(`reports/havid_dev_v5_sop_tuned/oof.json`): every agreement threshold below 1.0 raises the
out-of-fold false alarms at every support (support 20: 32 % → 40–55 %), so edges must be
universal; support 20 with the strict rule keeps 26 of 78 edges and scores best (coverage 90 %,
recording false alarms 32 %, step false alarms 5.6 %). No duration window reaches 5 % step false
alarms ([1, 99] % gives 6.6 %); the widest is used. Decision: `sop/ha-vid/tuned/` (support 20,
agreement 1.0, [1, 99] %) is the HA-ViD SOP knowledge from now on; `sop/ha-vid/` keeps the
support-3 files that v4 reproduces. On val this halves the ground-truth order false alarms
(16 → 8 of 18; 8.5 % of steps) and trims the duration ones (14 → 11; 6.0 % of steps) with recall
unchanged at 100 %; the predicted sequences of the v3 recogniser are still flagged everywhere, so
the next SOP work is instruction-sheet step granularity and segment smoothing before the checks,
not the rule.

## 2026-09-13 — post-hoc smoothing of predicted sequences does not rescue the SOP checks

`havid-sop --min-segment-frames 11` (the train 1st-percentile step length) absorbs the v3
recogniser's blips before the checks (`reports/havid_dev_v6_sop_smoothed`). It brings the
predicted step count in line with the ground truth (354 → 189 on val) and halves the duration
step false alarms, but every val recording still misses 1–4 mandatory steps and all 18 stay
flagged. Decision: smoothing stays an option (off by default); the deployable violation numbers
wait for a better recogniser, not for more post-processing.

## 2026-09-14 — HA-ViD comparisons need three seeds

`reports/havid_dev_v7_tas_seeds_{lh,rh}` repeat the v3 protocol with seeds 1 and 2. The standard
deviation over seeds is 0.3–2.7 points on every metric, as large as every single-seed difference
reported until now; the v1 → v3 left-hand gain (+2.9 F1@10 on seed 0) shrinks to +1.4 on the
three-seed mean. Decision: from now on a HA-ViD recogniser change is reported as mean ± std over
seeds 0–2 against the v7 numbers, and a difference smaller than about 3 points is not called an
improvement. Fusion: no rule dominates (geometric best on F1@10 for both hands, confidence-weighted
best on Edit), and fusion adds only +0.5 / +1.4 F1@10 over the best single causal view; the
causal-vs-offline gap (≈ 12–13 points) is the real lever.

## 2026-09-14 — SOP steps are instruction-sheet steps, not HR-SAT primitive tasks

HR-SAT labels encode the hole index and the tool (`sshc1dh`); the instruction sheets repeat such
steps per hole without fixing the hole order or requiring the tool. `havid_sop.sheet_step` drops
both (73 labels → 51 steps). Tuned out of fold on train with the same grid and rule as v5, the
order check's recording-level false alarms fall from 32 % to 8 % (step level 5.6 % → 2.0 %) at
94 % coverage, and the [1, 99] % duration window reaches 5.0 % step false alarms. On val
(`reports/havid_dev_v8_sop_sheet`) the clean ground-truth recordings flagged fall to 4 / 18
(order), 8 / 18 (duration, 4.3 % of steps) and 1 / 18 (omission), recall unchanged at 100 %; two of
the four order-flagged recordings contain native `w` annotations. Decision: `sop/ha-vid/sheet/`
(sheet granularity, support 10, agreement 1.0, [1, 99] %) is the HA-ViD SOP knowledge; the
primitive-task level remains the recogniser's output and is mapped before the checks.

## 2026-09-14 — the review UI is a local standard-library server, not a web framework

The W5 plan listed FastAPI and uvicorn. The first slice needs three endpoints and range-served
video, so `sop_monitor/review.py` uses `http.server` and adds no dependency (the CI install stays
unchanged). The server binds to 127.0.0.1, serves only the video ids of the split it was started
with and refuses a split file named `test`; decisions are an append-only JSONL. A framework can
replace it if the UI ever needs authentication or more than one reviewer at a time.

## 2026-09-14 — steps that start on the same frame do not precede each other; the online monitor matches the offline checks

`sop_monitor/online.py` adds `OnlineSOPMonitor`, the frame-by-frame form of `check_sequence`: per
hand a run-length segmenter with a minimum-duration confirmation, two-hand step joining, order and
unknown findings when a step is confirmed, too_long as soon as a running step passes its upper
bound, too_short when it ends, omissions when the recording finishes. Making its findings equal the
offline checker's exposed a tie: `check_order` walks a label list, so of two steps that start on
the same frame the one listed first counted as earlier, decided by the end frame, which an online
monitor does not know yet. `havid_sop.check_step_order` now treats every run of equal start frames
as simultaneous (each is checked against the steps before the run). The committed v4, v5, v6 and v8
`sop_checks.json` are unchanged; the out-of-fold grids moved only in rows with agreement < 1 (at
most 0.7 points, e.g. primitive-task support 20 / agreement 0.8: 54 % → 55 % recording false
alarms) and both selections are unchanged. With `min_frames=1` online and offline findings are
equal on 500 random two-hand sequences at both granularities; the one documented exception is two
adjacent segments with the same label, which a frame stream cannot separate (149 of 5,871
annotation boundaries).

## 2026-09-14 — the streaming recogniser recomputes the causal prefix; the plate is an input

`sop_monitor/online_head.py` runs the per-view causal MS-TCN++ checkpoints of a
`train-havid-tas --checkpoint-dir` run on a growing stream. Each push recomputes the network on
the prefix seen so far and keeps the posteriors of the new time steps. Causality makes this exact
(the output at frame t depends on frames up to t), and the MS-TCN++ receptive field of over
10,000 frames exceeds every HA-ViD recording, so a shorter sliding window would change the output.
Caching per-layer convolution state would make each push constant-time; it is deferred until the
measured compute per chunk says it matters. A network trained with look-ahead L emits the label of
frame f at time f + L, and the last L frames repeat the final output, which reproduces
`advance_outputs` exactly (tested with look-ahead 0, 2, 5 and 120, chunk sizes 1 to 200, all
three fusion rules). The plate is passed in rather than recognised: an assembly station knows its
work order, and the offline runs also take the plate from the annotation. The frame-to-deviation
path is therefore checkable against the offline run of the same checkpoints, frame by frame.
