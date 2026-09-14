# What this repository does not show

Written against the state of 2026-09-14. Each line names the evidence that is missing, not a plan.

## Generalisation

- **One test evaluation, seven subjects.** The HA-ViD test subjects were evaluated once
  (`reports/havid_test_v1.md`), 41 recordings; IndustReal's test split was never read. Every other
  number is on validation subjects that also drove model and setting selection. The L = 45 test row
  was added after the val results and is not a pre-registered result.
- **Six validation subjects.** HA-ViD val has 6 subjects and 18 recordings; bootstrap intervals are
  wide and seed noise (0.3–2.7 points) is as large as most differences between variants.
- **One laboratory, one product.** HA-ViD is one assembly box on one workbench with three fixed
  cameras. Nothing here says how the recogniser or the SOP checks behave on another product,
  station, lighting or camera placement, let alone on a factory line.
- **No unseen-view or missing-view result.** Every model sees the same three camera positions in
  training and evaluation.

## Recognition

- **The recogniser is weak.** Causal fusion F1@10 is 27–29 on the test subjects without delay and
  33–34 with a 3 s delay; its sequences miss and fragment steps, and every predicted test recording
  is flagged by the SOP checks. Any statement about deployable SOP monitoring waits for a better
  recogniser.
- **No atomic-action (219-class) results, no ASFormer, no learned fusion, no fine-tuned or video
  (clip) features.** The three parameter-free fusion rules and the single-view controls are in
  `reports/havid_dev_v1`–`v7`.
- **No HA-ViD online PSR-style metric.** Completion of a HA-ViD step is not defined the way
  IndustReal's procedure states are; the online replay reports alarms and their timing instead.
- **Real-time on one machine only, and only for replayed files.** On one RTX 4090, 24 paced 15 fps
  streams are decoded and embedded without falling behind (`reports/stream_bench_v1_dinov2`), and
  one station's three cameras run from mp4 to deviations at 0.32 × real time
  (`reports/havid_dev_v11_online_head_features`). Nothing is measured with live cameras, RTSP
  (mediamtx is not installed), a cross-process shared-memory ring buffer, network jitter, other
  GPUs, or several stations' heads at once. The streamed heads recompute the
  prefix, so their cost grows with the recording length.

## SOP checks and errors

- **Synthetic violations are not real violations.** The 100 % recall figures come from perturbing
  sequences (moving a step, deleting a mandatory step, stretching a step); real deviations are
  subtler and are not sampled from these three operators.
- **Native error detection is unmeasured in any useful sense.** HA-ViD val has 12 `wrong` segments
  (4 recordings); the recogniser detects none. Two of the four order-flagged ground-truth recordings
  contain `wrong` annotations — an observation on four positives, not a detection rate.
- **Procedure knowledge is learned, not specified.** The precedence graphs, mandatory steps and
  duration windows are statistics of 17 training subjects, not the manufacturer's SOP. An order
  finding means "unlike every training recording", not "forbidden".
- **No human-reviewed precision yet.** The review queue and UI exist (`docs/review.md`); no reviewer
  has judged a queue, so there is no reviewed precision of the deviations.
- **No VLM second opinion.** The queue reserves a field for it; nothing is implemented or evaluated.

## Publication

- **No published weights or features.** The code is public on GitHub; trained weights and the
  feature caches are non-commercial HA-ViD derivatives and are not uploaded anywhere (no Hugging
  Face artefact).
- **No static result figures.** Results are tables; the only figures are the animated mechanism
  figures under `docs/assets/` and the Mermaid structure diagrams.

## Non-goals (design spec §2)

- No object-detection main line and no bounding boxes as the primary output; HA-ViD's CVAT boxes
  are only an aid in the review UI.
- No real cameras, no DeepStream, no edge deployment.
- No multi-node or distributed training; every number comes from one RTX 4090.
- No chase of the HA-ViD leaderboard: the official test bundle does not state its subject rule and is
  not comparable with this project's subject-wise split.

## Scope and licences

- **No commercial use** is possible with the HA-ViD derivatives (CC BY-NC 4.0).
- **No safety or compliance guarantee** of any kind; deviations are suggestions for human review.
- **IndustReal results do not transfer** to HA-ViD tables; they validate the metric implementation
  and the online decoding approach on a different dataset.
