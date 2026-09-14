# docs/assets

Animated figures rendered by `make figures` (Manim Community, `figures/*.py`, GIFs built by
`figures/render.py`). Each GIF is 800 px wide at 12 fps on the dark data-viz surface; the alarm
colours are the reserved status colours and always carry a glyph and a label.

| file | scene | what it shows | source of the numbers |
|---|---|---|---|
| `pipeline.gif` | `figures/pipeline.py::PipelineScene` | the components in order: three cameras → frozen DINOv2 → one causal MS-TCN++ per view and hand → late fusion → online SOP monitor → review queue; the two alarms that pop are illustrative | mechanism only |
| `sop_timeline.gif` | `figures/timeline.py::TimelineScene` | one val recording (S18A06I01, gear plate) replayed frame by frame: ground-truth and predicted step bars per hand, every alarm at the frame it was detected | `reports/havid_dev_v8_sop_sheet/steps_val.csv`, `reports/havid_dev_v10_sop_online/deviations_val.csv` (stream `la0_s0`, confirmation 1 frame) |
| `causal_padding.gif` | `figures/concepts.py::CausalPaddingScene` | why a left-padded dilated convolution only sees frames ≤ t, and what symmetric padding sees instead | `reports/havid_dev_v7_tas_seeds_*` for the quoted F1@10 gap |
| `look_ahead.gif` | `figures/concepts.py::LookAheadScene` | look-ahead as a fixed output delay, and what 1 / 3 / 6 s buy on val | `reports/havid_dev_v9_step_recall` |
| `ring_buffer.gif` | `figures/concepts.py::RingBufferScene` | WAIT vs DROP_OLDEST when the consumer is slower than the producer | mechanism only (`src/sop_monitor/stream/ring_buffer.py`) |

`sop_timeline.gif` is drawn from HA-ViD annotations and model outputs on them; like the tables it
comes from, it is a derivative of CC BY-NC 4.0 data, used here for non-commercial research with
attribution to HA-ViD (Zheng, Lee, Lu, NeurIPS 2023 Datasets and Benchmarks). No video frame is
shown.
