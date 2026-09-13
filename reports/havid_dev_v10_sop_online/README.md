# havid_dev_v10_sop_online — the SOP checks run frame by frame (development result)

- Tier: **development result** on `splits/ha-vid/val.csv` (6 subjects, 18 recordings); the frozen
  test subjects were not read.
- Date: 2026-09-14. Author: kuotunyu. Machine: Windows 11, CPU only (the whole grid takes 4 s).
- Question: the v8 SOP checks ran on finished recordings. Run online, frame by frame, when does each
  deviation arrive, what does a minimum-duration confirmation (`min_frames`) change, and how does
  the recogniser's output delay add up?

## 1. Method

`sop-monitor havid-online` pushes each recording's two per-hand label streams, one frame at a time,
into `OnlineSOPMonitor` (`src/sop_monitor/online.py`) with the instruction-sheet knowledge of v8
(`sop/ha-vid/sheet`). Order and unknown findings fire when a step is confirmed, too_long as soon as a
running step passes its upper bound, too_short when a step ends, omissions at the end of the
recording. Streams: ground truth, and causal mean fusion of the v3 seeds (look-ahead 0) and of the v9
runs (look-ahead 15, 45, 90), three seeds each. The confirmation length `min_frames` m ∈ {1, 4, 8,
15, 30} absorbs label changes shorter than m frames. Alarm times include the stream's output delay.

Files: `deviations_val.csv` (every deviation with the frame it was detected at), `recordings_val.csv`
(length, plate, native `w`), `online.json` (summary), `tables.md` (per stream and seed-grouped);
`reproduce-lite` recomputes the summary from the two CSVs.

**Consistency with the offline checker.** At m = 1 the online findings equal `check_sequence` on 16
of 18 ground-truth recordings and on 18 of 18 recordings of every predicted stream. Both ground-truth
differences come from one step annotated as two adjacent segments with the same label (`ipsft` in
S08A04I01, `lck` in S12A06I01), which a frame stream cannot separate; the offline checker counts them
twice.

## 2. Results (`tables.md` is authoritative)

Recordings flagged (any of order, omission, duration) as with `w` · without `w`, alarms per recording,
median time of the first alarm into the recording; predicted streams are mean ± std over seeds 0–2.

| stream | delay | m = 1 | m = 8 | m = 30 | alarms / rec at m = 1 → 8 → 30 | first alarm s at m = 8 |
|---|---|---|---|---|---|---|
| ground truth | — | 3/4 · 6/14 | 3/4 · 7/14 | 4/4 · 12/14 | 1.9 → 1.9 → 3.3 | 40.4 |
| causal L = 0 | 0 s | 4/4 · 14/14 | 4/4 · 14/14 | 4/4 · 14/14 | 18.9 → 11.8 → 6.9 | 18.0 ± 3.1 |
| causal L = 15 | 1 s | 4/4 · 14/14 | 4/4 · 14/14 | 4/4 · 13.7/14 | 21.5 → 11.6 → 6.2 | 15.1 ± 1.4 |
| causal L = 45 | 3 s | 4/4 · 13.7/14 | 4/4 · 13.7/14 | 4/4 · 13.7/14 | 13.1 → 8.6 → 6.3 | 19.3 ± 1.3 |
| causal L = 90 | 6 s | 4/4 · 14/14 | 4/4 · 14/14 | 4/4 · 14/14 | 12.8 → 9.1 → 6.8 | 24.6 ± 1.4 |

Reading:

- **On predicted streams the monitor still flags every recording.** At every look-ahead and every m,
  13.7–14 of the 14 clean recordings raise an alarm. Online or offline, the recogniser decides the
  outcome: the v8 conclusion holds frame by frame.
- **Confirmation trades alarms for ground-truth fidelity.** m = 8 frames (0.5 s) cuts the alarms of
  the L = 0 stream from 18.9 to 11.8 per recording and costs one extra flagged clean ground-truth
  recording; m = 30 (2 s) cuts them to 6.9 but absorbs real short steps, so 12 of 14 clean
  ground-truth recordings get flagged. By the pre-registered rule, m\* = 8 at L\* = 90
  (`docs/decisions.md`).
- **Look-ahead helps the SOP layer less than the recogniser metrics suggest.** L = 45 and L = 90
  raise the fewest alarms at m = 1 (13.1 and 12.8, against 18.9 at L = 0) but still flag 13.7 and 14
  of 14 clean recordings.
- **Alarms arrive during the task.** With m = 8, the first alarm on a predicted stream comes 15–25 s
  into recordings that last 31–113 s, and alarms other than omissions arrive a median of 24–27 s
  before the recording ends; the ground-truth stream's first alarm comes at 40 s. An end-of-recording check would report all of them later.
- **Order alarms fire (m − 1) / 15 s + L / 15 s after the step starts** by construction; there is no
  other latency in the monitor itself (the whole grid, 13 streams × 5 values of m × 18 recordings,
  replays in 4 s; a rerun gives identical deviations).

## 3. Reproduce

```bash
export PYTHONUTF8=1
make havid-online-v10
uv run sop-monitor reproduce-lite
```

## 4. Not done here

- Real frames: this replays label streams. The frame → features → recogniser → monitor path is
  `havid-online-head` (`src/sop_monitor/online_head.py`), measured separately.
- Reviewed alarms: no person has judged these deviations.
- Native `w` detection: 4 recordings with `w`, all flagged in every predicted stream, but so is every
  clean recording; this is not a detection result.
- No number on the frozen test subjects.
