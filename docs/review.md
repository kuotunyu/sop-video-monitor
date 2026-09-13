# Deviation review (W5, first slice)

The SOP checks of a committed `havid-sop` run become a queue of deviations that a person reviews
in a browser against the three camera views. Everything runs locally with the standard library;
nothing is uploaded.

## Workflow

```bash
export PYTHONUTF8=1
# 1. queue: one item per finding of the run's recogniser output (or --source gt for the annotations)
uv run sop-monitor build-review-queue --run reports/havid_dev_v8_sop_sheet --source pred --out artifacts/review/v8_pred_queue.jsonl
# 2. review: open http://127.0.0.1:8765/ ; keys a = accept, r = reject, s = skip, j/k = next/previous, space = replay
uv run sop-monitor review-ui --queue artifacts/review/v8_pred_queue.jsonl --decisions artifacts/review/v8_pred_decisions.jsonl --reviewer <name>
# 3. summary: accepted / rejected per kind and the reviewed precision
uv run sop-monitor review-summary --queue artifacts/review/v8_pred_queue.jsonl --decisions artifacts/review/v8_pred_decisions.jsonl
```

`make review-queue` and `make review-ui` wrap steps 1 and 2 for the v8 run.

## What a queue item is

| field | meaning |
|---|---|
| `kind` | `omission` (mandatory step never observed), `order` (step before its learned predecessors), `duration` (step outside its learned window), `unknown` (step the plate's train recordings never showed) — reviewed in that order |
| `step`, `step_description` | the HR-SAT or sheet-level label and its reading from paper Figure 9 (`sshc` → "screw hex screw → cylinder plate hole") |
| `start_s`, `end_s` | the step's window at 15 fps; `null` for omissions (the whole recording) |
| `evidence` | the rule that fired and its numbers: missing predecessors, measured duration and window |
| `videos` | side / front / top video ids of the recording; the page plays them in sync around the window (± 2 s) |
| `id` | SHA-1 of run, source, recording, kind, step and window — stable across rebuilds of the same run |
| `second_opinion` | reserved for the asynchronous VLM verifier (not implemented) |

## Guarantees and limits

- The server binds to `127.0.0.1` only and serves only the video ids of the split file it was
  started with (default `splits/ha-vid/val.csv`); it refuses to start on a split file named
  `test`, and any other id returns 404.
- Decisions are appended to a JSONL file (`id`, `decision`, `note`, `reviewer`, UTC time); the last
  decision per item wins, earlier ones stay in the file. Queue and decision files stay under
  `artifacts/` (ignored by git): they name dataset videos and carry a reviewer name.
- A reviewed precision is a statement about one reviewer's judgement of one run's queue, not a
  detection metric; the synthetic tables in `reports/havid_dev_v*_sop_*` remain the headline.
- Deviations are suggestions for human review. Neither the queue nor a decision is a safety or
  compliance guarantee.
