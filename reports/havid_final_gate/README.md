# havid_final_gate — reproducibility gate for the final networks

- Tier: **check**, not a result. Protocol: `docs/havid_test_protocol.md` §3.
- Date: 2026-09-14. Author: kuotunyu. Machine: Windows 11, RTX 4090.

The committed dev runs saved no weights, so the networks used on the frozen test split are retrained
with `train-havid-tas --checkpoint-dir` (`make havid-final-L0`, `make havid-final-lstar LSTAR=…`).
The retrained runs and their weights live under `artifacts/` (not tracked). `sop-monitor
compare-runs` compares each retrained run's `predictions_val.csv` with the committed dev run of the
same setting, hand and seed, column by column (`gate_*.json`: frame agreement and metric deltas).

| file | pairs | result |
|---|---|---|
| `gate_L0.json` | causal L = 0 with offline twins vs `havid_dev_v3_tas_f1sel_{lh,rh}[_s1,_s2]` | 6 / 6 identical on all 13 prediction columns; same selected epoch for every network |
| `gate_L90.json` | causal L = 90 (L*, pre-registered) vs `havid_dev_v9_tas_la90_{lh,rh}_s{0,1,2}` | 6 / 6 identical on all 7 prediction columns |
| `gate_L45.json` | causal L = 45 (declared amendment, not pre-registered) vs `havid_dev_v9_tas_la45_{lh,rh}_s{0,1,2}` | 6 / 6 identical on all 7 prediction columns |

Identical predictions mean the test networks are the dev networks: `cudnn.deterministic` is set and
parallel training on the same GPU did not change a single frame.

The paths in the JSON files point to `artifacts/final_runs/`, which exists only on the training
machine; the committed half of every pair is in `reports/`.
