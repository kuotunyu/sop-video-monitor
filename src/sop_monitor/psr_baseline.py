"""Online procedure-step baselines on IndustReal val (spec 4.3 sanity run, development tier).

A state head ("component k is correctly installed", 11 sigmoid outputs) is fitted on frozen
DINOv2 features, decoded causally into step completions and scored per video with
:func:`sop_monitor.metrics.online.psr_performance` against ``PSR_labels.csv``. Two protocols:
leave-one-participant-out over the recordings in ``psr_dir`` (used when only the val labels were
on disk) and train-split -> eval-split (36 train recordings -> 16 val recordings). Decoder settings
are chosen either in-sample on the training recordings or, with ``selection="nested"``, on
out-of-fold predictions of those recordings (grouped 4-fold by participant); never on the evaluated
participants. Every completion the decoders emit — and every ground-truth completion — is written
to ``completions_val.csv`` (one ``run`` column per head x decoder, ``_s<seed>`` suffixed when
several seeds run) so the metrics can be recomputed without features.

Heads: ``linear`` (multi-label logistic regression per frame) and ``mstcn`` (causal MS-TCN++ with
sigmoid outputs, :mod:`sop_monitor.mstcn`). Decoders: ``plain`` (causal EMA + hysteresis, as in
``industreal_dev_v3_psr``) and ``prior_dwell`` (the same plus a minimum dwell time before an event
is emitted and a *procedure prior learned from the fold's training labels*: which components ever
change in recordings of that type — ``assy`` or ``main`` — and whether each starts installed).
Incorrect installs (``3k + 1``) are never predicted; the reference scores ``PSR_labels.csv``
(correct steps only) in exactly this way.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from sop_monitor.industreal import participant_of
from sop_monitor.industreal_psr import (
    INCORRECT,
    INSTALL,
    N_COMPONENTS,
    PROCEDURE_INFO,
    REMOVE,
    audit_recording,
    frame_offset,
    load_procedure_info,
    load_psr_labels,
    load_psr_raw,
    shift_completions,
)
from sop_monitor.metrics.offline import subject_bootstrap
from sop_monitor.metrics.online import Completion, psr_performance

HEADS: tuple[str, ...] = ("linear", "mstcn")
DECODERS: tuple[str, ...] = ("plain", "prior_dwell")
SELECTIONS: tuple[str, ...] = ("in_sample", "nested")

# ---------------------------------------------------------------------------------------------
# Targets and decoding (numpy only)


def frame_states(raw: Sequence[tuple[int, Sequence[int]]], n_frames: int) -> np.ndarray:
    """Forward-fill raw state rows to every frame; ``int8[n_frames, N_COMPONENTS]`` in {-1, 0, 1}."""
    states = np.zeros((n_frames, N_COMPONENTS), dtype=np.int8)
    rows = sorted(raw, key=lambda r: r[0])
    for (frame, values), (next_frame, _) in zip(rows, [*rows[1:], (n_frames, ())], strict=False):
        if frame >= n_frames:
            break
        states[frame : min(next_frame, n_frames)] = np.asarray(values, dtype=np.int8)
    return states


@dataclass(frozen=True)
class DecoderConfig:
    ema: float = 0.9  # weight of the previous average; 0 = no smoothing
    theta_on: float = 0.7
    theta_off: float = 0.3
    min_dwell: int = 0  # extra consecutive frames beyond the crossing before an event is emitted


@dataclass(frozen=True)
class ProcedurePrior:
    """What a fold's training labels say about one recording type (``assy`` / ``main``)."""

    active: np.ndarray  # bool[N_COMPONENTS]: the component ever changes state
    initial_installed: np.ndarray  # bool[N_COMPONENTS]: majority initial state


def ema_filter(probs: np.ndarray, ema: float) -> np.ndarray:
    """Causal exponential moving average along axis 0 (``ema`` = weight of the past)."""
    if not 0.0 <= ema < 1.0:
        raise ValueError("ema must be in [0, 1)")
    if ema == 0.0:
        return probs.astype(np.float64)
    out = np.empty(probs.shape, dtype=np.float64)
    running = np.zeros(probs.shape[1], dtype=np.float64)
    for t in range(probs.shape[0]):
        running = ema * running + (1.0 - ema) * probs[t]
        out[t] = running
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` of the ``True`` runs of a boolean vector (``end`` exclusive)."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def decode_from_average(
    average: np.ndarray,
    frames: np.ndarray,
    config: DecoderConfig,
    prior: ProcedurePrior | None = None,
) -> list[Completion]:
    """Hysteresis with optional dwell and prior over an already-smoothed ``average[t, k]``.

    An install (``3k``) is emitted ``min_dwell`` frames after the average first exceeds
    ``theta_on`` provided it stays above for that long; a removal (``3k + 2``) likewise below
    ``theta_off``. A component starts in ``prior.initial_installed`` (else not installed) and
    components with ``prior.active == False`` never emit. Output is in emission order.
    """
    if not config.theta_off < config.theta_on:
        raise ValueError("theta_off must be below theta_on")
    if config.min_dwell < 0:
        raise ValueError("min_dwell must be >= 0")
    events: list[tuple[int, int, int]] = []
    for k in range(average.shape[1]):
        if prior is not None and not prior.active[k]:
            continue
        installed = bool(prior.initial_installed[k]) if prior is not None else False
        above = [(s, e, True) for s, e in _runs(average[:, k] > config.theta_on)]
        below = [(s, e, False) for s, e in _runs(average[:, k] < config.theta_off)]
        for start, end, is_install in sorted(above + below):
            if end - start <= config.min_dwell:
                continue
            if is_install and not installed:
                installed = True
                events.append((start + config.min_dwell, k, 3 * k + INSTALL))
            elif not is_install and installed:
                installed = False
                events.append((start + config.min_dwell, k, 3 * k + REMOVE))
    return [Completion(frame=int(frames[t]), step=step) for t, _, step in sorted(events)]


def decode_completions(
    probs: np.ndarray,
    frames: np.ndarray,
    config: DecoderConfig,
    prior: ProcedurePrior | None = None,
) -> list[Completion]:
    """Causal EMA + hysteresis (+ dwell, + prior) over ``probs[t, k]``; completions in emission order."""
    return decode_from_average(ema_filter(probs, config.ema), frames, config, prior)


# ---------------------------------------------------------------------------------------------
# Completion tables and scoring (no features needed)


@dataclass(frozen=True)
class CompletionRow:
    video_id: str
    participant: str
    source: str  # "gt" | "pred"
    frame: int
    step: int
    fold: str = ""
    run: str = (
        ""  # head x decoder name for predictions; empty for ground truth or single-run tables
    )


COMPLETION_COLUMNS = ["video_id", "participant", "source", "frame", "step", "fold", "run"]


def write_completions(path: Path, rows: Sequence[CompletionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(COMPLETION_COLUMNS) + "\n")
        for r in rows:
            handle.write(
                f"{r.video_id},{r.participant},{r.source},{r.frame},{r.step},{r.fold},{r.run}\n"
            )


def read_completions(path: Path) -> list[CompletionRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header not in (COMPLETION_COLUMNS, COMPLETION_COLUMNS[:-1]):
            raise ValueError(f"unexpected completions header: {header}")
        return [
            CompletionRow(r[0], r[1], r[2], int(r[3]), int(r[4]), r[5], r[6] if len(r) > 6 else "")
            for r in reader
            if r
        ]


def _mean_of(items: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(m[key]) for m in items if m[key] is not None]  # type: ignore[arg-type]
    return float(np.mean(values)) if values else float("nan")


def _score_run(
    gts: Mapping[str, list[Completion]],
    preds: Mapping[str, list[Completion]],
    participants: Mapping[str, str],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    per_video = {
        video: psr_performance(
            sorted(gts[video], key=lambda c: c.frame),
            sorted(preds.get(video, []), key=lambda c: c.frame),
        )
        for video in sorted(gts)
    }
    per_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for video, metrics in per_video.items():
        per_subject[participants[video]].append(metrics)
    summary: dict[str, dict[str, object]] = {}
    for key in ("pos", "f1", "mean_delay_frames"):
        boot = subject_bootstrap(
            per_subject,  # type: ignore[arg-type]
            lambda items, key=key: _mean_of(items, key),  # type: ignore[arg-type]
            n_boot=n_boot,
            seed=seed,
        )
        summary[key] = {"mean_over_videos": boot.point, "ci95": [boot.lower, boot.upper]}
    totals = {
        k: int(sum(int(m[k]) for m in per_video.values()))  # type: ignore[arg-type]
        for k in ("system_tp", "system_fp", "system_fn", "n_gt", "n_pred")
    }
    return {"summary": summary, "totals": totals, "per_video": per_video}


def score_completions(
    rows: Sequence[CompletionRow],
    n_boot: int = 2000,
    seed: int = 0,
    run_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Per-video PSR metrics, their mean over videos (the reference's aggregation), participant CIs.

    With a single prediction run the layout is flat (``summary`` / ``totals`` / ``per_video``);
    with several runs they sit under ``runs[<name>]``. ``gt_sanity`` is shared.
    """
    gts: dict[str, list[Completion]] = defaultdict(list)
    preds: dict[str, dict[str, list[Completion]]] = defaultdict(lambda: defaultdict(list))
    for name in run_names or ():  # a run that emitted nothing must still be scored (all FN)
        preds[name] = defaultdict(list)
    participants: dict[str, str] = {}
    for r in rows:
        participants[r.video_id] = r.participant
        if r.source == "gt":
            gts[r.video_id].append(Completion(r.frame, r.step))
        else:
            preds[r.run or "pred"][r.video_id].append(Completion(r.frame, r.step))
    if not preds:
        preds["pred"] = defaultdict(list)
    out: dict[str, object] = {
        "n_videos": len(gts),
        "n_participants": len(set(participants.values())),
        "aggregation": "unweighted mean of per-video metrics (as psr_baseline.py); delay in frames at 10 fps",
        "bootstrap": {"unit": "participant", "n_boot": n_boot, "seed": seed, "alpha": 0.05},
        "gt_sanity": gt_sanity({v: sorted(gts[v], key=lambda c: c.frame) for v in sorted(gts)}),
    }
    scored = {run: _score_run(gts, preds[run], participants, n_boot, seed) for run in sorted(preds)}
    if len(scored) == 1:
        ((run, single),) = scored.items()
        out.update(single)
        out["run"] = run
    else:
        out["runs"] = scored
        seeds = seed_summary(scored)
        if seeds:
            out["seed_summary"] = seeds
    return out


def seed_summary(scored: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Mean and standard deviation over seeds of runs named ``<base>_s<seed>``."""
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for run, result in scored.items():
        base, sep, suffix = run.rpartition("_s")
        if sep and suffix.isdigit():
            groups[base].append(result)
    out: dict[str, dict[str, object]] = {}
    for base, results in sorted(groups.items()):
        if len(results) < 2:
            continue
        entry: dict[str, object] = {"n_seeds": len(results)}
        for key in ("pos", "f1", "mean_delay_frames"):
            values = [float(r["summary"][key]["mean_over_videos"]) for r in results]  # type: ignore[index]
            entry[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for key in ("system_tp", "system_fp", "system_fn"):
            entry[key] = float(np.mean([float(r["totals"][key]) for r in results]))  # type: ignore[index]
        out[base] = entry
    return out


def gt_sanity(gts: Mapping[str, Sequence[Completion]]) -> dict[str, dict[str, float]]:
    """Metric behaviour on the real label structures under controlled perturbations of the GT.

    ``identity`` must give POS 1, F1 ≈ 1 and delay 0; ``shift_plus_30_frames`` must keep POS/F1
    and report a 30-frame delay; ``drop_last`` costs one FN per video; ``swap_first_two``
    exchanges the times of the first two completions (at most one transposition, POS
    ``1 - 1/n``; nothing changes when the two share a frame).
    """

    def swap(gt: Sequence[Completion]) -> list[Completion]:
        if len(gt) < 2:
            return list(gt)
        return [Completion(gt[1].frame, gt[0].step), Completion(gt[0].frame, gt[1].step), *gt[2:]]

    cases = {
        "identity": lambda gt: list(gt),
        "shift_plus_30_frames": lambda gt: [Completion(c.frame + 30, c.step) for c in gt],
        "drop_last": lambda gt: list(gt[:-1]),
        "swap_first_two": swap,
    }
    out: dict[str, dict[str, float]] = {}
    for name, perturb in cases.items():
        # Perturbed completions are re-sorted by frame, as the decoder's emissions would be.
        results = [
            psr_performance(gt, sorted(perturb(gt), key=lambda c: c.frame))
            for gt in gts.values()
            if gt
        ]
        delays = [
            float(r["mean_delay_frames"]) for r in results if r["mean_delay_frames"] is not None
        ]  # type: ignore[arg-type]
        out[name] = {
            "pos": float(np.mean([float(r["pos"]) for r in results])),  # type: ignore[arg-type]
            "f1": float(np.mean([float(r["f1"]) for r in results])),  # type: ignore[arg-type]
            "mean_delay_frames": float(np.mean(delays)) if delays else float("nan"),
            "system_fp": float(sum(int(r["system_fp"]) for r in results)),  # type: ignore[arg-type]
            "system_fn": float(sum(int(r["system_fn"]) for r in results)),  # type: ignore[arg-type]
        }
    return out


# ---------------------------------------------------------------------------------------------
# Data, heads, priors


@dataclass(frozen=True)
class PSRSpec:
    weight_decay: float = 1e-3
    epochs: int = 300
    lr: float = 5e-3
    mstcn_epochs: int = 40
    mstcn_lr: float = 5e-4
    # Nested epoch selection (selection="nested" only): the inner-fold heads train up to
    # max(grid) epochs, the held-out BCE is measured at every epoch in the grid, and the epoch
    # with the lowest mean held-out BCE is used for the out-of-fold probabilities and the final
    # fit. Empty = fixed ``mstcn_epochs``.
    mstcn_epoch_grid: tuple[int, ...] = ()
    seed: int = 0
    n_boot: int = 2000
    # Widened once (0.98 / 0.99 and 0.95 / 0.1 added) after the first leave-one-out run showed
    # FP-dominated errors; selection still happens on each fold's training videos only.
    emas: tuple[float, ...] = (0.0, 0.8, 0.9, 0.95, 0.98, 0.99)
    theta_ons: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 0.95)
    theta_offs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    min_dwells: tuple[int, ...] = (0, 10, 30, 60)  # frames at 10 fps; used by prior_dwell only
    selection_metric: str = "f1"
    # Latency budgets in seconds: when given, every decoder is chosen once per budget as the
    # best-F1 setting whose mean delay on the selection videos stays within it (runs are
    # suffixed ``_cap<seconds>``); when empty, plain best F1.
    delay_caps_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class PSRVideo:
    video_id: str
    participant: str
    kind: str  # "assy" | "main"
    frames: np.ndarray
    features: np.ndarray
    targets: np.ndarray  # float32[n_sampled, N_COMPONENTS], 1 where correctly installed
    initial_state: np.ndarray  # bool[N_COMPONENTS], installed at frame 0
    gt: list[Completion]  # PSR_labels.csv (correct steps only)
    has_errors: bool
    n_frames: int


def recording_kind(video_id: str) -> str:
    return "main" if "_main_" in video_id else "assy"


def load_psr_videos(features_dir: Path, psr_dir: Path) -> list[PSRVideo]:
    videos: list[PSRVideo] = []
    for rec_dir in sorted(p for p in psr_dir.iterdir() if p.is_dir()):
        cache = features_dir / f"{rec_dir.name}.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"no cached features for {rec_dir.name}: {cache}")
        with np.load(cache) as payload:
            frames = payload["frames"].astype(np.int64)
            features = payload["features"]
            n_frames = int(payload["n_frames"])
        offset = frame_offset(rec_dir, n_frames)  # JPEG name -> video frame (0 for all but one)
        raw = [
            (max(0, frame - offset), values)
            for frame, values in load_psr_raw(rec_dir / "PSR_labels_raw.csv")
        ]
        states = frame_states(raw, n_frames)
        with_errors = load_psr_labels(rec_dir / "PSR_labels_with_errors.csv")
        videos.append(
            PSRVideo(
                video_id=rec_dir.name,
                participant=participant_of(rec_dir.name),
                kind=recording_kind(rec_dir.name),
                frames=frames,
                features=features,
                targets=(states[frames] == 1).astype(np.float32),
                initial_state=states[0] == 1,
                gt=shift_completions(load_psr_labels(rec_dir / "PSR_labels.csv"), offset),
                has_errors=any(c.step % 3 == INCORRECT for c in with_errors),
                n_frames=n_frames,
            )
        )
    return videos


def learn_prior(train_videos: Sequence[PSRVideo], kind: str) -> ProcedurePrior:
    """Which components ever change, and the majority initial state, in training recordings of ``kind``."""
    same = [v for v in train_videos if v.kind == kind] or list(train_videos)
    active = np.zeros(N_COMPONENTS, dtype=bool)
    for v in same:
        for c in v.gt:
            active[c.step // 3] = True
    initial = np.mean([v.initial_state for v in same], axis=0) > 0.5
    return ProcedurePrior(active=active, initial_installed=initial)


@dataclass(frozen=True)
class StateHead:
    """Standardisation + one affine layer with sigmoid outputs; inference is plain numpy."""

    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray  # (N_COMPONENTS, dim)
    bias: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = ((x.astype(np.float32) - self.mean) / self.std) @ self.weight.T + self.bias
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def train_state_head(x: np.ndarray, y: np.ndarray, spec: PSRSpec, device: str) -> StateHead:
    """Full-batch multi-label logistic regression (BCE) with Adam, seeded."""
    import torch

    mean = x.astype(np.float32).mean(axis=0)
    std = x.astype(np.float32).std(axis=0) + 1e-6
    torch.manual_seed(spec.seed)
    xt = torch.from_numpy((x.astype(np.float32) - mean) / std).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    layer = torch.nn.Linear(x.shape[1], y.shape[1]).to(device)
    optimiser = torch.optim.Adam(layer.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    for _ in range(spec.epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(layer(xt), yt)
        loss.backward()
        optimiser.step()
    return StateHead(
        mean=mean,
        std=std,
        weight=layer.weight.detach().cpu().numpy().astype(np.float32),
        bias=layer.bias.detach().cpu().numpy().astype(np.float32),
    )


class MSTCNStateHead:
    """Causal MS-TCN++ with sigmoid outputs over the 11 component states (torch inference)."""

    def __init__(self, model: object, mean: np.ndarray, std: np.ndarray, device: str) -> None:
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()  # type: ignore[attr-defined]
        xt = torch.from_numpy(((x.astype(np.float32) - self.mean) / self.std).astype(np.float32))
        with torch.inference_mode():
            logits = self.model(xt.T.unsqueeze(0).to(self.device))[-1]  # type: ignore[operator]
            return torch.sigmoid(logits)[0].T.cpu().numpy().astype(np.float32)


def train_state_mstcn(
    train_videos: Sequence[PSRVideo],
    spec: PSRSpec,
    device: str,
    eval_videos: Sequence[PSRVideo] = (),
    eval_epochs: Sequence[int] = (),
) -> tuple[MSTCNStateHead, dict[str, object], dict[int, list[np.ndarray]]]:
    """Causal MS-TCN++ trained with per-stage BCE + truncated-MSE smoothing for ``mstcn_epochs``.

    With ``eval_videos`` and ``eval_epochs`` the held-out BCE (last stage) is logged and the
    held-out probabilities are snapshotted at those epochs, which is what nested epoch selection
    consumes. Returns ``(head, log, snapshots)``; ``snapshots[epoch]`` is one array per eval video.
    """
    import torch
    from torch.nn import functional as F

    from sop_monitor.mstcn import MSTCNSpec, build_model

    torch.manual_seed(spec.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x_all = np.concatenate([v.features for v in train_videos]).astype(np.float32)
    mean, std = x_all.mean(axis=0), x_all.std(axis=0) + 1e-6
    mstcn_spec = MSTCNSpec(lr=spec.mstcn_lr, epochs=spec.mstcn_epochs, seed=spec.seed)
    model = build_model(x_all.shape[1], N_COMPONENTS, mstcn_spec, causal=True, activation="sigmoid")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.mstcn_lr)

    def to_tensor(v: PSRVideo) -> torch.Tensor:
        return (
            torch.from_numpy(((v.features.astype(np.float32) - mean) / std).astype(np.float32))
            .T.unsqueeze(0)
            .to(device)
        )

    tensors = [
        (to_tensor(v), torch.from_numpy(v.targets.T.astype(np.float32)).unsqueeze(0).to(device))
        for v in train_videos
    ]
    held = [
        (to_tensor(v), torch.from_numpy(v.targets.astype(np.float32)).to(device))
        for v in eval_videos
    ]
    order = np.random.default_rng(spec.seed)
    losses: list[float] = []
    heldout_bce: dict[int, float] = {}
    snapshots: dict[int, list[np.ndarray]] = {}
    started = time.perf_counter()
    for epoch in range(1, spec.mstcn_epochs + 1):
        model.train()
        total = 0.0
        for i in order.permutation(len(tensors)):
            x, y = tensors[i]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for logits in model(x):
                loss = loss + F.binary_cross_entropy_with_logits(logits, y)
                log_probs = F.logsigmoid(logits)
                smooth = torch.clamp(
                    F.mse_loss(
                        log_probs[:, :, 1:], log_probs.detach()[:, :, :-1], reduction="none"
                    ),
                    max=mstcn_spec.smoothing_clamp,
                )
                loss = loss + mstcn_spec.smoothing_weight * smooth.mean()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
        losses.append(total / len(tensors))
        if held and epoch in eval_epochs:
            model.eval()
            probs_at: list[np.ndarray] = []
            bces: list[float] = []
            with torch.inference_mode():
                for x, targets in held:
                    logits = model(x)[-1][0].T  # (T, C)
                    bces.append(float(F.binary_cross_entropy_with_logits(logits, targets).item()))
                    probs_at.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
            heldout_bce[epoch] = float(np.mean(bces))
            snapshots[epoch] = probs_at
    log: dict[str, object] = {
        "epochs": spec.mstcn_epochs,
        "final_train_loss": losses[-1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.perf_counter() - started,
    }
    if heldout_bce:
        log["heldout_bce"] = {str(e): b for e, b in sorted(heldout_bce.items())}
    return MSTCNStateHead(model, mean, std, device), log, snapshots


# ---------------------------------------------------------------------------------------------
# Decoder selection and the leave-one-participant-out run


@dataclass(frozen=True)
class GridPoint:
    """One decoder setting scored on the selection videos (means over videos)."""

    config: DecoderConfig
    f1: float
    pos: float
    delay_frames: float  # mean over videos that had a delay; inf when nothing was detected

    def to_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self.config),
            "f1": self.f1,
            "pos": self.pos,
            "delay_s": self.delay_frames / 10.0 if np.isfinite(self.delay_frames) else -1.0,
        }


def evaluate_grid(
    videos: Sequence[PSRVideo],
    probs: Sequence[np.ndarray],
    spec: PSRSpec,
    priors: Sequence[ProcedurePrior | None],
    dwells: Sequence[int],
) -> list[GridPoint]:
    """Score every decoder setting on ``videos`` (one prior per video): mean F1, POS and delay."""
    points: list[GridPoint] = []
    for ema in spec.emas:
        averages = [ema_filter(p, ema) for p in probs]
        for on, off, dwell in product(spec.theta_ons, spec.theta_offs, dwells):
            if off >= on:
                continue
            config = DecoderConfig(ema, on, off, dwell)
            f1s, poss, delays = [], [], []
            for v, avg, prior in zip(videos, averages, priors, strict=True):
                result = psr_performance(v.gt, decode_from_average(avg, v.frames, config, prior))
                f1s.append(float(result["f1"]))  # type: ignore[arg-type]
                poss.append(float(result["pos"]))  # type: ignore[arg-type]
                if result["mean_delay_frames"] is not None:
                    delays.append(float(result["mean_delay_frames"]))  # type: ignore[arg-type]
            points.append(
                GridPoint(
                    config,
                    float(np.mean(f1s)),
                    float(np.mean(poss)),
                    float(np.mean(delays)) if delays else float("inf"),
                )
            )
    return points


def pareto_front(points: Sequence[GridPoint]) -> list[GridPoint]:
    """Settings not dominated on (higher F1, lower delay), sorted by delay."""
    front = [
        p
        for p in points
        if not any(
            (q.f1 >= p.f1 and q.delay_frames < p.delay_frames)
            or (q.f1 > p.f1 and q.delay_frames <= p.delay_frames)
            for q in points
        )
    ]
    return sorted(front, key=lambda p: (p.delay_frames, -p.f1))


def choose_decoder(points: Sequence[GridPoint], delay_cap_s: float | None = None) -> GridPoint:
    """Best F1 (ties: lower delay); with ``delay_cap_s`` only among settings whose mean delay on
    the selection videos is within the cap — if none is, the fastest setting."""
    candidates = list(points)
    if delay_cap_s is not None:
        within = [p for p in candidates if p.delay_frames <= delay_cap_s * 10.0]
        if not within:
            return min(candidates, key=lambda p: (p.delay_frames, -p.f1))
        candidates = within
    return max(candidates, key=lambda p: (p.f1, -p.delay_frames))


def _cap_tag(cap: float | None) -> str:
    return "" if cap is None else f"_cap{cap:g}"


def _fit_heads(
    train: Sequence[PSRVideo], spec: PSRSpec, device: str, heads: Sequence[str], log_key: str
) -> tuple[dict[str, object], dict[str, object]]:
    predictors: dict[str, object] = {}
    logs: dict[str, object] = {}
    if "linear" in heads:
        predictors["linear"] = train_state_head(
            np.concatenate([v.features for v in train]).astype(np.float32),
            np.concatenate([v.targets for v in train]),
            spec,
            device,
        )
    if "mstcn" in heads:
        predictors["mstcn"], logs[f"mstcn/{log_key}"], _ = train_state_mstcn(train, spec, device)
    return predictors, logs


def inner_folds(
    train: Sequence[PSRVideo], n_folds: int = 4
) -> list[tuple[list[PSRVideo], list[PSRVideo]]]:
    """Deterministic grouped folds over the training participants (sorted ids, round-robin)."""
    participants = sorted({v.participant for v in train})
    groups = [set(participants[i::n_folds]) for i in range(n_folds)]
    return [
        (
            [v for v in train if v.participant not in group],
            [v for v in train if v.participant in group],
        )
        for group in groups
        if group
    ]


def _select_nested(
    train: Sequence[PSRVideo],
    spec: PSRSpec,
    device: str,
    heads: Sequence[str],
    decoders: Sequence[str],
    log_key: str,
) -> tuple[dict[str, list[GridPoint]], dict[str, object], int]:
    """Select every decoder on out-of-fold probabilities of the training recordings.

    Each inner fold fits the heads on the other training participants and predicts its own;
    the prior used while scoring a video is learned from that video's inner-train fold, so no
    label of a video influences the settings it is decoded with.
    """
    oof: dict[str, list[np.ndarray | None]] = {head: [None] * len(train) for head in heads}
    fold_priors: list[dict[str, ProcedurePrior] | None] = [None] * len(train)
    logs: dict[str, object] = {}
    position = {v.video_id: i for i, v in enumerate(train)}
    epoch_grid = tuple(sorted(spec.mstcn_epoch_grid)) if "mstcn" in heads else ()
    # With an epoch grid the inner MS-TCN++ heads train to max(grid) and snapshot their held-out
    # probabilities at every grid epoch; the epoch with the lowest mean held-out BCE wins.
    snapshots_by_fold: list[tuple[list[PSRVideo], dict[int, list[np.ndarray]]]] = []
    bce_by_epoch: dict[int, list[float]] = defaultdict(list)
    for k, (inner_train, inner_held) in enumerate(inner_folds(train)):
        fold_heads = [h for h in heads if not (h == "mstcn" and epoch_grid)]
        predictors, fold_logs = _fit_heads(
            inner_train, spec, device, fold_heads, f"{log_key}/inner{k}"
        )
        if epoch_grid:
            _, mstcn_log, snapshots = train_state_mstcn(
                inner_train,
                replace(spec, mstcn_epochs=max(epoch_grid)),
                device,
                eval_videos=inner_held,
                eval_epochs=epoch_grid,
            )
            fold_logs[f"mstcn/{log_key}/inner{k}"] = mstcn_log
            snapshots_by_fold.append((list(inner_held), snapshots))
            for epoch, bce in mstcn_log["heldout_bce"].items():  # type: ignore[union-attr]
                bce_by_epoch[int(epoch)].append(float(bce) * len(inner_held))
        logs.update(fold_logs)
        priors = {kind: learn_prior(inner_train, kind) for kind in ("assy", "main")}
        for v in inner_held:
            fold_priors[position[v.video_id]] = priors
            for head, predictor in predictors.items():
                oof[head][position[v.video_id]] = predictor.predict_proba(v.features)  # type: ignore[attr-defined]
    chosen_epochs = spec.mstcn_epochs
    if epoch_grid:
        mean_bce = {e: sum(v) / len(train) for e, v in bce_by_epoch.items()}
        chosen_epochs = min(mean_bce, key=lambda e: (mean_bce[e], e))
        for held_videos, snapshots in snapshots_by_fold:
            for v, probs in zip(held_videos, snapshots[chosen_epochs], strict=True):
                oof["mstcn"][position[v.video_id]] = probs
        logs[f"mstcn/{log_key}/epoch_selection"] = {
            "grid": list(epoch_grid),
            "mean_heldout_bce": {str(e): mean_bce[e] for e in sorted(mean_bce)},
            "chosen": chosen_epochs,
        }
    selected: dict[str, list[GridPoint]] = {}
    for head in heads:
        probs = [p for p in oof[head]]
        assert all(p is not None for p in probs) and all(fp is not None for fp in fold_priors)
        for decoder in decoders:
            with_prior = decoder == "prior_dwell"
            priors_per_video = [
                fp[v.kind] if with_prior else None  # type: ignore[index]
                for v, fp in zip(train, fold_priors, strict=True)
            ]
            selected[f"{head}_{decoder}"] = evaluate_grid(
                train,
                probs,  # type: ignore[arg-type]
                spec,
                priors_per_video,
                spec.min_dwells if with_prior else (0,),
            )
    return selected, logs, chosen_epochs


def _fit_and_decode(
    train: Sequence[PSRVideo],
    test: Sequence[PSRVideo],
    fold_name: str,
    spec: PSRSpec,
    device: str,
    heads: Sequence[str],
    decoders: Sequence[str],
    selection: str = "in_sample",
    run_suffix: str = "",
) -> tuple[list[CompletionRow], dict[str, object], dict[str, object]]:
    """Fit every head on ``train``, select every decoder (in-sample or nested), decode ``test``."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    rows: list[CompletionRow] = []
    training_logs: dict[str, object] = {}
    priors = {kind: learn_prior(train, kind) for kind in ("assy", "main")}
    nested: dict[str, tuple[DecoderConfig, float]] = {}
    if selection == "nested":
        nested, training_logs, chosen_epochs = _select_nested(
            train, spec, device, heads, decoders, f"{fold_name}{run_suffix}"
        )
        spec = replace(spec, mstcn_epochs=chosen_epochs)
    predictors, final_logs = _fit_heads(train, spec, device, heads, f"{fold_name}{run_suffix}")
    training_logs.update(final_logs)
    fold: dict[str, object] = {
        "train_videos": len(train),
        "train_participants": sorted({v.participant for v in train}),
        "eval_videos": len(test),
        "selection": selection,
        "mstcn_epochs_used": spec.mstcn_epochs,
        "prior": {
            kind: {
                "active": prior.active.tolist(),
                "initial_installed": prior.initial_installed.tolist(),
            }
            for kind, prior in priors.items()
        },
        "decoders": {},
    }
    for v in test:
        rows.extend(
            CompletionRow(v.video_id, v.participant, "gt", c.frame, c.step, fold_name, "")
            for c in v.gt
        )
    caps: list[float | None] = list(spec.delay_caps_s) if spec.delay_caps_s else [None]
    fold["pareto"] = {}
    for head, predictor in predictors.items():
        test_probs = [predictor.predict_proba(v.features) for v in test]  # type: ignore[attr-defined]
        train_probs = (
            [predictor.predict_proba(v.features) for v in train]  # type: ignore[attr-defined]
            if selection == "in_sample"
            else []
        )
        for decoder in decoders:
            with_prior = decoder == "prior_dwell"
            if selection == "nested":
                points = nested[f"{head}_{decoder}"]
            else:
                points = evaluate_grid(
                    train,
                    train_probs,
                    spec,
                    [priors[v.kind] if with_prior else None for v in train],
                    spec.min_dwells if with_prior else (0,),
                )
            fold["pareto"][f"{head}_{decoder}"] = [p.to_dict() for p in pareto_front(points)]  # type: ignore[index]
            for cap in caps:
                chosen = choose_decoder(points, cap)
                run = f"{head}_{decoder}{_cap_tag(cap)}{run_suffix}"
                fold["decoders"][run] = {  # type: ignore[index]
                    "decoder": asdict(chosen.config),
                    "grid_best": chosen.f1,
                    "selected": chosen.to_dict(),
                    "delay_cap_s": cap,
                }
                for v, p in zip(test, test_probs, strict=True):
                    pred = decode_completions(
                        p, v.frames, chosen.config, priors[v.kind] if with_prior else None
                    )
                    rows.extend(
                        CompletionRow(
                            v.video_id, v.participant, "pred", c.frame, c.step, fold_name, run
                        )
                        for c in pred
                    )
    return rows, fold, training_logs


def read_split_ids(path: Path) -> set[str]:
    """Video ids listed in a committed split CSV (``video_id,participant,n_segments``)."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["video_id"] for row in reader}


def run_psr_baseline(
    features_dir: Path,
    psr_dir: Path,
    out_dir: Path,
    spec: PSRSpec | None = None,
    device: str = "cpu",
    heads: Sequence[str] = HEADS,
    decoders: Sequence[str] = DECODERS,
    train_ids: set[str] | None = None,
    eval_ids: set[str] | None = None,
    selection: str = "in_sample",
    seeds: Sequence[int] = (0,),
) -> dict[str, object]:
    """Leave-one-participant-out over every recording in ``psr_dir``, or — when ``train_ids`` and
    ``eval_ids`` are given — one fit on the train recordings evaluated on the eval recordings.

    ``selection`` is how every decoder is chosen: ``in_sample`` on the same recordings the head
    was fitted on, ``nested`` on out-of-fold predictions of the training recordings. With several
    ``seeds`` every head x decoder is repeated per seed as ``<head>_<decoder>_s<seed>`` and
    ``metrics.json`` gains a ``seed_summary``.
    """
    spec = spec or PSRSpec()
    if any(h not in HEADS for h in heads) or any(d not in DECODERS for d in decoders):
        raise ValueError(f"heads must be in {HEADS} and decoders in {DECODERS}")
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    if (train_ids is None) != (eval_ids is None):
        raise ValueError("train_ids and eval_ids must be given together")
    if not seeds:
        raise ValueError("at least one seed is needed")
    started = time.perf_counter()
    videos = load_psr_videos(features_dir, psr_dir)
    steps = load_procedure_info(PROCEDURE_INFO)
    audits = [audit_recording(psr_dir / v.video_id, v.n_frames, steps) for v in videos]
    if any(a["problems"] for a in audits):
        raise ValueError(f"PSR label problems: {[a for a in audits if a['problems']]}")
    rows: list[CompletionRow] = []
    folds: dict[str, object] = {}
    training_logs: dict[str, object] = {}
    if train_ids is not None and eval_ids is not None:
        train = [v for v in videos if v.video_id in train_ids]
        test = [v for v in videos if v.video_id in eval_ids]
        missing = (train_ids | eval_ids) - {v.video_id for v in videos}
        if missing or not train or not test:
            raise ValueError(f"split videos without PSR labels/features: {sorted(missing)}")
        if {v.participant for v in train} & {v.participant for v in test}:
            raise ValueError("train and eval splits share a participant")
        how = (
            "decoder selected on those same recordings"
            if selection == "in_sample"
            else "decoder selected on out-of-fold predictions of those recordings (grouped 4-fold by participant)"
        )
        protocol = (
            f"fit once on the {len(train)} train-split recordings ({len({v.participant for v in train})} "
            f"participants), {how}, evaluated on the {len(test)} eval-split recordings"
        )
        splits = [(train, test, "train->eval")]
        evaluated = test
    else:
        participants = sorted({v.participant for v in videos})
        if len(participants) < 2:
            raise ValueError("leave-one-participant-out needs at least two participants")
        how = (
            "decoder selected on each fold's training videos"
            if selection == "in_sample"
            else "decoder selected on out-of-fold predictions of each fold's training videos"
        )
        protocol = f"leave-one-participant-out over the val participants; {how}"
        splits = [
            (
                [v for v in videos if v.participant != held_out],
                [v for v in videos if v.participant == held_out],
                held_out,
            )
            for held_out in participants
        ]
        evaluated = videos
    if len(seeds) > 1:
        protocol += f"; seeds {list(seeds)} reported as separate runs"
    for train, test, fold_name in splits:
        gt_written = False
        for seed in seeds:
            seeded = replace(spec, seed=seed)
            suffix = f"_s{seed}" if len(seeds) > 1 else ""
            fold_rows, fold, logs = _fit_and_decode(
                train, test, fold_name, seeded, device, heads, decoders, selection, suffix
            )
            if gt_written:
                fold_rows = [r for r in fold_rows if r.source != "gt"]
            gt_written = True
            rows.extend(fold_rows)
            training_logs.update(logs)
            if fold_name in folds:
                folds[fold_name]["decoders"].update(fold["decoders"])  # type: ignore[index]
            else:
                folds[fold_name] = fold
    videos = evaluated
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "psr_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_completions(out_dir / "completions_val.csv", rows)
    run_names = sorted({run for fold in folds.values() for run in fold["decoders"]})  # type: ignore[index]
    metrics = score_completions(rows, n_boot=spec.n_boot, seed=spec.seed, run_names=run_names)
    error_videos = sorted(v.video_id for v in videos if v.has_errors)
    metrics["videos_with_error_steps"] = error_videos
    clean_videos = [v.video_id for v in videos if not v.has_errors]
    subsets = {"no_errors": clean_videos, "with_errors": error_videos}
    for label, subset in subsets.items():
        if not subset:
            continue
        sub = score_completions(
            [r for r in rows if r.video_id in set(subset)],
            n_boot=spec.n_boot,
            seed=spec.seed,
            run_names=run_names,
        )
        if "runs" in metrics:
            for run, scored in sub["runs"].items():  # type: ignore[union-attr]
                metrics["runs"][run][f"summary_{label}"] = {  # type: ignore[index]
                    "n_videos": sub["n_videos"],
                    "summary": scored["summary"],
                }
        else:
            metrics[f"summary_{label}"] = {"n_videos": sub["n_videos"], "summary": sub["summary"]}
    meta_path = features_dir / "meta.json"
    config = {
        "protocol": protocol,
        "spec": asdict(spec),
        "heads": {
            "linear": "11-way multi-label logistic regression on frozen frame features (BCE, full-batch Adam)",
            "mstcn": "causal MS-TCN++ (sigmoid outputs) on frozen frame features, per-stage BCE + smoothing, fixed epochs",
        },
        "decoders": {
            "plain": "causal EMA + hysteresis; install = up-crossing of theta_on, remove = down-crossing of theta_off",
            "prior_dwell": "plain + minimum dwell before an event + procedure prior learned from the fold's training labels (active components and initial state per recording kind)",
        },
        "runs": [f"{h}_{d}" for h in heads for d in decoders],
        "folds": folds,
        "training": training_logs,
        "features": json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else {},
        "device": device,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tables = render_psr_tables(metrics, config)
    (out_dir / "tables.md").write_text(tables, encoding="utf-8", newline="\n")
    return {"config": config, "metrics": metrics, "tables": tables}


# ---------------------------------------------------------------------------------------------
# Tables


def _fmt(entry: Mapping[str, object], scale: float = 1.0, digits: int = 3) -> str:
    lo, hi = entry["ci95"]  # type: ignore[misc]
    point = float(entry["mean_over_videos"]) * scale  # type: ignore[arg-type]
    return f"{point:.{digits}f} [{float(lo) * scale:.{digits}f}, {float(hi) * scale:.{digits}f}]"


def _summary_rows(name: str, scored: Mapping[str, object], n_videos: object) -> list[str]:
    summary: Mapping[str, Mapping[str, object]] = scored["summary"]  # type: ignore[assignment]
    totals: Mapping[str, int] = scored["totals"]  # type: ignore[assignment]
    lines = [
        f"| {name} | all | {n_videos} | {_fmt(summary['pos'])} | {_fmt(summary['f1'])} | "
        f"{_fmt(summary['mean_delay_frames'], 0.1, 1)} | {totals['system_tp']} / {totals['system_fp']} / {totals['system_fn']} |"
    ]
    for label in ("no_errors", "with_errors"):
        sub = scored.get(f"summary_{label}")
        if sub:
            s: Mapping[str, Mapping[str, object]] = sub["summary"]  # type: ignore[index]
            lines.append(
                f"| {name} | {label} | {sub['n_videos']} | {_fmt(s['pos'])} | {_fmt(s['f1'])} | "  # type: ignore[index]
                f"{_fmt(s['mean_delay_frames'], 0.1, 1)} | — |"
            )
    return lines


def _per_video_rows(per_video: Mapping[str, Mapping[str, object]]) -> list[str]:
    lines = []
    for video in sorted(per_video):
        m = per_video[video]
        delay = (
            "—" if m["mean_delay_frames"] is None else f"{float(m['mean_delay_frames']) / 10:.1f}"
        )  # type: ignore[arg-type]
        lines.append(
            f"| {video} | {float(m['pos']):.3f} | {float(m['f1']):.3f} | {delay} | "  # type: ignore[arg-type]
            f"{m['system_tp']} / {m['system_fp']} / {m['system_fn']} | {m['n_gt']} / {m['n_pred']} |"
        )
    return lines


def render_psr_tables(metrics: Mapping[str, object], config: Mapping[str, object]) -> str:
    runs: Mapping[str, Mapping[str, object]] = (
        metrics["runs"] if "runs" in metrics else {str(metrics.get("run", "pred")): metrics}  # type: ignore[assignment]
    )
    lines = [
        f"Development result on `val` ({metrics['n_videos']} videos, {metrics['n_participants']} participants), "
        f"leave-one-participant-out; 95% CI = participant bootstrap "
        f"({metrics['bootstrap']['n_boot']} draws, seed {metrics['bootstrap']['seed']}).",  # type: ignore[index]
        "",
        "| run | subset | videos | POS | F1 (system) | mean delay (s) | TP / FP / FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, scored in runs.items():
        lines += _summary_rows(name, scored, metrics["n_videos"])
    for name, scored in runs.items():
        lines += [
            "",
            f"Per-video, `{name}`:",
            "",
            "| video | POS | F1 | delay (s) | TP / FP / FN | n_gt / n_pred |",
            "|---|---|---|---|---|---|",
        ]
        lines += _per_video_rows(scored["per_video"])  # type: ignore[arg-type]
    seeds: Mapping[str, Mapping[str, object]] = metrics.get("seed_summary", {})  # type: ignore[assignment]
    if seeds:
        lines += [
            "",
            "Across seeds (mean ± std of the per-seed means over videos):",
            "",
            "| run | seeds | POS | F1 (system) | mean delay (s) | TP / FP / FN (mean) |",
            "|---|---|---|---|---|---|",
        ]
        for base, entry in seeds.items():
            pos: Mapping[str, float] = entry["pos"]  # type: ignore[assignment]
            f1: Mapping[str, float] = entry["f1"]  # type: ignore[assignment]
            delay: Mapping[str, float] = entry["mean_delay_frames"]  # type: ignore[assignment]
            lines.append(
                f"| {base} | {entry['n_seeds']} | {pos['mean']:.3f} ± {pos['std']:.3f} | "
                f"{f1['mean']:.3f} ± {f1['std']:.3f} | {delay['mean'] / 10:.1f} ± {delay['std'] / 10:.1f} | "
                f"{entry['system_tp']:.1f} / {entry['system_fp']:.1f} / {entry['system_fn']:.1f} |"
            )
    folds: Mapping[str, Mapping[str, object]] = config.get("folds", {})  # type: ignore[assignment]
    if folds:
        lines += [
            "",
            "Decoder selected per fold (on that fold's training videos):",
            "",
            "| held-out participant | run | ema | theta_on | theta_off | min_dwell |",
            "|---|---|---|---|---|---|",
        ]
        for held_out in sorted(folds):
            fold = folds[held_out]
            entries: Mapping[str, Mapping[str, object]] = fold.get("decoders", {"pred": fold})  # type: ignore[assignment]
            for run in sorted(entries):
                d: Mapping[str, object] = entries[run]["decoder"]  # type: ignore[assignment]
                lines.append(
                    f"| {held_out} | {run} | {d['ema']} | {d['theta_on']} | {d['theta_off']} | {d.get('min_dwell', 0)} |"
                )
    return "\n".join(lines) + "\n"
