"""Online procedure-step baseline on IndustReal val (spec 4.3 sanity run, development tier).

Only the *val* recordings' PSR labels are on disk (the train and test archives were not
downloaded), so the head is trained leave-one-participant-out inside val: for each of the 5 val
participants, an 11-way multi-label logistic head ("component k is correctly installed") is fitted
on the frozen DINOv2 features of the other 4 participants' videos, decoded causally on the
held-out participant, and scored with :func:`sop_monitor.metrics.online.psr_performance` against
``PSR_labels.csv``. Decoder settings are chosen on the fold's own training videos, never on the
held-out participant. Every completion the decoder emits — and every ground-truth completion — is
written to ``completions_val.csv`` so the metrics can be recomputed without features.

Decoder (causal): exponential moving average of the per-component probability, then hysteresis —
an install completion (``3k``) is emitted when the average crosses ``theta_on`` upwards, a removal
(``3k + 2``) when it crosses ``theta_off`` downwards. Incorrect installs (``3k + 1``) are never
predicted: the head is trained on error-free state targets, and the reference scores
``PSR_labels.csv`` (correct steps only) in exactly this way.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    load_procedure_info,
    load_psr_labels,
    load_psr_raw,
)
from sop_monitor.metrics.offline import subject_bootstrap
from sop_monitor.metrics.online import Completion, psr_performance

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


def decode_completions(
    probs: np.ndarray, frames: np.ndarray, config: DecoderConfig
) -> list[Completion]:
    """Causal EMA + hysteresis over ``probs[t, k]``; returns completions in emission order."""
    if not 0.0 <= config.ema < 1.0:
        raise ValueError("ema must be in [0, 1)")
    if not config.theta_off < config.theta_on:
        raise ValueError("theta_off must be below theta_on")
    average = np.zeros(probs.shape[1], dtype=np.float64)
    installed = np.zeros(probs.shape[1], dtype=bool)
    out: list[Completion] = []
    for t in range(probs.shape[0]):
        average = config.ema * average + (1.0 - config.ema) * probs[t]
        for k in range(probs.shape[1]):
            if not installed[k] and average[k] > config.theta_on:
                installed[k] = True
                out.append(Completion(frame=int(frames[t]), step=3 * k + INSTALL))
            elif installed[k] and average[k] < config.theta_off:
                installed[k] = False
                out.append(Completion(frame=int(frames[t]), step=3 * k + REMOVE))
    return out


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


def write_completions(path: Path, rows: Sequence[CompletionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("video_id,participant,source,frame,step,fold\n")
        for r in rows:
            handle.write(f"{r.video_id},{r.participant},{r.source},{r.frame},{r.step},{r.fold}\n")


def read_completions(path: Path) -> list[CompletionRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["video_id", "participant", "source", "frame", "step", "fold"]:
            raise ValueError(f"unexpected completions header: {header}")
        return [CompletionRow(r[0], r[1], r[2], int(r[3]), int(r[4]), r[5]) for r in reader if r]


def score_completions(
    rows: Sequence[CompletionRow], n_boot: int = 2000, seed: int = 0
) -> dict[str, object]:
    """Per-video PSR metrics, their mean over videos (the reference's aggregation) and participant CIs."""
    by_video: dict[str, dict[str, list[Completion]]] = defaultdict(lambda: {"gt": [], "pred": []})
    participants: dict[str, str] = {}
    for r in rows:
        by_video[r.video_id][r.source].append(Completion(r.frame, r.step))
        participants[r.video_id] = r.participant
    per_video: dict[str, dict[str, object]] = {}
    for video in sorted(by_video):
        gt = sorted(by_video[video]["gt"], key=lambda c: c.frame)
        pred = sorted(by_video[video]["pred"], key=lambda c: c.frame)
        per_video[video] = psr_performance(gt, pred)
    per_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for video, metrics in per_video.items():
        per_subject[participants[video]].append(metrics)

    def mean_of(items: Sequence[Mapping[str, object]], key: str) -> float:
        values = [float(m[key]) for m in items if m[key] is not None]  # type: ignore[arg-type]
        return float(np.mean(values)) if values else float("nan")

    summary: dict[str, dict[str, object]] = {}
    for key in ("pos", "f1", "mean_delay_frames"):
        boot = subject_bootstrap(
            per_subject,  # type: ignore[arg-type]
            lambda items, key=key: mean_of(items, key),  # type: ignore[arg-type]
            n_boot=n_boot,
            seed=seed,
        )
        summary[key] = {"mean_over_videos": boot.point, "ci95": [boot.lower, boot.upper]}
    totals = {
        k: int(sum(int(m[k]) for m in per_video.values()))  # type: ignore[arg-type]
        for k in ("system_tp", "system_fp", "system_fn", "n_gt", "n_pred")
    }
    return {
        "n_videos": len(per_video),
        "n_participants": len(per_subject),
        "aggregation": "unweighted mean of per-video metrics (as psr_baseline.py); delay in frames at 10 fps",
        "bootstrap": {"unit": "participant", "n_boot": n_boot, "seed": seed, "alpha": 0.05},
        "summary": summary,
        "totals": totals,
        "per_video": per_video,
        "gt_sanity": gt_sanity(
            {v: sorted(by_video[v]["gt"], key=lambda c: c.frame) for v in sorted(by_video)}
        ),
    }


def gt_sanity(gts: Mapping[str, Sequence[Completion]]) -> dict[str, dict[str, float]]:
    """Metric behaviour on the real label structures under controlled perturbations of the GT.

    ``identity`` must give POS 1, F1 ≈ 1 and delay 0; ``shift_plus_30_frames`` must keep POS/F1
    and report a 30-frame delay; ``drop_last`` costs one FN per video; ``swap_first_two``
    exchanges the times of the first two completions (one transposition, POS ``1 - 1/n``).
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
# Leave-one-participant-out run


@dataclass(frozen=True)
class PSRSpec:
    weight_decay: float = 1e-3
    epochs: int = 300
    lr: float = 5e-3
    seed: int = 0
    n_boot: int = 2000
    emas: tuple[float, ...] = (0.0, 0.8, 0.9, 0.95)
    theta_ons: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)
    theta_offs: tuple[float, ...] = (0.2, 0.3, 0.4)
    selection_metric: str = "f1"


@dataclass(frozen=True)
class PSRVideo:
    video_id: str
    participant: str
    frames: np.ndarray
    features: np.ndarray
    targets: np.ndarray  # float32[n_sampled, N_COMPONENTS], 1 where correctly installed
    gt: list[Completion]  # PSR_labels.csv (correct steps only)
    has_errors: bool
    n_frames: int


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
        states = frame_states(load_psr_raw(rec_dir / "PSR_labels_raw.csv"), n_frames)[frames]
        with_errors = load_psr_labels(rec_dir / "PSR_labels_with_errors.csv")
        videos.append(
            PSRVideo(
                video_id=rec_dir.name,
                participant=participant_of(rec_dir.name),
                frames=frames,
                features=features,
                targets=(states == 1).astype(np.float32),
                gt=load_psr_labels(rec_dir / "PSR_labels.csv"),
                has_errors=any(c.step % 3 == INCORRECT for c in with_errors),
                n_frames=n_frames,
            )
        )
    return videos


@dataclass(frozen=True)
class StateHead:
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


def _select_decoder(
    videos: Sequence[PSRVideo], probs: Sequence[np.ndarray], spec: PSRSpec
) -> tuple[DecoderConfig, dict[str, float]]:
    """Grid-search the decoder on the given (training) videos by mean per-video ``selection_metric``."""
    grid: dict[str, float] = {}
    best: tuple[float, DecoderConfig] | None = None
    for ema, on, off in product(spec.emas, spec.theta_ons, spec.theta_offs):
        if off >= on:
            continue
        config = DecoderConfig(ema, on, off)
        scores = [
            psr_performance(v.gt, decode_completions(p, v.frames, config))[spec.selection_metric]
            for v, p in zip(videos, probs, strict=True)
        ]
        score = float(np.mean([s for s in scores if s is not None]))  # type: ignore[arg-type]
        grid[f"ema={ema},on={on},off={off}"] = score
        if best is None or score > best[0]:
            best = (score, config)
    assert best is not None
    return best[1], grid


def run_psr_baseline(
    features_dir: Path,
    psr_dir: Path,
    out_dir: Path,
    spec: PSRSpec | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    spec = spec or PSRSpec()
    started = time.perf_counter()
    videos = load_psr_videos(features_dir, psr_dir)
    steps = load_procedure_info(PROCEDURE_INFO)
    audits = [audit_recording(psr_dir / v.video_id, v.n_frames, steps) for v in videos]
    if any(a["problems"] for a in audits):
        raise ValueError(f"PSR label problems: {[a for a in audits if a['problems']]}")
    participants = sorted({v.participant for v in videos})
    if len(participants) < 2:
        raise ValueError("leave-one-participant-out needs at least two participants")
    rows: list[CompletionRow] = []
    folds: dict[str, object] = {}
    for held_out in participants:
        train = [v for v in videos if v.participant != held_out]
        test = [v for v in videos if v.participant == held_out]
        head = train_state_head(
            np.concatenate([v.features for v in train]).astype(np.float32),
            np.concatenate([v.targets for v in train]),
            spec,
            device,
        )
        train_probs = [head.predict_proba(v.features) for v in train]
        config, grid = _select_decoder(train, train_probs, spec)
        folds[held_out] = {
            "decoder": asdict(config),
            "train_videos": len(train),
            "grid_best": max(grid.values()),
        }
        for v in test:
            pred = decode_completions(head.predict_proba(v.features), v.frames, config)
            rows.extend(
                CompletionRow(v.video_id, v.participant, "gt", c.frame, c.step, held_out)
                for c in v.gt
            )
            rows.extend(
                CompletionRow(v.video_id, v.participant, "pred", c.frame, c.step, held_out)
                for c in pred
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "psr_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_completions(out_dir / "completions_val.csv", rows)
    metrics = score_completions(rows, n_boot=spec.n_boot, seed=spec.seed)
    error_videos = sorted(v.video_id for v in videos if v.has_errors)
    metrics["videos_with_error_steps"] = error_videos
    for label, subset in (
        ("no_errors", [v.video_id for v in videos if not v.has_errors]),
        ("with_errors", error_videos),
    ):
        if subset:
            sub = score_completions(
                [r for r in rows if r.video_id in set(subset)], n_boot=spec.n_boot, seed=spec.seed
            )
            metrics[f"summary_{label}"] = {"n_videos": sub["n_videos"], "summary": sub["summary"]}
    meta_path = features_dir / "meta.json"
    config = {
        "protocol": "leave-one-participant-out over the val participants; decoder selected on each fold's training videos",
        "spec": asdict(spec),
        "head": "11-way multi-label logistic regression on frozen frame features (BCE, full-batch Adam)",
        "decoder": "causal EMA + hysteresis; install = up-crossing of theta_on, remove = down-crossing of theta_off",
        "folds": folds,
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


def _fmt(entry: Mapping[str, object], scale: float = 1.0, digits: int = 3) -> str:
    lo, hi = entry["ci95"]  # type: ignore[misc]
    point = float(entry["mean_over_videos"]) * scale  # type: ignore[arg-type]
    return f"{point:.{digits}f} [{float(lo) * scale:.{digits}f}, {float(hi) * scale:.{digits}f}]"


def render_psr_tables(metrics: Mapping[str, object], config: Mapping[str, object]) -> str:
    summary: Mapping[str, Mapping[str, object]] = metrics["summary"]  # type: ignore[assignment]
    totals: Mapping[str, int] = metrics["totals"]  # type: ignore[assignment]
    lines = [
        f"Development result on `val` ({metrics['n_videos']} videos, {metrics['n_participants']} participants), "
        f"leave-one-participant-out; 95% CI = participant bootstrap "
        f"({metrics['bootstrap']['n_boot']} draws, seed {metrics['bootstrap']['seed']}).",  # type: ignore[index]
        "",
        "| subset | videos | POS | F1 (system) | mean delay (s) | TP / FP / FN |",
        "|---|---|---|---|---|---|",
        f"| all | {metrics['n_videos']} | {_fmt(summary['pos'])} | {_fmt(summary['f1'])} | "
        f"{_fmt(summary['mean_delay_frames'], 0.1, 1)} | {totals['system_tp']} / {totals['system_fp']} / {totals['system_fn']} |",
    ]
    for label in ("no_errors", "with_errors"):
        sub = metrics.get(f"summary_{label}")
        if sub:
            s: Mapping[str, Mapping[str, object]] = sub["summary"]  # type: ignore[index]
            lines.append(
                f"| {label} | {sub['n_videos']} | {_fmt(s['pos'])} | {_fmt(s['f1'])} | "  # type: ignore[index]
                f"{_fmt(s['mean_delay_frames'], 0.1, 1)} | — |"
            )
    per_video: Mapping[str, Mapping[str, object]] = metrics["per_video"]  # type: ignore[assignment]
    lines += [
        "",
        "Per-video:",
        "",
        "| video | POS | F1 | delay (s) | TP / FP / FN | n_gt / n_pred |",
        "|---|---|---|---|---|---|",
    ]
    for video in sorted(per_video):
        m = per_video[video]
        delay = (
            "—" if m["mean_delay_frames"] is None else f"{float(m['mean_delay_frames']) / 10:.1f}"
        )  # type: ignore[arg-type]
        lines.append(
            f"| {video} | {float(m['pos']):.3f} | {float(m['f1']):.3f} | {delay} | "  # type: ignore[arg-type]
            f"{m['system_tp']} / {m['system_fp']} / {m['system_fn']} | {m['n_gt']} / {m['n_pred']} |"
        )
    folds: Mapping[str, Mapping[str, object]] = config.get("folds", {})  # type: ignore[assignment]
    if folds:
        lines += [
            "",
            "Decoder selected per fold (on that fold's training videos):",
            "",
            "| held-out participant | ema | theta_on | theta_off |",
            "|---|---|---|---|",
        ]
        for held_out in sorted(folds):
            d: Mapping[str, float] = folds[held_out]["decoder"]  # type: ignore[assignment]
            lines.append(f"| {held_out} | {d['ema']} | {d['theta_on']} | {d['theta_off']} |")
    return "\n".join(lines) + "\n"
