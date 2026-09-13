"""HA-ViD offline TAS runner on synthetic caches: per-view + fused runs, I3D export, recomputation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sop_monitor.baseline import read_predictions, score_predictions  # noqa: E402
from sop_monitor.havid import (  # noqa: E402
    CAMERA_OF_VIEW,
    OFFICIAL_TRIM,
    SUBDATASETS,
    load_official_zip,
    load_temporal_zip,
    parse_temporal,
)
from sop_monitor.havid_tas import (  # noqa: E402
    HavidTASSpec,
    export_official_features,
    fuse_views,
    load_havid_videos,
    run_havid_tas,
)
from sop_monitor.mstcn import MSTCNSpec  # noqa: E402

RECORDINGS = {"S01A04I01": "train", "S03A04I01": "train", "S02A04I01": "val"}
N_FRAMES = 30
DIM = 6
TEMPORAL = {
    "lh_pt": "0 9 null\n10 19 ipsft\n20 29 w\n",
    "rh_pt": "0 14 null\n15 29 ibscb\n",
    "lh_aa": "0 9 null\n10 14 gsh\n15 29 msh\n",
    "rh_aa": "0 29 null\n",
}
SMALL = MSTCNSpec(
    num_layers_pg=3,
    num_layers_r=2,
    num_refinement_stages=1,
    num_f_maps=4,
    dropout=0.0,
    epochs=2,
    eval_every=1,
    n_boot=20,
    selection_metric="f1@10",
)


def build(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    temporal_zip = tmp_path / "temporal.zip"
    with zipfile.ZipFile(temporal_zip, "w") as archive:
        for sub in SUBDATASETS:
            for rec in RECORDINGS:
                archive.writestr(
                    f"HAViD_temporalAnnotation/{sub}/temporal_timestamps/{rec}_{sub}_temporal.txt",
                    TEMPORAL[sub[6:]],
                )
    official_zip = tmp_path / "official.zip"
    with zipfile.ZipFile(official_zip, "w") as archive:
        for sub in SUBDATASETS:
            camera = CAMERA_OF_VIEW[int(sub[4])]
            labels = sorted({s.label for s in parse_temporal(TEMPORAL[sub[6:]])})
            archive.writestr(
                f"{sub}/mapping.txt", "".join(f"{i} {label}\n" for i, label in enumerate(labels))
            )
            archive.writestr(f"{sub}/splits/train.split1.bundle", "")
            archive.writestr(f"{sub}/splits/test.split1.bundle", "")
            for rec in RECORDINGS:
                archive.writestr(f"{sub}/groundTruth/{rec}{camera}.txt", "")
        for rec in RECORDINGS:
            for camera in CAMERA_OF_VIEW.values():
                array = rng.standard_normal((DIM, N_FRAMES - 2 * OFFICIAL_TRIM))
                buffer = tmp_path / "tmp.npy"
                np.save(buffer, array)
                archive.write(buffer, f"features/{rec}{camera}.npy")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "val"):
        lines = ["video_id,subject,recording,view"]
        for rec, where in RECORDINGS.items():
            if where == split:
                lines += [f"{rec}{CAMERA_OF_VIEW[v]},{rec[:3]},{rec},{v}" for v in range(3)]
        (split_dir / f"{split}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    features_dir = tmp_path / "dinov2"
    features_dir.mkdir()
    (features_dir / "meta.json").write_text(
        json.dumps({"model": "toy", "stride": 1}), encoding="utf-8"
    )
    for rec in RECORDINGS:
        for camera in CAMERA_OF_VIEW.values():
            np.savez(
                features_dir / f"{rec}{camera}.npz",
                frames=np.arange(N_FRAMES, dtype=np.int64),
                features=rng.standard_normal((N_FRAMES, DIM)).astype(np.float16),
                n_frames=np.int64(N_FRAMES),
            )
    return {
        "temporal": temporal_zip,
        "official": official_zip,
        "splits": split_dir,
        "features": features_dir,
    }


def test_run_writes_per_view_and_fused_runs(tmp_path: Path) -> None:
    paths = build(tmp_path)
    out = tmp_path / "run"
    result = run_havid_tas(
        paths["features"],
        paths["splits"],
        paths["temporal"],
        paths["official"],
        out,
        HavidTASSpec(mstcn=SMALL),
        device="cpu",
    )
    fused = {
        f"fusion{rule}_{name}" for rule in ("", "_geo", "_conf") for name in ("causal", "offline")
    }
    trained = {f"view{v}_{name}" for v in range(3) for name in ("causal", "offline")}
    expected = {"majority"} | fused | trained
    assert set(result["metrics"]["runs"]) == expected
    rows = read_predictions(out / "predictions_val.csv")
    assert len(rows) == N_FRAMES
    assert {r.video_id for r in rows} == {"S02A04I01"} and rows[0].participant == "S02"
    assert result["config"]["classes"]["n_classes"] == 4  # background + null, ipsft, w
    assert result["config"]["classes"]["class_id_to_label"] == {"0": "ipsft", "1": "null", "2": "w"}
    assert set(result["config"]["training"]) == trained
    assert all("best_val_f1@10" in log for log in result["config"]["training"].values())
    assert "f1@10" in result["config"]["model"]
    assert set(result["config"]["runs"]) == expected
    recomputed = score_predictions(rows, n_boot=20, seed=0)
    assert recomputed["runs"] == result["metrics"]["runs"]
    tables = (out / "tables.md").read_text(encoding="utf-8")
    assert tables.startswith("Development result on `val`") and "| fusion_causal |" in tables


def test_fusion_rules_are_parameter_free_and_differ_where_they_should() -> None:
    # two views, three frames, three classes; view 1 is confident, view 0 is unsure at frame 0
    view0 = np.array([[0.34, 0.33, 0.33], [0.6, 0.2, 0.2], [0.0, 0.5, 0.5]])
    view1 = np.array([[0.05, 0.9, 0.05], [0.5, 0.25, 0.25], [0.8, 0.1, 0.1]])
    fused = fuse_views(np.stack([view0, view1]))
    assert set(fused) == {"", "_geo", "_conf"}
    for rule, probs in fused.items():
        assert probs.shape == (3, 3) and np.allclose(probs.sum(axis=1), 1.0), rule
    # frame 0: the mean and the confidence-weighted mean both follow the confident view
    assert fused[""][0].argmax() == 1 and fused["_conf"][0].argmax() == 1
    # frame 0: confidence weighting moves further towards view 1 than the plain mean does
    assert fused["_conf"][0, 1] > fused[""][0, 1]
    # frame 2: view 0 assigns ~0 to class 0, which the geometric mean vetoes and the mean does not
    assert fused[""][2].argmax() == 0 and fused["_geo"][2].argmax() != 0
    with pytest.raises(ValueError):
        fuse_views(view0)


def test_official_features_export_and_loader(tmp_path: Path) -> None:
    paths = build(tmp_path)
    out = tmp_path / "i3d"
    assert export_official_features(paths["official"], out, ["S02A04I01M0", "S02A04I01S1"]) == 2
    assert export_official_features(paths["official"], out, ["S02A04I01M0"]) == 0
    with np.load(out / "S02A04I01M0.npz") as payload:
        assert payload["frames"].tolist() == list(range(OFFICIAL_TRIM, N_FRAMES - OFFICIAL_TRIM))
        assert int(payload["n_frames"]) == N_FRAMES
        assert payload["features"].shape == (N_FRAMES - 2 * OFFICIAL_TRIM, DIM)
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["model"] == "i3d_official"
    temporal, _ = load_temporal_zip(paths["temporal"])
    mapping = load_official_zip(paths["official"], ["view0_lh_pt"])["view0_lh_pt"].mapping
    ids = {label: i for i, label in mapping.items()}
    videos = load_havid_videos(out, paths["splits"] / "val.csv", temporal["view0_lh_pt"], ids, 0)
    assert len(videos) == 1 and videos[0].video_id == "S02A04I01"
    assert videos[0].gt.tolist() == [ids["null"]] * 5 + [ids["ipsft"]] * 10 + [ids["w"]] * 5
    np.savez(
        out / "S02A04I01S2.npz",
        frames=np.arange(3),
        features=np.zeros((3, DIM), np.float32),
        n_frames=np.int64(3),
    )
    with pytest.raises(ValueError, match="cache covers 3 frames"):
        load_havid_videos(out, paths["splits"] / "val.csv", temporal["view0_lh_pt"], ids, 2)
    with pytest.raises(FileNotFoundError):
        export_official_features(paths["official"], out, ["S09A04I01M0"])


def test_lookahead_delays_targets_and_realigns_outputs(tmp_path: Path) -> None:
    from sop_monitor.baseline import VideoFeatures
    from sop_monitor.havid_tas import advance_outputs, delay_labels

    video = VideoFeatures("r", "S01", np.arange(5), np.zeros((5, 2)), np.array([1, 1, 2, 2, 3]))
    assert delay_labels([video], 0)[0] is video
    assert delay_labels([video], 2)[0].gt.tolist() == [1, 1, 1, 1, 2]
    assert delay_labels([video], 9)[0].gt.tolist() == [1, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        delay_labels([video], -1)
    probs = np.arange(10, dtype=float).reshape(5, 2)
    assert advance_outputs(probs, 2)[:, 0].tolist() == [4.0, 6.0, 8.0, 8.0, 8.0]
    assert advance_outputs(probs, 7)[:, 0].tolist() == [8.0] * 5
    # a perfect delayed causal network, re-aligned, reproduces the original labels exactly
    delayed = delay_labels([video], 2)[0].gt
    one_hot = np.eye(4)[delayed]
    assert advance_outputs(one_hot, 2).argmax(axis=1)[:3].tolist() == video.gt[:3].tolist()

    paths = build(tmp_path)
    result = run_havid_tas(
        paths["features"],
        paths["splits"],
        paths["temporal"],
        paths["official"],
        tmp_path / "la",
        HavidTASSpec(mstcn=SMALL, lookahead=3, offline=False),
        device="cpu",
    )
    runs = set(result["metrics"]["runs"])
    assert runs == {"majority", "fusion_causal", "fusion_geo_causal", "fusion_conf_causal"} | {
        f"view{v}_causal" for v in range(3)
    }
    assert (
        result["config"]["spec"]["lookahead"] == 3 and result["config"]["spec"]["offline"] is False
    )
    assert "t + 3" in result["config"]["runs"]["view0_causal"]
    assert len(read_predictions(tmp_path / "la" / "predictions_val.csv")) == N_FRAMES


def test_checkpoints_restore_the_selected_networks(tmp_path: Path) -> None:
    from sop_monitor.havid import CAMERA_OF_VIEW
    from sop_monitor.mstcn import load_checkpoint, predict_probs

    paths = build(tmp_path)
    checkpoints = tmp_path / "ckpt"
    result = run_havid_tas(
        paths["features"],
        paths["splits"],
        paths["temporal"],
        paths["official"],
        tmp_path / "run",
        HavidTASSpec(mstcn=SMALL, offline=False),
        device="cpu",
        checkpoint_dir=checkpoints,
    )
    meta = json.loads((checkpoints / "meta.json").read_text(encoding="utf-8"))
    assert meta["networks"] == ["view0_causal", "view1_causal", "view2_causal"]
    assert meta["classes"] == result["config"]["classes"]
    saved = np.load(checkpoints / "probs_val.npz")
    rows = read_predictions(tmp_path / "run" / "predictions_val.csv")
    inverse = np.array(meta["classes"]["index_to_class_id"])
    fused = saved["fusion_causal__S02A04I01"].astype(np.float32)
    assert [r.preds["fusion_causal"] for r in rows] == inverse[fused.argmax(axis=1)].tolist()
    for view in range(3):
        model, standardise, info = load_checkpoint(checkpoints / f"view{view}_causal.pt")
        assert info["causal"] is True
        assert (
            info["best_epoch"] == result["config"]["training"][f"view{view}_causal"]["best_epoch"]
        )
        cache = np.load(paths["features"] / f"S02A04I01{CAMERA_OF_VIEW[view]}.npz")
        probs = predict_probs(model, cache["features"], standardise, "cpu")
        np.testing.assert_allclose(
            probs, saved[f"view{view}_causal__S02A04I01"].astype(np.float32), atol=2e-3
        )


def test_online_head_streams_a_checkpoint_run(tmp_path: Path) -> None:
    from sop_monitor.havid import CAMERA_OF_VIEW
    from sop_monitor.online_head import cached_feature_chunks, run_online_head
    from sop_monitor.sop_graph import TaskGraph

    paths = build(tmp_path)
    checkpoints = {}
    for hand in ("lh", "rh"):
        checkpoints[hand] = tmp_path / f"ckpt_{hand}"
        run_havid_tas(
            paths["features"],
            paths["splits"],
            paths["temporal"],
            paths["official"],
            tmp_path / f"run_{hand}",
            HavidTASSpec(hand=hand, mstcn=SMALL, lookahead=2, offline=False),
            device="cpu",
            checkpoint_dir=checkpoints[hand],
        )
    rec = "S02A04I01"
    result = run_online_head(
        checkpoints,
        lambda r: cached_feature_chunks(
            {v: paths["features"] / f"{r}{CAMERA_OF_VIEW[v]}.npz" for v in range(3)}, 7
        ),
        [rec],
        {rec: "p"},
        ({"p": TaskGraph({}, {})}, {"p": {}}, {"p": []}),
        "cpu",
        granularity="pt",
    )
    entry = result["recordings"][rec]
    assert result["lookahead"] == {"lh": 2, "rh": 2}
    assert entry["agreement_with_checkpoint_run"] == {"lh": 1.0, "rh": 1.0}
    assert len(entry["labels"]["lh"]) == N_FRAMES
    rows = read_predictions(tmp_path / "run_lh" / "predictions_val.csv")
    labels = json.loads((checkpoints["lh"] / "meta.json").read_text(encoding="utf-8"))["classes"]
    expected = [labels["class_id_to_label"][str(r.preds["fusion_causal"])] for r in rows]
    assert entry["labels"]["lh"] == expected
    assert entry["latency"]["frames"] == N_FRAMES and entry["latency"]["wall_real_time_factor"] > 0
