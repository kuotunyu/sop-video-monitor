"""Streaming causal heads: chunked inference equals whole-recording inference, look-ahead alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sop_monitor.havid_tas import advance_outputs, fuse_views  # noqa: E402
from sop_monitor.mstcn import MSTCNSpec, Standardiser, build_model, predict_probs  # noqa: E402
from sop_monitor.online import OnlineSOPMonitor  # noqa: E402
from sop_monitor.online_head import (  # noqa: E402
    HandRecogniser,
    OnlineSOPPipeline,
    StreamingHead,
    cached_feature_chunks,
    latency_summary,
    run_pipeline,
    video_feature_chunks,
)
from sop_monitor.sop_graph import TaskGraph  # noqa: E402

DIM, N_CLASSES = 6, 4
TINY = MSTCNSpec(num_layers_pg=4, num_layers_r=3, num_refinement_stages=1, num_f_maps=8)
LABELS = {"10": "null", "11": "pckbx", "12": "icbck", "13": "ibscb"}
INVERSE = np.array([10, 11, 12, 13])


def _head(seed: int) -> StreamingHead:
    torch.manual_seed(seed)
    model = build_model(DIM, N_CLASSES, TINY, causal=True).eval()
    rng = np.random.default_rng(seed)
    standardise = Standardiser(
        rng.standard_normal(DIM).astype(np.float32), np.ones(DIM, np.float32)
    )
    return StreamingHead(model, standardise, "cpu")


def _features(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float16)


@pytest.mark.parametrize("chunk", [1, 7, 15, 200])
def test_streaming_head_equals_whole_recording(chunk: int) -> None:
    head = _head(0)
    features = _features(97)
    full = predict_probs(head.model, features, head.standardise, "cpu")
    streamed = np.concatenate(
        [head.push(features[i : i + chunk]) for i in range(0, len(features), chunk)]
    )
    np.testing.assert_allclose(streamed, full, atol=1e-5)
    assert head.n_seen == 97 and head.push(np.zeros((0, DIM))).size == 0


@pytest.mark.parametrize(("lookahead", "chunk"), [(0, 10), (5, 3), (5, 40), (120, 16)])
def test_hand_recogniser_matches_the_offline_fused_run(lookahead: int, chunk: int) -> None:
    features = {view: _features(90, seed=view) for view in range(3)}
    offline_heads = {view: _head(view) for view in range(3)}
    stack = np.stack(
        [
            predict_probs(offline_heads[v].model, features[v], offline_heads[v].standardise, "cpu")
            for v in range(3)
        ]
    )
    for rule in ("", "_geo", "_conf"):
        expected_ids = INVERSE[advance_outputs(fuse_views(stack)[rule], lookahead).argmax(axis=1)]
        expected = [LABELS[str(i)] for i in expected_ids]
        recogniser = HandRecogniser(
            {view: _head(view) for view in range(3)}, INVERSE, LABELS, lookahead, rule
        )
        got: list[str] = []
        for start in range(0, 90, chunk):
            got.extend(recogniser.push({v: features[v][start : start + chunk] for v in range(3)}))
            assert len(got) == max(0, min(90, start + chunk) - lookahead)
        got.extend(recogniser.finish())
        assert got == expected
    with pytest.raises(ValueError):
        HandRecogniser({}, INVERSE, LABELS, 0, "_max")


def test_pipeline_feeds_the_monitor_frame_by_frame(tmp_path: Path) -> None:
    graph = TaskGraph(
        dict.fromkeys(("pckbx", "icbck", "ibscb"), "PT"),
        {"precedesPT": (("pckbx", "icbck"),)},
    )
    features = {view: _features(60, seed=10 + view) for view in range(3)}

    def pipeline(lookahead: int) -> OnlineSOPPipeline:
        hands = {
            hand: HandRecogniser(
                {view: _head(view + offset) for view in range(3)}, INVERSE, LABELS, lookahead
            )
            for hand, offset in (("lh", 0), ("rh", 5))
        }
        return OnlineSOPPipeline(hands, OnlineSOPMonitor(graph, {}, ["pckbx"], "pt"))

    live = pipeline(4)
    found = run_pipeline(
        live, ({v: features[v][i : i + 8] for v in range(3)} for i in range(0, 60, 8))
    )
    assert all(len(live.labels[hand]) == 60 for hand in ("lh", "rh"))
    reference = OnlineSOPMonitor(graph, {}, ["pckbx"], "pt")
    expected = []
    for frame in range(60):
        expected.extend(reference.push(frame, {h: live.labels[h][frame] for h in ("lh", "rh")}))
    expected.extend(reference.finish())
    assert [d.to_dict() for d in found] == [d.to_dict() for d in expected]
    summary = latency_summary(live.timings, fps=15.0)
    assert summary["chunks"] == 8 and summary["frames"] == 60 and summary["real_time_factor"] > 0

    caches = {}
    for view in range(3):
        caches[view] = tmp_path / f"v{view}.npz"
        np.savez(caches[view], frames=np.arange(60), features=features[view], n_frames=np.int64(60))
    replayed = pipeline(4)
    run_pipeline(replayed, cached_feature_chunks(caches, 8))
    assert replayed.labels == live.labels
    np.savez(
        caches[1], frames=np.arange(0, 60, 2), features=features[1][:30], n_frames=np.int64(60)
    )
    with pytest.raises(ValueError, match="every frame"):
        list(cached_feature_chunks(caches, 8))


def test_video_feature_chunks_decode_views_in_lockstep(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from sop_monitor.video import write_synthetic_video

    videos = {}
    for view in range(3):
        videos[view] = tmp_path / f"v{view}.mp4"
        write_synthetic_video(videos[view], n_frames=23)
    timings: list[float] = []
    chunks = list(
        video_feature_chunks(
            videos,
            10,
            lambda frames: frames.reshape(len(frames), -1)[:, :4].astype(np.float32),
            size=(32, 24),
            decode_timings=timings,
        )
    )
    assert [len(c[0]) for c in chunks] == [10, 10, 3]
    assert all(set(c) == {0, 1, 2} for c in chunks) and len(timings) == 3
    write_synthetic_video(videos[2], n_frames=20)
    with pytest.raises(ValueError, match="differ in length"):
        list(video_feature_chunks(videos, 10, lambda f: f.reshape(len(f), -1), size=(32, 24)))
