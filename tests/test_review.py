"""Deviation queue, decision store, range parsing and the local review server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from sop_monitor.havid_sop import (
    Step,
    learn_duration_bounds,
    learn_plate_graphs,
    mandatory_steps,
    write_knowledge,
    write_steps,
)
from sop_monitor.review import (
    DecisionStore,
    build_queue,
    describe_step,
    make_server,
    parse_range,
    read_jsonl,
    review_summary,
    write_queue,
)

ORDER = ["pckbx", "icbck", "ibscb", "sshc1"]
PRIORITY = {"omission": 0, "order": 1, "duration": 2, "unknown": 3}


def _seq(labels: list[str], length: int = 30) -> list[Step]:
    return [Step(label, i * length, (i + 1) * length - 1, "lh") for i, label in enumerate(labels)]


def _sop_run(tmp_path: Path) -> Path:
    train = {f"S{n:02d}A04I01": _seq(ORDER) for n in range(1, 7)}
    plates = {rec: "cylinder" for rec in train}
    graphs_dir = tmp_path / "knowledge"
    write_knowledge(
        graphs_dir,
        learn_plate_graphs(train, plates, min_support=3),
        learn_duration_bounds(train, plates, min_count=5),
        mandatory_steps(train, plates),
        {"granularity": "pt"},
    )
    run = tmp_path / "sop_run"
    run.mkdir()
    s08 = [
        Step("pckbx", 0, 299, "lh"),
        Step("icbck", 300, 329, "lh"),
        Step("sshc1", 330, 359, "lh"),
    ]
    s08.append(Step("pbx", 360, 369, "rh"))  # never seen in train, ibscb omitted, pckbx too long
    val = {"S07A04I01": _seq(["icbck", "pckbx", "ibscb", "sshc1"]), "S08A04I01": s08}
    write_steps(run / "steps_val.csv", {"gt": val, "pred": val}, {r: "cylinder" for r in val})
    (run / "wrong_val.csv").write_text("recording,source,hand,start,end\n", encoding="utf-8")
    (run / "sop_checks.json").write_text(
        json.dumps({"config": {"graphs_dir": graphs_dir.as_posix(), "seed": 0}}), encoding="utf-8"
    )
    return run


def test_describe_step_reads_hrsat_and_sheet_labels() -> None:
    assert (
        describe_step("sshc1dh") == "screw hex screw → cylinder plate hole 1 with hex screwdriver"
    )
    assert describe_step("sshc") == "screw hex screw → cylinder plate hole"
    assert describe_step("rgw") == "rotate worm gear"
    assert describe_step("null") == "pause" and describe_step("zzz") == "zzz"


def test_queue_items_are_deterministic_and_prioritised(tmp_path: Path) -> None:
    run = _sop_run(tmp_path)
    items = build_queue(run, "pred")
    kinds = [i.kind for i in items]
    assert kinds == sorted(kinds, key=PRIORITY.__getitem__)
    assert set(kinds) == {"order", "omission", "duration", "unknown"}
    order = next(i for i in items if i.kind == "order")
    assert order.recording == "S07A04I01" and order.start_frame == 0
    assert order.evidence["missing_predecessors"][0]["step"] == "pckbx"
    omission = next(i for i in items if i.kind == "omission")
    assert omission.step == "ibscb" and omission.start_frame is None
    duration = next(i for i in items if i.kind == "duration")
    assert duration.evidence["kind"] == "too_long"
    assert duration.to_dict()["end_s"] == pytest.approx(20.0)
    assert order.videos == {"side": "S07A04I01M0", "front": "S07A04I01S1", "top": "S07A04I01S2"}
    assert [i.id for i in build_queue(run, "pred")] == [i.id for i in items]
    with pytest.raises(ValueError):
        build_queue(run, "other")
    path = tmp_path / "queue.jsonl"
    write_queue(path, items)
    assert [r["id"] for r in read_jsonl(path)] == [i.id for i in items]


def test_decision_store_latest_wins_and_summary(tmp_path: Path) -> None:
    items = [
        {"id": "a", "kind": "order"},
        {"id": "b", "kind": "order"},
        {"id": "c", "kind": "omission"},
    ]
    store = DecisionStore(tmp_path / "d.jsonl", ["a", "b", "c"])
    store.record("a", "reject", "false alarm")
    store.record("a", "accept", "on second look")
    store.record("b", "reject")
    store.record("c", "skip")
    with pytest.raises(KeyError):
        store.record("x", "accept")
    with pytest.raises(ValueError):
        store.record("a", "maybe")
    latest = store.latest()
    assert latest["a"]["decision"] == "accept" and len(read_jsonl(tmp_path / "d.jsonl")) == 4
    summary = review_summary(items, latest)
    assert summary["order"] == {
        "queued": 2,
        "accepted": 1,
        "rejected": 1,
        "skipped": 0,
        "precision_reviewed": 0.5,
    }
    assert summary["omission"]["precision_reviewed"] is None


def test_parse_range() -> None:
    assert parse_range(None, 100) is None and parse_range("bytes=abc", 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-500", 100) == (50, 99)
    assert parse_range("bytes=200-", 100) is None


def test_server_serves_items_decisions_and_only_split_videos(tmp_path: Path) -> None:
    run = _sop_run(tmp_path)
    queue = tmp_path / "queue.jsonl"
    write_queue(queue, build_queue(run, "pred"))
    rgb = tmp_path / "rgb" / "s07"
    rgb.mkdir(parents=True)
    (rgb / "S07A04I01M0.mp4").write_bytes(bytes(range(256)) * 4)
    (rgb / "S09A04I01M0.mp4").write_bytes(b"not in the split")
    split = tmp_path / "val.csv"
    split.write_text(
        "video_id,subject,recording,view\nS07A04I01M0,S07,S07A04I01,0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="test"):
        make_server(queue, tmp_path / "d.jsonl", tmp_path / "test.csv", tmp_path / "rgb", port=0)
    server = make_server(
        queue, tmp_path / "d.jsonl", split, tmp_path / "rgb", port=0, reviewer="r1"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert "SOP deviation review" in page
        data = json.loads(urllib.request.urlopen(base + "/api/items").read())
        assert len(data["items"]) == len(read_jsonl(queue)) and data["reviewer"] == "r1"
        item_id = data["items"][0]["id"]
        request = urllib.request.Request(
            base + "/api/decisions",
            data=json.dumps({"id": item_id, "decision": "accept", "note": "seen"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        answer = json.loads(urllib.request.urlopen(request).read())
        assert answer["decision"]["reviewer"] == "r1" and answer["decision"]["decision"] == "accept"
        bad = urllib.request.Request(
            base + "/api/decisions", data=b'{"id": "nope", "decision": "accept"}', method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(bad)
        assert caught.value.code == 400
        ranged = urllib.request.Request(
            base + "/video/S07A04I01M0.mp4", headers={"Range": "bytes=10-19"}
        )
        response = urllib.request.urlopen(ranged)
        assert response.status == 206 and response.read() == bytes(range(10, 20))
        assert response.headers["Content-Range"] == "bytes 10-19/1024"
        for blocked in ("S09A04I01M0", "..%2F..%2Fsecret"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + f"/video/{blocked}.mp4")
            assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
    assert read_jsonl(tmp_path / "d.jsonl")[0]["note"] == "seen"
