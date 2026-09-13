"""Deviation queue and local human-review server (spec W5, first slice).

The SOP checks of a committed ``havid-sop`` run directory become a *deviation queue*: one item per
finding (order violation, omitted mandatory step, step outside its duration window, step the plate's
graph never saw), with the time window, the evidence and the three camera videos of the recording.
A reviewer accepts or rejects items in a browser; decisions are appended to a JSONL file (the last
decision per item wins), so the queue and the decisions stay separate, append-only and diffable.

Everything is standard library: the server is :class:`http.server.ThreadingHTTPServer` bound to
``127.0.0.1``, serves the videos with HTTP range requests (browsers need them to seek), and only
serves video ids listed in the split file it was started with — the frozen test videos are never
served. Deviations are suggestions for human review, never safety or compliance decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sop_monitor.havid import CAMERA_OF_VIEW, VERBS, VIEW_NAMES
from sop_monitor.havid_sop import FPS, check_sequence, load_knowledge, read_steps

KIND_PRIORITY = {"omission": 0, "order": 1, "duration": 2, "unknown": 3}
DECISIONS = ("accept", "reject", "skip")

OBJECT_NAMES = {
    "ba": "ball",
    "bs": "ball seat",
    "bx": "box",
    "c1": "cylinder plate hole 1",
    "c2": "cylinder plate hole 2",
    "c3": "cylinder plate hole 3",
    "c4": "cylinder plate hole 4",
    "c": "cylinder plate hole",
    "cb": "cylinder base",
    "cc": "cylinder cap",
    "ck": "cylinder bracket",
    "cs": "cylinder subassembly",
    "dh": "hex screwdriver",
    "dp": "philips screwdriver",
    "ft": "gear shaft",
    "g1": "gear plate hole 1",
    "g2": "gear plate hole 2",
    "g3": "gear plate hole 3",
    "g": "gear plate hole",
    "gl": "large gear",
    "gs": "small gear",
    "gw": "worm gear",
    "hd": "dial",
    "hq": "quarter-turn handle",
    "hw": "hand-wheel",
    "ib": "bar",
    "ir": "rod",
    "lb": "linear bearing",
    "n1": "general plate hole 1",
    "n2": "general plate hole 2",
    "n3": "general plate hole 3",
    "n4": "general plate hole 4",
    "n5": "general plate stud",
    "n6": "general plate usb female",
    "n": "general plate hole",
    "nt": "nut",
    "pl": "large spacer",
    "ps": "small spacer",
    "sb": "bolt",
    "sh": "hex screw",
    "sp": "philips screw",
    "us": "usb male",
    "wn": "nut wrench",
    "ws": "shaft wrench",
}
"""HR-SAT object and tool codes (paper Figure 9); one-letter entries are sheet-level hole groups."""


def describe_step(label: str) -> str:
    """Human-readable reading of an HR-SAT or sheet-level step label (``null``/``w`` included)."""
    if label == "null":
        return "pause"
    if label == "w":
        return "wrong (annotated error)"
    verb = VERBS.get(label[:1])
    rest = label[1:]
    codes: list[str] = []
    while rest:
        two = rest[:2]
        if two in OBJECT_NAMES and len(rest) != 1:
            codes.append(two)
            rest = rest[2:]
        elif rest[:1] in OBJECT_NAMES:
            codes.append(rest[:1])
            rest = rest[1:]
        else:
            return label
    if verb is None or not codes:
        return label
    words = [verb, OBJECT_NAMES[codes[0]]]
    if len(codes) > 1:
        words += ["→", OBJECT_NAMES[codes[1]]]
    if len(codes) > 2:
        words += ["with", OBJECT_NAMES[codes[2]]]
    return " ".join(words)


@dataclass(frozen=True)
class QueueItem:
    id: str
    run: str
    source: str
    recording: str
    subject: str
    plate: str
    kind: str
    step: str
    step_description: str
    start_frame: int | None
    end_frame: int | None
    evidence: dict[str, object]
    videos: dict[str, str]

    @property
    def priority(self) -> tuple[int, str, int]:
        return (KIND_PRIORITY[self.kind], self.recording, self.start_frame or -1)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "run": self.run,
            "source": self.source,
            "recording": self.recording,
            "subject": self.subject,
            "plate": self.plate,
            "kind": self.kind,
            "step": self.step,
            "step_description": self.step_description,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": None if self.start_frame is None else self.start_frame / FPS,
            "end_s": None if self.end_frame is None else (self.end_frame + 1) / FPS,
            "evidence": self.evidence,
            "videos": self.videos,
            "second_opinion": None,
        }


def _item_id(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def build_queue(run_dir: Path, source: str = "pred") -> list[QueueItem]:
    """Deviation queue of one source (``pred`` or ``gt``) of a ``havid-sop`` run directory."""
    config = json.loads((run_dir / "sop_checks.json").read_text(encoding="utf-8"))["config"]
    graphs, bounds, mandatory = load_knowledge(Path(config["graphs_dir"]))
    sequences, plates = read_steps(run_dir / "steps_val.csv")
    if source not in sequences:
        raise ValueError(f"{run_dir}: no {source!r} sequences (have {sorted(sequences)})")
    items: list[QueueItem] = []
    for recording in sorted(sequences[source]):
        steps = sequences[source][recording]
        plate = plates[recording]
        findings = check_sequence(steps, graphs[plate], bounds[plate], mandatory[plate])
        videos = {VIEW_NAMES[v]: f"{recording}{camera}" for v, camera in CAMERA_OF_VIEW.items()}
        common = {
            "run": run_dir.name,
            "source": source,
            "recording": recording,
            "subject": recording[:3],
            "plate": plate,
            "videos": videos,
        }

        def add(
            kind: str,
            label: str,
            start: int | None,
            end: int | None,
            evidence: dict,
            common: dict = common,
        ) -> None:
            items.append(
                QueueItem(
                    id=_item_id(
                        common["run"],
                        common["source"],
                        common["recording"],
                        kind,
                        label,
                        start,
                        end,
                    ),
                    kind=kind,
                    step=label,
                    step_description=describe_step(label),
                    start_frame=start,
                    end_frame=end,
                    evidence=evidence,
                    **common,  # type: ignore[arg-type]
                )
            )

        for violation in findings["violations"]:  # type: ignore[union-attr]
            step = steps[int(violation["index"])]
            missing = list(violation["missing_predecessors"])
            add(
                "order",
                step.label,
                step.start,
                step.end,
                {
                    "rule": "step started before all of its learned predecessors were observed",
                    "missing_predecessors": [
                        {"step": m, "description": describe_step(m)} for m in missing
                    ],
                },
            )
        for label in findings["omissions"]:  # type: ignore[union-attr]
            add(
                "omission",
                label,
                None,
                None,
                {"rule": f"mandatory step of the {plate} plate never observed in the recording"},
            )
        for finding in findings["durations"]:  # type: ignore[union-attr]
            step = steps[int(finding["index"])]
            window = bounds[plate][step.label]
            add(
                "duration",
                step.label,
                step.start,
                step.end,
                {
                    "rule": "step duration outside the learned window",
                    "kind": finding["kind"],
                    "duration_s": round(float(finding["duration"]), 2),
                    "window_s": [round(window.low, 2), round(window.high, 2)],
                },
            )
        unknown_seen: set[str] = set()
        for index, step in enumerate(steps):
            if step.label in findings["unknown"] and step.label not in unknown_seen:  # type: ignore[operator]
                unknown_seen.add(step.label)
                add(
                    "unknown",
                    step.label,
                    step.start,
                    step.end,
                    {
                        "rule": f"step never seen in the {plate} plate's train recordings",
                        "first_index": index,
                    },
                )
    return sorted(items, key=lambda item: item.priority)


def write_queue(path: Path, items: Sequence[QueueItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class DecisionStore:
    """Append-only JSONL of review decisions; the latest decision per item id wins."""

    def __init__(self, path: Path, item_ids: Iterable[str]) -> None:
        self.path = path
        self.item_ids = set(item_ids)
        self._lock = threading.Lock()

    def record(self, item_id: str, decision: str, note: str = "", reviewer: str = "") -> dict:
        if item_id not in self.item_ids:
            raise KeyError(f"unknown queue item {item_id!r}")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        entry = {
            "id": item_id,
            "decision": decision,
            "note": note[:2000],
            "reviewer": reviewer[:100],
            "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry

    def latest(self) -> dict[str, dict[str, object]]:
        with self._lock:
            entries = read_jsonl(self.path)
        return {str(e["id"]): e for e in entries if e.get("id") in self.item_ids}


def review_summary(
    items: Sequence[Mapping[str, object]], decisions: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Per kind: queued, reviewed, accepted, rejected, skipped, and the reviewed precision."""
    per_kind: dict[str, Counter[str]] = {}
    for item in items:
        kind = str(item["kind"])
        counts = per_kind.setdefault(kind, Counter())
        counts["queued"] += 1
        decision = decisions.get(str(item["id"]), {}).get("decision")
        if decision:
            counts[str(decision)] += 1
    out: dict[str, object] = {}
    for kind in sorted(per_kind, key=lambda k: KIND_PRIORITY.get(k, 9)):
        c = per_kind[kind]
        judged = c["accept"] + c["reject"]
        out[kind] = {
            "queued": c["queued"],
            "accepted": c["accept"],
            "rejected": c["reject"],
            "skipped": c["skip"],
            "precision_reviewed": c["accept"] / judged if judged else None,
        }
    return out


# ---- server -------------------------------------------------------------------------------------

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """``(start, end)`` inclusive for a single ``bytes=`` range, ``None`` for no/invalid header."""
    if not header:
        return None
    match = _RANGE.match(header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        return None
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        length = int(match.group(2))
        start, end = max(0, size - length), size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def make_handler(
    items: Sequence[Mapping[str, object]],
    store: DecisionStore,
    videos: Mapping[str, Path],
    page: str,
    reviewer: str,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "sop-monitor-review"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = page.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/items":
                latest = store.latest()
                self._json(
                    {
                        "items": [
                            {**item, "decision": latest.get(str(item["id"]))} for item in items
                        ],
                        "summary": review_summary(items, latest),
                        "reviewer": reviewer,
                    }
                )
            elif path.startswith("/video/") and path.endswith(".mp4"):
                self._video(path[len("/video/") : -len(".mp4")])
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/decisions":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 10_000:
                self._json({"error": "payload too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                entry = store.record(
                    str(payload["id"]),
                    str(payload["decision"]),
                    str(payload.get("note", "")),
                    reviewer,
                )
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"decision": entry, "summary": review_summary(items, store.latest())})

        def _video(self, video_id: str) -> None:
            path = videos.get(video_id)
            if path is None:
                self._json({"error": "video not in the review split"}, HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            span = parse_range(self.headers.get("Range"), size)
            start, end = span if span else (0, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if span:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

    return Handler


def allowed_videos(split_csv: Path, rgb_dir: Path) -> dict[str, Path]:
    """Video id → mp4 path for the videos of ``split_csv`` found under ``rgb_dir``."""
    from sop_monitor.havid_tas import read_split

    wanted = {row["video_id"] for row in read_split(split_csv)}
    return {p.stem: p for p in rgb_dir.rglob("*.mp4") if p.stem in wanted}


def make_server(
    queue_path: Path,
    decisions_path: Path,
    split_csv: Path,
    rgb_dir: Path,
    port: int = 8765,
    reviewer: str = "",
) -> ThreadingHTTPServer:
    from sop_monitor.review_page import PAGE

    if "test" in split_csv.stem:
        raise ValueError("the review server never serves the frozen test split")
    items = read_jsonl(queue_path)
    store = DecisionStore(decisions_path, (str(item["id"]) for item in items))
    videos = allowed_videos(split_csv, rgb_dir)
    handler = make_handler(items, store, videos, PAGE, reviewer)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
