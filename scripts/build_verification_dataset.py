#!/usr/bin/env python3
"""Build a small Assembly101 verification subset.

The script consumes the public mistake annotations and local Assembly101
videos. It never calls the teammate's action recognizer: observed_action is a
placeholder derived from the annotated verb/objects and can be replaced by
the recognizer's JSON output later.

Example on the Nano:
  python scripts/build_verification_dataset.py \
    --video-root /data/assembly101/videos \
    --output /data/assemlens/verification_100 \
    --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/assembly-101/assembly101-mistake-detection"
API_URL = "https://api.github.com/repos/assembly-101/assembly101-mistake-detection"
RAW_URL = REPO_URL + "/raw/main/annots/"
LABELS = ("correct", "mistake", "correction")
_VIDEO_INDEX: dict[str, dict[str, list[Path]]] = {}


def fetch_annotation_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    index_url = API_URL + "/git/trees/main?recursive=1"
    with urllib.request.urlopen(index_url) as response:
        tree = json.load(response)["tree"]
    files = [item["path"] for item in tree if item["path"].startswith("annots/") and item["path"].endswith(".csv")]
    for relative in files:
        target = path / Path(relative).name
        if not target.exists():
            with urllib.request.urlopen(RAW_URL + target.name) as response:
                target.write_bytes(response.read())
    return path


def parse_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, raw in enumerate(csv.reader(handle)):
            if not raw or not any(cell.strip() for cell in raw):
                continue
            if raw[0].strip().lower() == "start":
                continue
            if len(raw) < 6:
                raise ValueError(f"{path}: expected at least 6 columns, got {raw}")
            start, end, verb, this, that, label = [cell.strip() for cell in raw[:6]]
            remark = raw[6].strip() if len(raw) > 6 else ""
            label = label.lower().replace(" ", "_")
            if label not in LABELS:
                raise ValueError(f"{path}: unsupported label {label!r}")
            rows.append({"index": index, "start": int(start), "end": int(end), "verb": verb,
                         "this": this, "that": that, "label": label, "remark": remark})
    return rows


def sequence_from_filename(path: Path) -> str:
    # Filename is the exact sequence name, without the .csv suffix.
    return path.stem


def video_index(video_root: Path) -> dict[str, list[Path]]:
    key = str(video_root.resolve())
    if key not in _VIDEO_INDEX:
        index: dict[str, list[Path]] = defaultdict(list)
        print(f"Indexing videos under {video_root} ...", flush=True)
        for path in video_root.rglob("*.mp4"):
            if path.is_file():
                index[path.parent.name].append(path)
        _VIDEO_INDEX[key] = index
        print(f"Indexed {sum(len(v) for v in index.values())} videos in {len(index)} sequences", flush=True)
    return _VIDEO_INDEX[key]


def find_video(video_root: Path, sequence: str, view: str | None) -> Path | None:
    index = video_index(video_root)
    candidates = index.get(sequence, []) + index.get(f"assembly_{sequence}", [])
    if view:
        candidates = [path for path in candidates if path.name == view]
    else:
        candidates = [path for path in candidates if path.name.endswith("_rgb.mp4")]
    unique = []
    for item in candidates:
        if item.is_file() and item not in unique:
            unique.append(item)
    return unique[0] if unique else None


def frame(video: Path, seconds: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, seconds):.4f}",
               "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(target)]
    subprocess.run(command, check=True)
    if not target.is_file():
        raise FileNotFoundError(f"FFmpeg produced no frame at {target} (timestamp {seconds:.3f}s)")


def sample_times(start: int, end: int, unit: str) -> tuple[list[float], list[float], list[float]]:
    scale = 30.0 if unit == "frames" else 1000.0
    begin, finish = start / scale, end / scale
    duration = max(0.03, finish - begin)
    before = [max(0.0, begin - max(0.25, duration * 0.5)), max(0.0, begin - 0.05)]
    action = [begin + duration * 0.25, begin + duration * 0.75]
    after = [finish + 0.05, finish + max(0.25, duration * 0.5)]
    return before, action, after


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else Path(args.output) / "mistake_annotations"
    if args.download_annotations:
        fetch_annotation_dir(annotation_dir)
    files = sorted(annotation_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No mistake CSV files found in {annotation_dir}")

    parsed = [(sequence_from_filename(path), path, parse_csv(path)) for path in files]
    # Filter at sequence level before balancing. This matters when the user
    # downloaded only a bounded HF video subset: do not spend the requested
    # example budget on annotation files whose videos are not local.
    available = []
    for sequence, path, rows in parsed:
        if args.sequence and sequence not in args.sequence:
            continue
        if find_video(Path(args.video_root), sequence, args.view) is not None:
            available.append((sequence, path, rows))
    # Select complete sequences, then balance labels as much as possible.
    selected = []
    counts = Counter()
    for sequence, path, rows in available:
        for row in rows:
            if len(selected) >= args.limit:
                break
            if counts[row["label"]] >= max(1, args.limit // 4) and len(selected) < args.limit - 3:
                continue
            selected.append((sequence, path, rows, row))
            counts[row["label"]] += 1
        if len(selected) >= args.limit:
            break
    if not selected:
        raise RuntimeError("No annotations selected")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = []
    for number, (sequence, annotation_path, rows, row) in enumerate(selected):
        video = find_video(Path(args.video_root), sequence, args.view)
        if video is None:
            skipped.append({"sequence": sequence, "annotation": str(annotation_path), "reason": "video_not_found"})
            continue
        before_t, action_t, after_t = sample_times(row["start"], row["end"], args.time_unit)
        view_id = video.stem
        root = output / "frames" / f"{number:05d}"
        frame_groups = {"before": before_t, "action": action_t, "after": after_t}
        paths = {}
        try:
            for group, times in frame_groups.items():
                paths[group] = []
                for index, seconds in enumerate(times):
                    target = root / group / f"{index:02d}.jpg"
                    frame(video, seconds, target)
                    paths[group].append(str(target.resolve()))
        except (subprocess.CalledProcessError, OSError) as exc:
            skipped.append({"sequence": sequence, "annotation": str(annotation_path), "reason": f"frame_extract:{exc}"})
            continue
        print(f"Built {len(records) + 1}/{args.limit}: {sequence}:{rows.index(row)} {row['label']}", flush=True)

        position = rows.index(row)
        previous = rows[max(0, position - args.history):position]
        previous_actions = [{"verb": x["verb"], "objects": [x["this"], x["that"]],
                             "assessment": x["label"], "remark": x["remark"]} for x in previous]
        expected = {"step_id": f"{sequence}:{position}",
                    "instruction": f"{row['verb'].capitalize()} {row['this']} with {row['that']}",
                    "verb": row["verb"], "objects": [row["this"], row["that"]],
                    "source": "Assembly101 mistake annotation; ordering is sequence-contextual"}
        state = {"completed_steps": [f"{sequence}:{i}" for i, x in enumerate(previous) if x["label"] == "correct"],
                 "unresolved_mistakes": [f"{sequence}:{i}" for i, x in enumerate(previous) if x["label"] == "mistake"],
                 "state_summary": "Derived benchmark context; not a physical-state tracker"}
        record = {"example_id": f"{sequence}:{position}:{view_id}", "sequence_id": sequence,
                  "video_id": str(video.resolve()), "view_id": view_id, "expected_step": expected,
                  "previous_actions": previous_actions, "previous_verified_state": state,
                  "before_frames": paths["before"], "action_frames": paths["action"],
                  "after_frames": paths["after"],
                  "observed_action": {"verb": row["verb"], "object": row["this"],
                                      "related_object": row["that"], "source": "annotation_placeholder"},
                  "ground_truth_assessment": row["label"],
                  "mistake_type": row["remark"] or ("unknown" if row["label"] == "mistake" else None),
                  "remark": row["remark"],
                  "source": {"annotation_file": str(annotation_path.resolve()), "row_index": row["index"],
                             "start": row["start"], "end": row["end"], "time_unit": args.time_unit,
                             "fps": 30}}
        records.append(record)

    if not records:
        raise RuntimeError("No examples were built; check --video-root and --time-unit")
    with (output / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {"n": len(records), "requested": args.limit, "label_counts": dict(Counter(r["ground_truth_assessment"] for r in records)),
                "skipped": skipped, "time_unit": args.time_unit, "source": REPO_URL}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotation-dir")
    parser.add_argument("--download-annotations", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--view", help="Optional exact filename such as C10404_rgb.mp4")
    parser.add_argument("--time-unit", choices=("frames", "milliseconds"), default="frames")
    parser.add_argument("--sequence", action="append")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
