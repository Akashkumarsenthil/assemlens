#!/usr/bin/env python3
"""Add independent Assembly101 fine-action labels to verifier examples.

This does not use the teammate's model. It joins the official fine-grained
action annotations to the existing mistake examples by sequence, view, and
maximum temporal overlap. The result is an independent verifier experiment;
it must not be described as end-to-end action-model evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def sequence_from_video(value: str) -> str:
    parts = Path(str(value)).parts
    for index, part in enumerate(parts):
        if part == "recordings" and index + 1 < len(parts):
            return parts[index + 1]
    return parts[-2] if len(parts) >= 2 else Path(str(value)).stem


def view_from_video(value: str) -> str:
    return Path(str(value)).stem


def load_fine_rows(paths: list[Path]):
    index = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sequence = sequence_from_video(row["video"])
                view = view_from_video(row["video"])
                index.setdefault((sequence, view), []).append(row)
    return index


def enrich(args):
    examples = [json.loads(line) for line in Path(args.examples).read_text().splitlines() if line.strip()]
    fine = load_fine_rows([Path(x) for x in args.fine_csv])
    output = []
    missing = []
    for row in examples:
        source = row.get("source", {})
        sequence = row["sequence_id"]
        view = row.get("view_id", "")
        candidates = fine.get((sequence, view), [])
        start = int(source.get("start", 0))
        end = int(source.get("end", start))

        def overlap(candidate):
            left = max(start, int(candidate["start_frame"]))
            right = min(end, int(candidate["end_frame"]))
            return max(0, right - left)

        candidates = [candidate for candidate in candidates if overlap(candidate) > 0]
        if not candidates:
            missing.append(row["example_id"])
            output.append(row)
            continue
        chosen = max(candidates, key=overlap)
        row = dict(row)
        row["observed_action"] = {
            "action": chosen["action_cls"],
            "verb": chosen["verb_cls"],
            "object": chosen["noun_cls"],
            "source": "assembly101_fine_grained_annotation",
            "annotation_video": chosen["video"],
            "overlap_frames": overlap(chosen),
        }
        output.append(row)

    target = Path(args.output)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    report = {"n": len(output), "matched": len(output) - len(missing), "missing": len(missing), "missing_ids": missing,
              "source": "Assembly101 fine-grained annotations; not teammate model predictions"}
    target.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--fine-csv", required=True, action="append")
    parser.add_argument("--output", required=True)
    enrich(parser.parse_args())


if __name__ == "__main__":
    main()
