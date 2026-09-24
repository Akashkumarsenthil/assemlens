"""Shared schema and small utilities for the AssemLens verification stage.

This module deliberately does not perform action recognition.  The observed
action is an input produced by the teammate's model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

ASSESSMENTS = ("correct", "mistake", "correction", "uncertain")


@dataclass
class ExpectedStep:
    """Procedure context supplied by a product package or benchmark adapter."""

    step_id: str
    instruction: str
    verb: str = ""
    objects: list[str] = field(default_factory=list)
    source: str = "assembly101_annotation"


@dataclass
class VerifiedState:
    """State known before the current action; never inferred from the label alone."""

    completed_steps: list[str] = field(default_factory=list)
    unresolved_mistakes: list[str] = field(default_factory=list)
    state_summary: str = ""


@dataclass
class VerificationExample:
    example_id: str
    sequence_id: str
    video_id: str
    view_id: str
    expected_step: dict[str, Any]
    previous_actions: list[dict[str, Any]]
    previous_verified_state: dict[str, Any]
    before_frames: list[str]
    action_frames: list[str]
    after_frames: list[str]
    observed_action: dict[str, Any]
    ground_truth_assessment: str
    mistake_type: str | None
    remark: str
    source: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(label: str) -> str:
    value = str(label).strip().lower().replace(" ", "_")
    aliases = {"ok": "correct", "error": "mistake", "fixed": "correction"}
    value = aliases.get(value, value)
    if value not in ASSESSMENTS:
        raise ValueError(f"Unsupported assessment: {label!r}")
    return value


def action_text(action: dict[str, Any]) -> str:
    """Render a teammate action record without claiming it is verified."""

    if action.get("action"):
        return str(action["action"])
    verb = str(action.get("verb", "")).strip()
    obj = str(action.get("object", action.get("noun", ""))).strip()
    return " ".join(x for x in (verb, obj) if x)


def validate_example(row: dict[str, Any]) -> None:
    required = {
        "example_id", "sequence_id", "video_id", "expected_step",
        "previous_actions", "previous_verified_state", "before_frames",
        "action_frames", "after_frames", "observed_action",
        "ground_truth_assessment",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    if normalize_label(row["ground_truth_assessment"]) != row["ground_truth_assessment"]:
        raise ValueError("ground_truth_assessment must be normalized")
    if not isinstance(row["expected_step"], dict):
        raise ValueError("expected_step must be an object")
    if not isinstance(row["previous_actions"], list):
        raise ValueError("previous_actions must be a list")


def confusion_matrix(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {actual: {pred: 0 for pred in ASSESSMENTS} for actual in ASSESSMENTS}
    for row in rows:
        actual = normalize_label(row["ground_truth_assessment"])
        pred = normalize_label(row.get("prediction", "uncertain"))
        matrix[actual][pred] += 1
    return matrix
