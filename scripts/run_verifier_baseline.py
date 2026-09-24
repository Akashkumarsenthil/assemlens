#!/usr/bin/env python3
"""Zero-shot Qwen-VL baseline for the four-class verification task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from assemlens.verifier import ASSESSMENTS, confusion_matrix, normalize_label

PROMPT = """You are an assembly task verifier, not an action recognizer.
Compare the expected step, observed action, chronological evidence, and prior context.
Choose exactly one assessment: correct, mistake, correction, uncertain.
Use uncertain when the camera evidence cannot establish the resulting state.
Return only JSON with keys assessment, confidence, visible_evidence, reason,
corrective_instruction, step_verified, needs_another_view.
Do not treat recognizing an action as proof that it was correct."""


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_for(row):
    context = {
        "expected_step": row["expected_step"],
        "observed_action": row["observed_action"],
        "previous_actions": row["previous_actions"],
        "previous_verified_state": row["previous_verified_state"],
    }
    return PROMPT + "\nContext:\n" + json.dumps(context, ensure_ascii=False)


def load_images(row):
    paths = row["before_frames"] + row["action_frames"] + row["after_frames"]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-revision")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = read_rows(Path(args.examples))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("No examples")
    kwargs = {"min_pixels": 64 * 32 * 32, "max_pixels": 196 * 32 * 32}
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    processor = AutoProcessor.from_pretrained(args.model, **kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **({"revision": args.model_revision} if args.model_revision else {})
    ).to("cuda")
    model.eval()
    results = []
    skipped = []
    for index, row in enumerate(rows, 1):
        try:
            images = load_images(row)
        except FileNotFoundError as exc:
            skipped.append({"example_id": row["example_id"], "reason": str(exc)})
            print(f"{index}/{len(rows)} SKIP missing frame: {exc}", flush=True)
            continue
        messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt_for(row)}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
        started = time.monotonic()
        with torch.inference_mode():
            tokens = model.generate(**batch, max_new_tokens=160, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        raw = processor.batch_decode(tokens[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        try:
            prediction = json.loads(raw)
            assessment = normalize_label(prediction.get("assessment", "uncertain"))
        except Exception:
            prediction = {"assessment": "uncertain", "parse_error": True}
            assessment = "uncertain"
        results.append({"example_id": row["example_id"], "ground_truth_assessment": row["ground_truth_assessment"],
                        "prediction": assessment, "raw": raw, "seconds": time.monotonic() - started})
        print(f"{index}/{len(rows)} {assessment} gold={row['ground_truth_assessment']}", flush=True)

    correct = sum(x["prediction"] == x["ground_truth_assessment"] for x in results)
    if not results:
        raise RuntimeError("No examples could be evaluated; every example has missing frames")
    metrics = {"n_requested": len(rows), "n_evaluated": len(results), "n_skipped": len(skipped),
               "accuracy": correct / len(results), "confusion_matrix": confusion_matrix(results)}
    for label in ASSESSMENTS:
        tp = sum(x["prediction"] == label and x["ground_truth_assessment"] == label for x in results)
        fp = sum(x["prediction"] == label and x["ground_truth_assessment"] != label for x in results)
        fn = sum(x["prediction"] != label and x["ground_truth_assessment"] == label for x in results)
        metrics[label] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                          "recall": tp / (tp + fn) if tp + fn else 0.0}
    output = {"metrics": metrics, "predictions": results, "skipped": skipped,
              "scope": "zero-shot VLM verification baseline"}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
