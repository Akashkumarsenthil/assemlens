"""One lazily loaded Qwen worker on the Nano GPU."""

import json
import os
import threading
import time
from pathlib import Path

from scripts.jeep_state_eval import parse, prompt

ROOT = Path(__file__).resolve().parents[1]


class ReferenceModel:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("The Nano CUDA GPU is unavailable.")
        config = json.loads((ROOT / "runs/verifier_v2/config.json").read_text())
        processor = AutoProcessor.from_pretrained(
            config["model"], revision=config["model_revision"],
            min_pixels=64 * 32 * 32, max_pixels=256 * 32 * 32,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model"], revision=config["model_revision"],
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
        self._processor, self._model, self._torch = processor, model, torch

    def predict(self, step: dict, observation):
        from PIL import Image

        with Image.open(step["reference_path"]) as source:
            reference = source.convert("RGB")
        return self.compare(reference, observation, step["instruction"], step["criteria"])

    def compare(self, reference, observation, instruction: str, criteria: str):
        if os.environ.get("ASSEMLENS_ALLOW_GPU_INFERENCE") != "1":
            raise RuntimeError("Nano inference is paused while another GPU job runs.")
        with self._lock:
            if self._model is None:
                self._load()
            processor, model, torch = self._processor, self._model, self._torch
            goal = {"instruction": instruction, "visible_completion": criteria}
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"},
                {"type": "text", "text": prompt(goal)},
            ]}]
            formatted = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            started = time.monotonic()
            batch = processor(text=[formatted], images=[reference, observation],
                              return_tensors="pt").to("cuda")
            with torch.inference_mode():
                tokens = model.generate(**batch, max_new_tokens=192, do_sample=False)
            torch.cuda.synchronize()
            elapsed = round((time.monotonic() - started) * 1000)
            raw = processor.batch_decode(
                tokens[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            result = parse(raw)
            if result["assessment"] == "invalid_json":
                return {"assessment": "uncertain", "evidence": "The model returned an unreadable response.",
                        "nextAction": "Take another clear view.", "latencyMs": elapsed}
            return {"assessment": result["assessment"], "evidence": result["evidence"],
                    "nextAction": result["next_action"], "latencyMs": elapsed}
