"""Load only reviewed product packages with local reference images."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class ProductPackage:
    id: str
    name: str
    description: str
    steps: tuple[dict, ...]


def load_product(root: Path, product_id: str) -> ProductPackage | None:
    if not IDENTIFIER.fullmatch(product_id):
        return None
    source = root / f"{product_id}.json"
    if not source.is_file():
        return None
    try:
        data = json.loads(source.read_text())
        if data.get("id") != product_id or data.get("approved") is not True:
            return None
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            return None
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        package_steps = []
        seen = set()
        for step in steps:
            if not isinstance(step, dict) or not IDENTIFIER.fullmatch(step.get("id", "")):
                return None
            if step["id"] in seen:
                return None
            seen.add(step["id"])
            if any(not isinstance(step.get(key), str) or not step[key].strip()
                   for key in ("title", "instruction", "criteria", "reference")):
                return None
            reference = (root / step["reference"]).resolve()
            if not reference.is_relative_to(root.resolve()) or not reference.is_file():
                return None
            package_steps.append({**step, "reference_path": reference})
        return ProductPackage(product_id, data["name"].strip(),
                              str(data.get("description") or ""), tuple(package_steps))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def public_product(package: ProductPackage) -> dict:
    return {
        "id": package.id,
        "name": package.name,
        "description": package.description,
        "steps": [{
            "id": step["id"], "title": step["title"],
            "instruction": step["instruction"], "criteria": step["criteria"],
            "referenceUrl": f"/api/products/{package.id}/references/{step['id']}",
        } for step in package.steps],
    }
