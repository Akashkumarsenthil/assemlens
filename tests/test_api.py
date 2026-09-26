"""API contract and verification gate without loading a GPU model."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from assemlens_api.server import create_app


def jpeg(color):
    out = io.BytesIO()
    Image.new("RGB", (20, 20), color).save(out, format="JPEG")
    return out.getvalue()


class StubPredictor:
    def compare(self, reference, observation, instruction, criteria):
        return {"assessment": "mismatch", "evidence": "Part differs.",
                "nextAction": "Check the part.", "latencyMs": 7}

    def predict(self, step, observation):
        return {"assessment": "matches", "evidence": "Visible parts match.",
                "nextAction": "Confirm the step.", "latencyMs": 5}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "references").mkdir()
        (self.root / "references" / "step_01.jpg").write_bytes(jpeg("green"))
        self.product = {
            "id": "test-jeep", "name": "Test jeep", "description": "Test package",
            "approved": True,
            "steps": [{"id": "step_01", "title": "Visual check",
                       "instruction": "Check the part.", "criteria": "Part is visible.",
                       "reference": "references/step_01.jpg"}],
        }
        (self.root / "test-jeep.json").write_text(json.dumps(self.product))
        self.client = TestClient(create_app(self.root, StubPredictor()))

    def post_image(self, session_id, data):
        return self.client.post(f"/api/sessions/{session_id}/observations",
                                data={"stepId": "step_01"},
                                files={"image": ("capture.jpg", data, "image/jpeg")})

    def test_product_session_and_two_distinct_captures(self):
        product = self.client.get("/api/products/test-jeep")
        self.assertEqual(product.status_code, 200)
        self.assertEqual(product.json()["steps"][0]["referenceUrl"],
                         "/api/products/test-jeep/references/step_01")
        session = self.client.post("/api/sessions", json={"productId": "test-jeep"}).json()["id"]
        first = self.post_image(session, jpeg("red"))
        self.assertEqual(first.status_code, 200)
        self.assertFalse(first.json()["readyToAdvance"])
        self.assertFalse(self.post_image(session, jpeg("red")).json()["readyToAdvance"])
        self.assertEqual(self.client.post(f"/api/sessions/{session}/advance").status_code, 409)
        self.assertTrue(self.post_image(session, jpeg("blue")).json()["readyToAdvance"])
        self.assertEqual(self.client.post(f"/api/sessions/{session}/advance").json(),
                         {"complete": True, "stepId": None})

    def test_experimental_compare_without_product_package(self):
        response = self.client.post("/api/compare",
            data={"instruction": "Check the part.", "criteria": "Part is visible."},
            files={"reference": ("reference.jpg", jpeg("green"), "image/jpeg"),
                   "image": ("current.jpg", jpeg("red"), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assessment"], "mismatch")
        self.assertTrue(response.json()["experimental"])
        self.assertFalse(response.json()["readyToAdvance"])
        invalid = self.client.post("/api/compare",
            data={"instruction": "Check the part.", "criteria": "Part is visible."},
            files={"reference": ("reference.jpg", b"bad", "image/jpeg"),
                   "image": ("current.jpg", jpeg("red"), "image/jpeg")})
        self.assertEqual(invalid.status_code, 422)

    def test_unreviewed_package_and_invalid_image_rejected(self):
        self.product["approved"] = False
        (self.root / "test-jeep.json").write_text(json.dumps(self.product))
        self.assertEqual(self.client.get("/api/products/test-jeep").status_code, 404)
        self.product["approved"] = True
        (self.root / "test-jeep.json").write_text(json.dumps(self.product))
        session = self.client.post("/api/sessions", json={"productId": "test-jeep"}).json()["id"]
        self.assertEqual(self.post_image(session, b"not an image").status_code, 422)
        self.assertEqual(self.client.post(f"/api/sessions/{session}/observations",
                                          data={"stepId": "wrong"},
                                          files={"image": ("a.jpg", jpeg("red"), "image/jpeg")}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
