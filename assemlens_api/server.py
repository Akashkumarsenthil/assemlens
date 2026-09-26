"""AssemLens API and static frontend. Run with one Uvicorn worker."""

import asyncio
import hashlib
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError

from assemlens.policy import StepVerifier
from assemlens_api.model import ReferenceModel
from assemlens_api.products import load_product, public_product

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SESSION_SECONDS = 60 * 60


class NewSession(BaseModel):
    productId: str


@dataclass
class Session:
    product_id: str
    step_index: int
    verifier: StepVerifier
    touched: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as original:
            original.verify()
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Use a JPEG, PNG, or WebP image.")
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image resolution is too large.")
            return ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded image cannot be read.") from error


def create_app(products_root: Path | None = None, predictor=None) -> FastAPI:
    products_root = products_root or ROOT / "products"
    predictor = predictor or ReferenceModel()
    sessions: dict[str, Session] = {}
    app = FastAPI(title="AssemLens Nano API")

    def get_package(product_id: str):
        package = load_product(products_root, product_id)
        if package is None:
            raise HTTPException(404, "No reviewed package exists for this product.")
        return package

    def get_session(session_id: str):
        session = sessions.get(session_id)
        if session is None or time.monotonic() - session.touched > SESSION_SECONDS:
            sessions.pop(session_id, None)
            raise HTTPException(404, "Session expired or was not found.")
        session.touched = time.monotonic()
        return session

    @app.get("/api/health")
    def health():
        try:
            import torch
            gpu = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if gpu else None
        except (ImportError, RuntimeError):
            gpu, name = False, None
        return {"status": "ok", "gpuAvailable": gpu, "gpuName": name,
                "inferenceEnabled": os.environ.get("ASSEMLENS_ALLOW_GPU_INFERENCE") == "1",
                "modelLoaded": predictor._model is not None if isinstance(predictor, ReferenceModel) else False}

    @app.get("/api/products/{product_id}")
    def product(product_id: str):
        return public_product(get_package(product_id))

    @app.get("/api/products/{product_id}/references/{step_id}")
    def reference(product_id: str, step_id: str):
        package = get_package(product_id)
        step = next((s for s in package.steps if s["id"] == step_id), None)
        if step is None:
            raise HTTPException(404, "Step not found.")
        return FileResponse(step["reference_path"])

    @app.post("/api/sessions")
    def create_session(body: NewSession):
        package = get_package(body.productId)
        now = time.monotonic()
        for key, old in list(sessions.items()):
            if now - old.touched > SESSION_SECONDS:
                sessions.pop(key, None)
        session_id = uuid.uuid4().hex
        sessions[session_id] = Session(package.id, 0, StepVerifier(package.steps[0]["id"]))
        return {"id": session_id, "stepId": package.steps[0]["id"]}

    @app.post("/api/sessions/{session_id}/observations")
    async def observation(session_id: str, stepId: str = Form(...), image: UploadFile = File(...)):
        session = get_session(session_id)
        async with session.lock:
            package = get_package(session.product_id)
            step = package.steps[session.step_index]
            if stepId != step["id"]:
                raise HTTPException(409, "This is not the current step.")
            data = await image.read(MAX_UPLOAD_BYTES + 1)
            await image.close()
            if not data or len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Image is empty or exceeds 10 MB.")
            try:
                photo = decode_image(data)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            try:
                result = await asyncio.to_thread(predictor.predict, step, photo)
            except Exception as error:
                LOG.exception("Product observation inference failed")
                raise HTTPException(503, "The Nano model could not analyze this view.") from error
            assessment = result.get("assessment")
            if assessment not in ("matches", "mismatch", "uncertain"):
                assessment = "uncertain"
                result = {"assessment": assessment, "evidence": "The model response was unclear.",
                          "nextAction": "Take another clear view."}
            verdict = {"matches": "complete", "mismatch": "incorrect", "uncertain": "uncertain"}[assessment]
            ready = session.verifier.observe(stepId, hashlib.sha256(data).hexdigest(), verdict)
            return {"assessment": assessment,
                    "evidence": str(result.get("evidence", "")),
                    "nextAction": str(result.get("nextAction", "")),
                    "latencyMs": result.get("latencyMs"),
                    "readyToAdvance": ready}

    @app.post("/api/compare")
    async def compare(reference: UploadFile = File(...), image: UploadFile = File(...),
                      instruction: str = Form(...), criteria: str = Form(...)):
        instruction, criteria = instruction.strip(), criteria.strip()
        if not instruction or not criteria or len(instruction) > 500 or len(criteria) > 500:
            raise HTTPException(422, "Describe a visible goal and conditions in 500 characters or fewer.")
        photos = []
        for upload in (reference, image):
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            await upload.close()
            if not data or len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Each image must be smaller than 10 MB.")
            try:
                photos.append(decode_image(data))
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        try:
            result = await asyncio.to_thread(predictor.compare, photos[0], photos[1],
                                             instruction, criteria)
        except Exception as error:
            LOG.exception("Live comparison inference failed")
            raise HTTPException(503, "The Nano model could not compare these views.") from error
        if result.get("assessment") not in ("matches", "mismatch", "uncertain"):
            result = {"assessment": "uncertain", "evidence": "The model response was unclear.",
                      "nextAction": "Take another clear view."}
        return {"assessment": result["assessment"],
                "evidence": str(result.get("evidence", "")),
                "nextAction": str(result.get("nextAction", "")),
                "latencyMs": result.get("latencyMs"),
                "readyToAdvance": False, "experimental": True}

    @app.post("/api/sessions/{session_id}/advance")
    async def advance(session_id: str):
        session = get_session(session_id)
        async with session.lock:
            package = get_package(session.product_id)
            if not session.verifier.ready:
                raise HTTPException(409, "Two distinct matching captures are required.")
            if session.step_index + 1 == len(package.steps):
                return {"complete": True, "stepId": None}
            session.step_index += 1
            step_id = package.steps[session.step_index]["id"]
            session.verifier.advance(step_id)
            return {"complete": False, "stepId": step_id}

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()
