# AssemLens frontend

This is a dependency-free, responsive browser interface. It includes product QR scanning, manual product ID entry, camera preview, frame capture, photo upload, result display, and a clearly labeled UI preview. Preview results are simulated and never presented as model judgments.

From the repository root, start the combined frontend and Nano API in the existing HP project environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-api.txt
ASSEMLENS_ALLOW_GPU_INFERENCE=1 python -m uvicorn assemlens_api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Omit `ASSEMLENS_ALLOW_GPU_INFERENCE=1` for preview-only mode. Open `http://localhost:8000`. Camera access on a phone requires a secure HTTPS origin; `localhost` is the development exception. Browsers without `BarcodeDetector` can use the manual product ID field. The sample preview ID is `fyd-jeep-v1`. See [the Nano API guide](../docs/nano_api.md) for product packages and GPU inference.

The **Compare with Nano AI** button opens an experimental live comparison. Upload a correct reference photo, describe the visible goal, then capture or upload the current view. This calls the Nano GPU model; it does not advance a product step or establish assembly safety. The first request can take longer while the model loads.

The Settings dialog switches between UI preview and Nano API mode. With a blank API base URL, the browser calls the same origin. A reviewed product package and reference image are required before Nano API mode can analyze a product.

## API contract

The frontend expects:

| Request | Response |
|---|---|
| `POST /api/compare` with multipart `reference`, `image`, `instruction`, `criteria` | `{ "assessment": "matches|mismatch|uncertain", "evidence": "...", "nextAction": "...", "latencyMs": 1234, "experimental": true }` |
| `GET /api/products/{productId}` | `{ "id", "name", "description", "steps": [{ "id", "title", "instruction", "criteria", "referenceUrl" }] }` |
| `POST /api/sessions` with JSON `{ "productId" }` | `{ "id": "session-id" }` |
| `POST /api/sessions/{sessionId}/observations` with multipart fields `stepId` and `image` | `{ "assessment": "matches|mismatch|uncertain", "evidence": "...", "nextAction": "...", "readyToAdvance": false, "latencyMs": 1234 }` |

QR codes may contain the product ID directly or an HTTP(S) URL whose path is `/p/{productId}`. A scanned URL supplies only the product ID; it does not change the API host. The API should reject unsupported products and should not expose packages with missing reviewed references.

The frontend sends only selected captures to the API. Its live camera preview remains in the browser. The backend must validate image content and enforce size limits independently.
