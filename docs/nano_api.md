# Nano API and frontend

The API serves the frontend and accepts selected camera captures. Keep one API process and one model instance on the GB10; use one Uvicorn worker.

```bash
cd /home/hp15/git_happens/assemlens
source .venv/bin/activate
python -m pip install -r requirements-api.txt
ASSEMLENS_ALLOW_GPU_INFERENCE=1 python -m uvicorn assemlens_api.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Forward port 8000 in VS Code Remote SSH. Open `http://localhost:8000` on the client. `/api/health` reports CUDA availability and whether live inference is enabled. Omit `ASSEMLENS_ALLOW_GPU_INFERENCE=1` to keep the preview UI available without loading the model. The model loads on the first live request, which can take longer than later requests. Preview mode and experimental live comparison do not need a reviewed product package. Live comparison accepts a user-provided reference, visible goal, and current capture at `POST /api/compare`; it returns model guidance without step advancement. Avoid running model inference while another teammate is training on the same GPU.

## Product package

Only `products/<product-id>.json` files with `approved: true`, complete step fields, and existing local reference images become available through the API. A future reviewed jeep package would have this structure:

```json
{
  "id": "fyd-jeep-v1",
  "name": "FYD take-apart jeep",
  "description": "Visible checkpoint guide",
  "approved": true,
  "steps": [
    {
      "id": "step_01",
      "title": "Exact visible checkpoint name",
      "instruction": "Exact step from the kit instructions",
      "criteria": "Visible conditions that must all be confirmed",
      "reference": "references/fyd-jeep-v1-step-01.jpg"
    }
  ]
}
```

Keep `approved` false or omit the package until a teammate reviews the reference and conditions. The existing PC template and placeholder jeep task are not supported products.

The API contract is in [frontend/README.md](../frontend/README.md). A session holds the current step in memory. Two distinct matching captures mark it ready; advancing requires an explicit `POST /api/sessions/{id}/advance`. A restart clears sessions. Uploaded images are limited to 10 MB and 16 million pixels; JPEG, PNG, and WebP are accepted. Images stay in memory for inference and are not saved by this service.

The model uses the base Qwen3-VL-4B-Instruct revision recorded in `runs/verifier_v2/config.json` and the reference/current-image prompt from `scripts/jeep_state_eval.py`. The old verifier adapter is intentionally not used because its validation mistake recall was poor. Invalid model JSON becomes an uncertain result. Model errors return HTTP 503. Model output is experimental guidance; the frontend does not automatically advance a step.
