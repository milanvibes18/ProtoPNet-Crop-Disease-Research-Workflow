# backend/app.py
"""
POWER-ProtoPNet FastAPI backend.

Routes:
    GET  /health          — liveness / model status
    GET  /classes         — 38-class metadata
    POST /predict         — full inference with XAI
    POST /predict/gradcam — GradCAM++ only (faster)
    GET  /prototypes/{id} — saved prototype patch image
    GET  /docs            — auto-generated OpenAPI docs

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from config import (
    API_HOST, API_PORT, ALLOWED_ORIGINS, MAX_UPLOAD_MB,
    NUM_CLASSES, NUM_PROTOTYPES, PROTO_IMG_DIR,
)
from predict import load_model, load_class_names, load_class_metadata, predict, get_device
from utils.response_models import PredictResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# ─── Application State ────────────────────────────────────────────────────────

class _State:
    model        = None
    device       = None
    class_names: list[str] = []
    class_meta:  list[dict] = []


state = _State()


# ─── Lifespan (model loading) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    logger.info("⚡ Starting POWER-ProtoPNet backend…")
    try:
        state.device       = get_device()
        state.model, _     = load_model(device=state.device)
        state.class_names  = load_class_names()
        state.class_meta   = load_class_metadata()
        logger.info(f"✅ Model loaded on {state.device} | {NUM_CLASSES} classes | {NUM_PROTOTYPES} prototypes")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}\n{traceback.format_exc()}")
    yield
    logger.info("👋 Shutting down POWER-ProtoPNet backend")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "POWER-ProtoPNet API",
    description = (
        "Explainable plant disease diagnostics powered by POWER-ProtoPNet "
        "(ConvNeXt-Tiny + SE Attention + SRM + Prototype Layer). "
        "Returns predictions, top-k distributions, prototype activations, "
        "GradCAM++ heatmaps, and SHAP pixel attributions."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile, max_mb: int = MAX_UPLOAD_MB) -> bytes:
    """Read uploaded file with size guard."""
    content = await file.read()
    if len(content) > max_mb * 1_048_576:
        raise HTTPException(413, f"File exceeds {max_mb} MB limit")
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image files are accepted")
    return content


def _ensure_model():
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    """Liveness probe — returns model loading status."""
    return HealthResponse(
        status         = "ok" if state.model else "degraded",
        model_loaded   = state.model is not None,
        num_classes    = NUM_CLASSES,
        num_prototypes = NUM_PROTOTYPES,
        device         = str(state.device) if state.device else "unknown",
    )


@app.get("/classes", tags=["Metadata"])
async def get_classes():
    """Return the full 38-class metadata list."""
    return {"classes": state.class_meta, "total": len(state.class_meta)}


@app.get("/classes/{class_index}", tags=["Metadata"])
async def get_class(class_index: int):
    """Return metadata for a single class."""
    if class_index < 0 or class_index >= len(state.class_meta):
        raise HTTPException(404, f"Class index {class_index} out of range (0–{NUM_CLASSES-1})")
    return state.class_meta[class_index]


@app.post(
    "/predict",
    response_model = PredictResponse,
    tags           = ["Inference"],
    summary        = "Full POWER-ProtoPNet inference with XAI",
)
async def predict_endpoint(
    file:        UploadFile = File(..., description="Leaf image (JPEG/PNG/WEBP, max 20 MB)"),
    gradcam:     bool       = Query(True,  description="Compute GradCAM++ heatmap"),
    shap:        bool       = Query(False, description="Compute SHAP attribution (slow)"),
    top_k:       int        = Query(5,     ge=1, le=38, description="Number of top-k predictions"),
    max_protos:  int        = Query(10,    ge=1, le=190, description="Max prototype activations"),
):
    """
    Upload a leaf image and receive:
    - Predicted disease + confidence
    - Top-K probability distribution
    - Top activated prototype patches with similarity scores
    - Disease information (scientific name, severity, treatment)
    - GradCAM++ spatial attribution heatmap
    - (Optional) SHAP pixel-level attribution
    """
    _ensure_model()
    raw = await _read_upload(file)

    try:
        result = predict(
            raw_bytes    = raw,
            model        = state.model,
            device       = state.device,
            class_names  = state.class_names,
            run_gradcam_ = gradcam,
            run_shap_    = shap,
            top_k        = top_k,
            max_protos   = max_protos,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Inference failed: {str(e)}")

    return JSONResponse(content=result)


@app.post(
    "/predict/gradcam",
    tags    = ["Inference"],
    summary = "GradCAM++ only — faster than full predict",
)
async def gradcam_only(
    file:        UploadFile = File(...),
    class_index: Optional[int] = Query(None, ge=0, lt=38, description="Target class (auto if omitted)"),
):
    """
    Return GradCAM++ overlay without running prototype analysis or SHAP.
    Useful for quick visual inspection.
    """
    _ensure_model()
    raw = await _read_upload(file)

    from predict import _INFER_TRANSFORM, load_pil_from_bytes_local
    from utils.preprocessing import load_pil_from_bytes
    import torch

    pil_orig = load_pil_from_bytes(raw)
    from PIL import Image as PILImg
    pil_224  = pil_orig.resize((224, 224), PILImg.LANCZOS)
    tensor   = _INFER_TRANSFORM(pil_orig).unsqueeze(0).to(state.device)

    with torch.no_grad():
        logits, _, _ = state.model(tensor)
    pred_cls = logits.argmax(dim=1).item() if class_index is None else class_index

    try:
        from explainability.gradcam import run_gradcam
        gc = run_gradcam(state.model, tensor, pred_cls, pil_224)
    except Exception as e:
        raise HTTPException(500, f"GradCAM++ failed: {e}")

    return JSONResponse(content={
        "class_name":   state.class_names[pred_cls],
        "class_index":  pred_cls,
        **gc,
    })


@app.get(
    "/prototypes/{proto_id}/image",
    tags    = ["Prototypes"],
    summary = "Return saved prototype patch image",
)
async def prototype_image(proto_id: int):
    """
    Return the JPEG image of a saved prototype patch.
    Requires prototype push images to be saved in PROTO_IMG_DIR.
    """
    if proto_id < 0 or proto_id >= NUM_PROTOTYPES:
        raise HTTPException(404, f"Prototype ID {proto_id} out of range (0–{NUM_PROTOTYPES-1})")
    from pathlib import Path
    path = Path(PROTO_IMG_DIR) / f"proto_{proto_id:04d}.jpg"
    if not path.exists():
        raise HTTPException(404, f"No saved image for prototype {proto_id}. Run save_prototype_images() after training.")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.get(
    "/prototypes/{proto_id}/info",
    tags    = ["Prototypes"],
    summary = "Prototype metadata",
)
async def prototype_info(proto_id: int):
    """Return class assignment and similarity statistics for a prototype."""
    if proto_id < 0 or proto_id >= NUM_PROTOTYPES:
        raise HTTPException(404)
    from config import PROTOTYPES_PER_CLASS
    cls_idx = proto_id // PROTOTYPES_PER_CLASS
    rank    = proto_id % PROTOTYPES_PER_CLASS
    cls_info = state.class_meta[cls_idx] if cls_idx < len(state.class_meta) else {}
    return {
        "proto_id":     proto_id,
        "class_index":  cls_idx,
        "class_name":   state.class_names[cls_idx] if cls_idx < len(state.class_names) else "Unknown",
        "rank_in_class": rank,
        "class_info":   cls_info,
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = True,
        workers = 1,          # model is not multiprocess-safe without shared memory
    )
