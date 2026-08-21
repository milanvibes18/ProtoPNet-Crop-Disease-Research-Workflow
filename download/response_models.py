# backend/utils/response_models.py
"""
Pydantic v2 schemas for all API request/response bodies.
Import these in app.py for automatic OpenAPI documentation.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ─── Prediction sub-models ────────────────────────────────────────────────────

class TopKEntry(BaseModel):
    """One entry in the top-k ranked predictions."""
    model_config = ConfigDict(frozen=True)

    name:       str            = Field(..., description="Human-readable class name")
    index:      int            = Field(..., ge=0, description="Class index (0–37)")
    probability: float         = Field(..., ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Primary prediction and top-k distribution."""
    class_name:  str           = Field(..., description="Predicted class label")
    class_index: int           = Field(..., ge=0)
    confidence:  float         = Field(..., ge=0.0, le=1.0)
    top_k:       list[TopKEntry]


# ─── Prototype sub-models ─────────────────────────────────────────────────────

class PrototypeActivation(BaseModel):
    """Single prototype activation record."""
    id:          int           = Field(..., ge=0, lt=190, description="Global prototype index")
    rank:        int           = Field(..., ge=1, description="Rank by similarity (1 = highest)")
    class_name:  str
    class_index: int
    similarity:  float         = Field(..., description="Cosine similarity [-1, +1]")
    patch_h:     Optional[int] = Field(None, description="Best-matching patch row in 7×7 grid")
    patch_w:     Optional[int] = Field(None, description="Best-matching patch column in 7×7 grid")
    patch_image_b64: Optional[str] = Field(None, description="Base64-encoded crop of best-matching patch")


# ─── Disease info sub-models ──────────────────────────────────────────────────

class ActionItem(BaseModel):
    level:  str   = Field(..., description="urgent | high | medium | low")
    text:   str


class DiseaseInfo(BaseModel):
    scientific:  Optional[str]
    causal:      Optional[str]
    severity:    str
    description: str
    symptoms:    list[str]
    actions:     list[ActionItem]


# ─── XAI sub-models ───────────────────────────────────────────────────────────

class GradCAMResult(BaseModel):
    """GradCAM++ output."""
    overlay_b64:   str = Field(..., description="Base64 JPEG: heatmap blended onto image")
    heatmap_b64:   str = Field(..., description="Base64 JPEG: raw jet-colormap heatmap")
    target_layer:  str
    target_class:  int


class SHAPResult(BaseModel):
    """SHAP pixel-level attribution."""
    shap_map_b64:      str   = Field(..., description="Base64 JPEG: SHAP heatmap (green=positive, red=negative)")
    positive_regions:  list[str]
    negative_regions:  list[str]
    mean_positive_shap: float
    mean_negative_shap: float


# ─── Top-level response ───────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Full inference response returned by POST /predict.
    XAI fields are null when the relevant module is disabled or fails.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core prediction
    prediction:   PredictionResult
    prototypes:   list[PrototypeActivation]

    # Disease information
    disease_info: Optional[DiseaseInfo] = None

    # XAI outputs (None if not computed)
    gradcam:      Optional[GradCAMResult] = None
    shap:         Optional[SHAPResult]   = None

    # Meta
    model_version: str  = "POWER-ProtoPNet-v1.0"
    inference_ms:  float = Field(..., description="Total server-side inference time in ms")


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    num_classes:   int
    num_prototypes: int
    device:        str
