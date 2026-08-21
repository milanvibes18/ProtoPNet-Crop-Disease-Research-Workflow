# backend/predict.py
"""
POWER-ProtoPNet inference pipeline.

Exposes:
    load_model()      — loads checkpoint, returns model on correct device
    predict()         — full inference: image → PredictResponse dict
    get_device()      — resolves "auto" to cuda / mps / cpu
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from config import (
    MODEL_CKPT, CLASSES_JSON, DEVICE, TOPK,
    MAX_PROTO_RETURN, NUM_PROTOTYPES, NUM_CLASSES,
)
from model import PowerProtoPNet
from utils.preprocessing import (
    preprocess_bytes, tensor_to_b64_jpeg, pil_to_b64_jpeg, extract_patch,
)
from explainability.gradcam           import run_gradcam
from explainability.shap_explainer    import run_shap
from explainability.prototype_visualizer import analyse_prototypes
from torchvision import transforms

logger = logging.getLogger("predict")


# ─── Disease information database ─────────────────────────────────────────────
# Curated per-class descriptions, symptoms, and action plans.

_DISEASE_DB: dict[int, dict] = {
    0:  {"scientific":"Venturia inaequalis","causal":"Fungal","severity":"Moderate",
         "description":"Apple scab lesions appear as olive-green spots that darken to brown or black. The pathogen overwinters in fallen leaves and infects during wet spring conditions.",
         "symptoms":["Olive-green to brown-black lesions on leaves","Scab-like corky lesions on fruit","Premature leaf drop","Reduced fruit quality"],
         "actions":[{"level":"high","text":"Apply captan or myclobutanil fungicide at bud break"},{"level":"medium","text":"Remove and destroy all fallen leaves after harvest"},{"level":"low","text":"Plant resistant cultivars in future seasons"}]},
    5:  {"scientific":"Podosphaera clandestina","causal":"Fungal","severity":"Moderate",
         "description":"An obligate biotrophic fungus causing white powdery mycelial colonies on cherry leaf surfaces. Thrives in warm, dry conditions with high humidity.",
         "symptoms":["White powdery patches on adaxial leaf surface","Circular lesion patterns near leaf margins","Leaf curling and distortion","Premature defoliation in severe cases"],
         "actions":[{"level":"urgent","text":"Apply sulfur-based or DMI class fungicide within 48 hours"},{"level":"high","text":"Remove and destroy infected leaves to reduce inoculum"},{"level":"medium","text":"Improve canopy airflow by strategic pruning"},{"level":"low","text":"Switch to drip irrigation to reduce foliar wetness"}]},
    9:  {"scientific":"Exserohilum turcicum","causal":"Fungal","severity":"High",
         "description":"Northern corn leaf blight causes long, elliptical, tan to grayish-green lesions on corn leaves. Severe infections can reduce grain yield by 50%.",
         "symptoms":["Long tan-coloured elliptical lesions","Dark olive-green spore masses on lesion surface","Lesions coalesce causing blighting of entire leaf","Premature death of lower leaves"],
         "actions":[{"level":"urgent","text":"Apply triazole or strobilurin fungicide if disease exceeds 50% leaf area"},{"level":"high","text":"Scout fields twice weekly during humid periods"},{"level":"medium","text":"Plant resistant hybrids in following season"},{"level":"low","text":"Maintain crop rotation with non-host crops"}]},
    11: {"scientific":"Guignardia bidwellii","causal":"Fungal","severity":"High",
         "description":"Grape black rot causes circular reddish-brown leaf spots and shrivelled black mummified fruit (raisins) that remain on the vine.",
         "symptoms":["Circular tan to brown lesions with dark borders","Black pycnidia dots within lesions","Berry shrivelling and mummification","Dark tendrils and shoot infections"],
         "actions":[{"level":"urgent","text":"Apply mancozeb or myclobutanil at first sign of infection"},{"level":"high","text":"Remove and destroy all mummified fruit from vines"},{"level":"medium","text":"Prune to improve air circulation"},{"level":"low","text":"Apply dormant-season lime sulfur spray"}]},
    15: {"scientific":"Candidatus Liberibacter asiaticus","causal":"Bacterial","severity":"Critical",
         "description":"Citrus greening (Huanglongbing) is the most destructive citrus disease globally. Transmitted by psyllid insects. No cure exists; infected trees must be removed.",
         "symptoms":["Asymmetric yellowing (blotchy mottle) of leaves","Small, lopsided, bitter fruit","Stunted shoot growth","Zinc-deficiency-like symptoms"],
         "actions":[{"level":"critical","text":"Contact local plant health authority immediately — notifiable disease"},{"level":"critical","text":"Remove and destroy infected trees to prevent spread"},{"level":"urgent","text":"Control Asian citrus psyllid vector with systemic insecticide"},{"level":"high","text":"Plant certified psyllid-free nursery stock only"}]},
    21: {"scientific":"Phytophthora infestans","causal":"Oomycete","severity":"Critical",
         "description":"The causative agent of the 1845 Irish Potato Famine. Destroys entire fields within days in cool, wet conditions. Airborne sporangia can travel miles.",
         "symptoms":["Dark water-soaked lesions on leaves and stems","White sporulating mycelium on abaxial surface","Rapid brown-black necrosis","Foul-smelling tuber rot"],
         "actions":[{"level":"critical","text":"Immediately destroy all infected foliage"},{"level":"urgent","text":"Apply metalaxyl-M or fluopicolide at first symptoms"},{"level":"high","text":"Harvest tubers early to prevent storage rot"},{"level":"medium","text":"Improve field drainage; reduce foliar wetness"}]},
    30: {"scientific":"Phytophthora infestans","causal":"Oomycete","severity":"Critical",
         "description":"Tomato late blight. Same pathogen as potato late blight. Capable of destroying unprotected crops within 7–14 days under favourable conditions.",
         "symptoms":["Dark greasy water-soaked lesions on leaves","White sporulating mold on abaxial surface","Brown-black necrotic tissue collapse","Firm brown rot on fruit"],
         "actions":[{"level":"critical","text":"Remove and destroy all infected plant material immediately"},{"level":"urgent","text":"Apply copper-based or mancozeb fungicide to entire crop"},{"level":"high","text":"Alert neighbouring farms for coordinated management"},{"level":"medium","text":"Reduce leaf wetness; improve airflow and drainage"}]},
    35: {"scientific":"Tomato yellow leaf curl virus (TYLCV)","causal":"Viral","severity":"Critical",
         "description":"Begomovirus transmitted by the whitefly Bemisia tabaci. Causes severe stunting and yield loss up to 100%. No chemical cure; vector control is paramount.",
         "symptoms":["Upward and inward leaf curling","Yellow leaf margins (chlorotic)","Stunted plant growth","Reduced fruit set and small distorted fruit"],
         "actions":[{"level":"critical","text":"Remove and bag infected plants to prevent vector acquisition"},{"level":"urgent","text":"Apply systemic insecticide to control whitefly vector"},{"level":"high","text":"Install UV-reflective mulch to deter whiteflies"},{"level":"medium","text":"Use insect-proof netting in nursery phase"}]},
}

_HEALTHY_INFO = {
    "scientific": None, "causal": None, "severity": "None",
    "description": "No disease symptoms detected. The specimen appears healthy based on prototype activation patterns consistent with healthy tissue classes.",
    "symptoms": ["No lesions or discolouration detected"],
    "actions": [{"level": "low", "text": "Continue routine monitoring; maintain optimal growing conditions"}],
}

_DEFAULT_INFO = {
    "scientific": "See specialist", "causal": "Mixed", "severity": "Moderate",
    "description": "Disease detected. Consult agronomist with this diagnostic report for a tailored treatment plan.",
    "symptoms": ["Abnormal visual patterns detected in specimen"],
    "actions": [{"level": "medium", "text": "Consult a local plant pathologist or extension service"}],
}


# ─── Device selection ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    cfg = DEVICE.lower()
    if cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg)


# ─── Class label loader ───────────────────────────────────────────────────────

def load_class_names() -> list[str]:
    with open(CLASSES_JSON) as f:
        data = json.load(f)
    return [c["name"] for c in sorted(data["classes"], key=lambda x: x["index"])]


def load_class_metadata() -> list[dict]:
    with open(CLASSES_JSON) as f:
        data = json.load(f)
    return sorted(data["classes"], key=lambda x: x["index"])


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: Optional[str] = None,
    device:    Optional[torch.device] = None,
) -> tuple[PowerProtoPNet, torch.device]:
    """
    Instantiate POWER-ProtoPNet and load checkpoint weights.

    If checkpoint does not exist (first run), the model is returned with
    pretrained ConvNeXt-Tiny weights only (add-on modules are randomly
    initialised) — suitable for development / UI testing.
    """
    if device is None:
        device = get_device()

    model = PowerProtoPNet().to(device)
    model.eval()

    path = Path(ckpt_path or MODEL_CKPT)
    if path.exists():
        logger.info(f"Loading checkpoint from {path}")
        model.load_checkpoint(str(path), device)
    else:
        logger.warning(
            f"Checkpoint not found at {path}. "
            "Using pretrained backbone only. Predictions will be random."
        )

    return model, device


# ─── Inference transforms (mirrors training val pipeline) ─────────────────────

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 256 / 224)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Full prediction pipeline ─────────────────────────────────────────────────

def predict(
    raw_bytes:      bytes,
    model:          PowerProtoPNet,
    device:         torch.device,
    class_names:    list[str],
    run_gradcam_:   bool = True,
    run_shap_:      bool = False,   # expensive; enable via request param
    top_k:          int  = TOPK,
    max_protos:     int  = MAX_PROTO_RETURN,
) -> dict:
    """
    End-to-end inference.

    Args:
        raw_bytes:   Raw uploaded image bytes
        model:       Loaded PowerProtoPNet (eval mode)
        device:      Inference device
        class_names: 38-element label list from classes.json
        run_gradcam_: Whether to compute GradCAM++ (fast, recommended)
        run_shap_:    Whether to compute SHAP (slow, optional)

    Returns:
        dict matching PredictResponse schema (utils/response_models.py)
    """
    t0 = time.perf_counter()

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    from utils.preprocessing import load_pil_from_bytes
    from PIL import Image as PILImage

    pil_orig = load_pil_from_bytes(raw_bytes)
    pil_224  = pil_orig.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
    tensor   = _INFER_TRANSFORM(pil_orig).unsqueeze(0).to(device)

    # ── 2. Model forward ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits, sim, act_maps = model(tensor)   # [1,38], [1,190], [1,190,7,7]

    probs = torch.softmax(logits, dim=1)[0]     # [38]
    sim_v = sim[0]                              # [190]
    maps  = act_maps[0]                         # [190,7,7]

    # ── 3. Top-K predictions ───────────────────────────────────────────────────
    topk_vals, topk_idxs = probs.topk(min(top_k, NUM_CLASSES))
    pred_cls   = topk_idxs[0].item()
    confidence = topk_vals[0].item()

    top_k_list = [
        {"name": class_names[i.item()], "index": i.item(), "probability": round(v.item(), 6)}
        for v, i in zip(topk_vals, topk_idxs)
    ]

    # ── 4. Prototype activations ───────────────────────────────────────────────
    protos = analyse_prototypes(
        sim=sim_v, act_maps=maps,
        original_image=pil_224,
        class_names=class_names,
        top_n=max_protos,
    )

    # ── 5. Disease information ────────────────────────────────────────────────
    cls_meta  = load_class_metadata()
    is_healthy = cls_meta[pred_cls]["healthy"] if pred_cls < len(cls_meta) else False
    disease_info = (
        _HEALTHY_INFO
        if is_healthy
        else _DISEASE_DB.get(pred_cls, _DEFAULT_INFO)
    )

    # ── 6. GradCAM++ ──────────────────────────────────────────────────────────
    gradcam_result = None
    if run_gradcam_:
        try:
            # Re-run with grad enabled on a fresh tensor
            tensor_gc = _INFER_TRANSFORM(pil_orig).unsqueeze(0).to(device)
            gradcam_result = run_gradcam(
                model=model,
                input_tensor=tensor_gc,
                class_idx=pred_cls,
                original_image=pil_224,
            )
        except Exception as e:
            logger.warning(f"GradCAM++ failed: {e}")

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    shap_result = None
    if run_shap_:
        try:
            shap_result = run_shap(
                model=model,
                original_image=pil_224,
                target_class=pred_cls,
                device=device,
                transform=_INFER_TRANSFORM,
            )
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    # ── 8. Assemble response ──────────────────────────────────────────────────
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "prediction": {
            "class_name":  class_names[pred_cls],
            "class_index": pred_cls,
            "confidence":  round(confidence, 6),
            "top_k":       top_k_list,
        },
        "prototypes":    protos,
        "disease_info":  disease_info,
        "gradcam":       gradcam_result,
        "shap":          shap_result,
        "model_version": "POWER-ProtoPNet-v1.0",
        "inference_ms":  inference_ms,
    }
