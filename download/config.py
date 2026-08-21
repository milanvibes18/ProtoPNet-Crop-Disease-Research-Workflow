# backend/config.py
# Central configuration for POWER-ProtoPNet deployment.
# Mirrors the training hyperparameters from PowerProtoNet_Explained.pdf.

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODEL_CKPT      = BASE_DIR / "checkpoints" / "power_protopnet_best.pth"
CLASSES_JSON    = BASE_DIR / "classes.json"
PROTO_IMG_DIR   = BASE_DIR / "prototype_images"   # extracted patch PNGs
LOG_DIR         = BASE_DIR / "logs"

# ─── Model Architecture (must match training config) ─────────────────────────
NUM_CLASSES         = 38
IMG_SIZE            = 224
PROTOTYPE_DIM       = 256       # SRM output channels = prototype vector dimension
PROTOTYPES_PER_CLASS = 5
NUM_PROTOTYPES      = NUM_CLASSES * PROTOTYPES_PER_CLASS   # 190
MID_CHANNELS        = 512       # reserved / future
BACKBONE            = "convnext_tiny"
SE_REDUCTION        = 16        # Squeeze-Excitation reduction ratio

# ─── Training Hyperparameters (for reference / re-training) ──────────────────
CE_WEIGHT           = 1.0
CLUSTER_WEIGHT      = 1.0
SEPARATION_WEIGHT   = 0.2
L1_WEIGHT           = 1e-5
LABEL_SMOOTHING     = 0.1
DROPOUT_SRM         = 0.3
BATCH_SIZE          = 32
EARLY_STOP_PATIENCE = 12
EARLY_STOP_MIN_DELTA = 1e-4

PHASE1_EPOCHS       = 10
PHASE1_LR           = 1e-3

PHASE2_EPOCHS       = 60
PHASE2_LR_BACKBONE  = 2e-5
PHASE2_LR_PROTO     = 1e-3
PHASE2_LR_FC        = 1e-3
PROTO_PUSH_EVERY    = 2

PHASE4_EPOCHS       = 10
PHASE4_LR_FC        = 1e-5

MIXUP_ALPHA         = 0.2
CUTMIX_ALPHA        = 1.0
WEIGHT_DECAY_BACKBONE = 5e-4
WEIGHT_DECAY_ADDON  = 1e-4

# ─── ImageNet Normalisation ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Inference Settings ───────────────────────────────────────────────────────
TOPK                = 5         # number of top-k predictions to return
MAX_PROTO_RETURN    = 10        # max prototype activations in API response
DEVICE              = "auto"    # "auto" | "cpu" | "cuda" | "mps"
USE_AMP             = True

# ─── Explainability ──────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "features.7"   # ConvNeXt-Tiny last stage
GRADCAM_COLORMAP    = "jet"           # matplotlib colormap name
GRADCAM_ALPHA       = 0.55            # overlay transparency
SHAP_NSAMPLES       = 50             # KernelSHAP background samples (speed vs quality)
SHAP_BATCH_SIZE     = 8

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST            = "0.0.0.0"
API_PORT            = 8000
ALLOWED_ORIGINS     = [
    "http://localhost:3000",    # Vite / CRA dev server
    "http://localhost:5173",    # Vite default
    "http://127.0.0.1:3000",
]
MAX_UPLOAD_MB       = 20
