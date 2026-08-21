# POWER-ProtoPNet · PlantWild AI Research Platform

**Prototype-based Explainable Deep Learning for Plant Disease Recognition**  
ConvNeXt-Tiny + SE Attention + SRM + Prototype Layer · 38 Classes · 99.65% Test Accuracy

---

## Project Structure

```
plantwild-ai/
│
├── frontend/
│   └── App.jsx                        ← Full React SPA (4 tabs: Diagnose, Research, Architecture, Paper)
│
├── backend/
│   ├── app.py                         ← FastAPI application (all routes)
│   ├── model.py                       ← POWER-ProtoPNet architecture (PyTorch)
│   ├── predict.py                     ← Inference pipeline
│   ├── config.py                      ← All hyperparameters & paths
│   ├── classes.json                   ← 38-class metadata
│   ├── requirements.txt
│   │
│   ├── utils/
│   │   ├── preprocessing.py           ← Image transforms, denorm, patch extraction
│   │   └── response_models.py         ← Pydantic v2 API schemas
│   │
│   ├── explainability/
│   │   ├── gradcam.py                 ← GradCAM++ (hooks on ConvNeXt last stage)
│   │   ├── shap_explainer.py          ← KernelSHAP on superpixel segments
│   │   └── prototype_visualizer.py    ← Prototype ranking + patch extraction
│   │
│   └── checkpoints/                   ← Place model checkpoint here
│       └── power_protopnet_best.pth   ← ← ← YOUR TRAINED CHECKPOINT
│
└── README.md
```

---

## Quick Start

### 1 · Frontend (Demo Mode)

No backend required in demo mode. Open `App.jsx` in a Vite/CRA project:

```bash
npm create vite@latest plantwild-ui -- --template react
cd plantwild-ui
npm install recharts
cp /path/to/App.jsx src/App.jsx
npm run dev
```

The app runs in **demo mode** (`DEMO_MODE = true` in App.jsx) and shows a
sample Cherry Powdery Mildew result for any uploaded image, with real training
charts rendered from JSON data.

To connect to the live backend, set `DEMO_MODE = false` in `App.jsx`.

---

### 2 · Backend

**Prerequisites:** Python ≥ 3.10, PyTorch ≥ 2.2, CUDA optional

```bash
cd backend
pip install -r requirements.txt
```

**Place your checkpoint:**

```bash
mkdir -p checkpoints
cp /path/to/your/best_checkpoint.pth checkpoints/power_protopnet_best.pth
```

**Start the API server:**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### `POST /predict`

Upload a leaf image for full inference.

**Request:** `multipart/form-data`

| Field      | Type    | Default | Description                             |
|------------|---------|---------|-----------------------------------------|
| file       | File    | —       | Leaf image (JPEG/PNG/WEBP, max 20 MB)   |
| gradcam    | bool    | true    | Compute GradCAM++ heatmap               |
| shap       | bool    | false   | Compute SHAP attribution (slow ~5-10s)  |
| top_k      | int     | 5       | Number of top-K predictions             |
| max_protos | int     | 10      | Max prototype activations to return     |

**Response:**

```json
{
  "prediction": {
    "class_name":  "Cherry — Powdery Mildew",
    "class_index": 5,
    "confidence":  0.9854,
    "top_k": [
      { "name": "Cherry — Powdery Mildew", "index": 5, "probability": 0.9854 },
      ...
    ]
  },
  "prototypes": [
    {
      "id":              27,
      "rank":            1,
      "class_name":      "Cherry — Powdery Mildew",
      "class_index":     5,
      "similarity":      0.924,
      "patch_h":         3,
      "patch_w":         4,
      "patch_image_b64": "<base64 JPEG>"
    },
    ...
  ],
  "disease_info": {
    "scientific":  "Podosphaera clandestina",
    "causal":      "Fungal",
    "severity":    "Moderate",
    "description": "...",
    "symptoms":    ["..."],
    "actions":     [{ "level": "urgent", "text": "..." }]
  },
  "gradcam": {
    "overlay_b64":  "<base64 JPEG>",
    "heatmap_b64":  "<base64 JPEG>",
    "target_layer": "features.7",
    "target_class": 5
  },
  "shap": null,
  "model_version": "POWER-ProtoPNet-v1.0",
  "inference_ms":  143.7
}
```

### `GET /health`

```json
{
  "status":         "ok",
  "model_loaded":   true,
  "num_classes":    38,
  "num_prototypes": 190,
  "device":         "cuda"
}
```

### `GET /classes`

Returns the full 38-class metadata list from `classes.json`.

### `GET /prototypes/{id}/image`

Returns a JPEG of the saved prototype push image.  
Requires running `save_prototype_images()` after training.

---

## Model Architecture

```
Input [B, 3, 224, 224]
    │
    ▼  ConvNeXt-Tiny (pretrained ImageNet-1K)
    │  7×7 DW kernels · Layer Norm · GELU · Residual
    │  Output: [B, 768, 7, 7]
    │
    ▼  SE Attention Block (r=16)
    │  GAP → FC(768→48) → ReLU → FC(48→768) → Sigmoid → Scale
    │  Output: [B, 768, 7, 7]
    │
    ▼  Spatial Refinement Module
    │  DWConv3×3 → BN → ReLU → PWConv1×1 → BN → Dropout2d(0.3)
    │  Output: [B, 256, 7, 7]
    │
    ▼  Prototype Layer (190 prototypes, 256-dim, cosine similarity)
    │  sim[b,j] = max_s cos(f̂[b,:,s], p̂_j)
    │  Output: [B, 190]
    │
    ▼  FC Classification Head (bias-free)
    │  W_fc[c,j] = +1.0 (own-class), −0.5 (others)
    │  Output: [B, 38] logits
    │
    ▼  Softmax → Prediction + Confidence
```

**Key numbers:**

| Component       | Parameters | Output Shape    |
|-----------------|-----------|-----------------|
| ConvNeXt-Tiny   | 28.6M     | B×768×7×7       |
| SE Block        | ~75K      | B×768×7×7       |
| SRM             | ~200K     | B×256×7×7       |
| Prototype Layer | 190×256   | B×190           |
| FC Head         | 38×190    | B×38            |

---

## Training Protocol

| Phase   | Epochs | Modules Trained        | LR (backbone / addon)  | Notes                          |
|---------|--------|------------------------|------------------------|--------------------------------|
| Phase 1 | 10     | SRM + FC               | — / 10⁻³              | Warmup, backbone frozen        |
| Phase 2 | 28     | All                    | 2×10⁻⁵ / 10⁻³         | Prototype push every 2 epochs  |
| Phase 4 | 10     | FC only                | — / 10⁻⁵              | L1 sparsity fine-tuning        |

**Augmentation:** MixUp (α=0.2) + CutMix (α=1.0) applied 50/50 during Phase 2.

**Achieved Results:**
- Best Val Accuracy: **99.77%** (Epoch 16)
- Final Test Accuracy: **99.65%**
- Min Val Loss: **0.7408** (Epoch 8)
- Generalisation Gap: ≤ 0.30%

---

## Enabling Live XAI in the Frontend

When the backend is running, set in `App.jsx`:

```js
const DEMO_MODE = false;
const API_BASE = "http://localhost:8000"; // or your production URL
```

The frontend will then:
- Call `POST /predict` with the uploaded image
- Display real prototype similarity scores and patch images
- Render the actual GradCAM++ overlay from the backend
- Show SHAP attribution if `shap=true` is passed

---

## Saving Prototype Images (after training)

After training completes, save prototype patches once so the API can serve them:

```python
from predict import load_model, load_class_names
from explainability.prototype_visualizer import save_prototype_images
from torch.utils.data import DataLoader
# ... your dataset

model, device = load_model()
class_names   = load_class_names()
loader        = DataLoader(your_train_dataset, batch_size=32)
save_prototype_images(model, loader, device, class_names)
```

This creates `backend/prototype_images/proto_XXXX.jpg` for all 190 prototypes.

---

## Frontend Tab Overview

| Tab | Content |
|----|---------|
| **Diagnose** | Upload → Inference → Confidence gauge + Top-K chart + Disease info + Prototype cards + GradCAM + SHAP |
| **Research Metrics** | Live recharts from JSON logs: accuracy, loss, gen-gap, prototype push impact, timeline |
| **Architecture** | Expandable pipeline explorer: each stage with description, shape, formula |
| **Research Paper** | Abstract, methodology, results, explainability analysis, contributions |

---

## Team

| Name              | Role                    |
|-------------------|-------------------------|
| Milan Kundu       | ML Research & Development |
| Nilanjan Pan      | ML Research & Development |
| Soumo Chatterjee  | ML Research & Development |
| Ayan Halder       | ML Research & Development |

---

## Citation

```
@misc{powerprotopnet2025,
  title     = {POWER-ProtoPNet: Prototype-based Explainable Deep Learning for Plant Disease Recognition},
  author    = {Kundu, Milan and Pan, Nilanjan and Chatterjee, Soumo and Halder, Ayan},
  year      = {2025},
  note      = {PlantWild v3 · ConvNeXt-Tiny · 99.65\% test accuracy}
}
```
