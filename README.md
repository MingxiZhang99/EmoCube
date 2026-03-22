<div align="center">

# EmoCube

**Real-Time Facial Emotion Recognition for Interactive XR Experiences**

[![Unity](https://img.shields.io/badge/Unity-6000.0-black?logo=unity)](https://unity.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Opset%2013-purple?logo=onnx)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

EmoCube is an emotion-driven XR prototype that captures real-time webcam feeds, detects faces via **BlazeFace**, classifies emotions through a lightweight **MobileNetV3-Small** model, and visualizes the results by dynamically changing a 3D cube's color — all running on-device in Unity.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Emotion Recognition Pipeline](#emotion-recognition-pipeline)
  - [Approach 1 — DINOv3 + SVM](#approach-1--dinov3--svm)
  - [Approach 2 — MobileNetV3-Small (Lightweight)](#approach-2--mobilenetv3-small-lightweight)
- [ONNX Export for Unity](#onnx-export-for-unity)
- [Unity Integration](#unity-integration)
  - [Face Detection with BlazeFace](#face-detection-with-blazeface)
  - [Emotion Inference at Runtime](#emotion-inference-at-runtime)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Training & Export](#training--export)
  - [Unity Setup](#unity-setup)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

EmoCube bridges the gap between deep-learning-based emotion recognition research and real-time XR interactivity. The project explores two distinct approaches to facial emotion classification — a high-accuracy **DINOv3 + SVM** pipeline and a deployment-optimized **MobileNetV3-Small** network — then deploys the lightweight model into **Unity 6** via ONNX for real-time inference.

### Key Features

- **Dual-approach experimentation**: research-grade DINOv3 embeddings + SVM vs. lightweight MobileNetV3-Small for on-device deployment
- **End-to-end ONNX pipeline**: PyTorch training → ONNX export (opset 13, fixed batch) → Unity Inference Engine
- **Real-time face detection**: BlazeFace short-range model running at GPU speed via Unity's Inference Engine with in-graph NMS
- **Emotion-to-color mapping**: 4-class emotion output (Happy / Sad / Angry / Fear) dynamically drives a 3D cube's appearance
- **Shared webcam architecture**: singleton-based `WebcamManager` ensures a single camera stream is reused across all components

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PYTHON  (Training & Export)                     │
│                                                                     │
│   DiffusionEmotion_S Dataset                                        │
│          │                                                          │
│          ├──→ DINOv3 + SVM  (research / high accuracy)              │
│          │                                                          │
│          └──→ MobileNetV3-Small  (lightweight / deployable)         │
│                     │                                               │
│                     ▼                                               │
│          torch.onnx.export  (opset 13, batch=1, no dynamic axes)    │
│                     │                                               │
│          emotion_mnv3_4cls_unity.onnx + labels.json + preprocess.json│
└────────────────────────┬────────────────────────────────────────────┘
                         │  ONNX model + metadata
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     UNITY 6  (Real-Time Inference)                  │
│                                                                     │
│  WebcamManager (singleton)                                          │
│       │                                                             │
│       ├──→ BlazeFace ONNX  ──→  Face bounding box + 6 keypoints    │
│       │        (128×128 input, GPU compute, in-graph NMS)           │
│       │                                                             │
│       ├──→ MobileNetV3 ONNX ──→  4-class emotion  ──→  Cube color  │
│       │        (224×224 input, ImageNet normalization)               │
│       │                                                             │
│       └──→ WebcamToMaterial  ──→  Live preview on Quad              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Emotion Recognition Pipeline

### Approach 1 — DINOv3 + SVM

> **Purpose**: High-accuracy baseline for research and data collection.

We leverage **DINOv3** (`facebook/dinov3-vits16plus-pretrain-lvd1689m`) as a frozen feature extractor and train an **RBF-kernel SVM** on the CLS-token embeddings for emotion classification.

| Step | Details |
|------|---------|
| **Feature Extraction** | Pass face crops through DINOv3 ViT-S/16; extract the `[CLS]` token embedding (384-d) |
| **Classifier** | Fit an `sklearn.svm.SVC(kernel='rbf')` on the embeddings |
| **Data Collection** | Automatic pipeline using **OpenFace** (RetinaFace detector + AU predictor) to capture face crops and emotion labels from a live webcam at fixed intervals |
| **Inference** | Real-time webcam loop with OpenCV — DINOv3 embedding → SVM prediction per frame |

This approach yields strong accuracy but requires a large vision transformer at inference time, making it impractical for on-device XR deployment. It served as the validation baseline that guided the lightweight model design.

**Notebook**: [`DINOV3+SVM.ipynb`](DINOV3+SVM.ipynb)

### Approach 2 — MobileNetV3-Small (Lightweight)

> **Purpose**: Efficient on-device model suitable for real-time Unity/XR deployment.

We fine-tune a **MobileNetV3-Small** (ImageNet-pretrained) on the [DiffusionEmotion_S](https://huggingface.co/datasets/) dataset, mapping 5 raw emotion categories into 4 classes:

| Raw Label | Mapped Class | Index |
|-----------|-------------|-------|
| happy     | Happy       | 0     |
| sad       | Sad         | 1     |
| angry     | Angry       | 2     |
| fear      | Fear        | 3     |
| disgust   | Fear        | 3     |

**Training Details**:

| Hyperparameter | Value |
|----------------|-------|
| Architecture | `torchvision.models.mobilenet_v3_small` (ImageNet-pretrained) |
| Final layer | `classifier[3] → nn.Linear(1024, 4)` |
| Input size | 224 × 224 × 3 (RGB) |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` |
| Optimizer | AdamW (`lr=3e-4`, `weight_decay=1e-4`) |
| Loss | `CrossEntropyLoss` with inverse-frequency class weights |
| Sampler | `WeightedRandomSampler` to handle class imbalance |
| Data split | 70 / 15 / 15 stratified train / val / test |
| Augmentation | Random horizontal flip, color jitter (brightness=0.2, contrast=0.2, saturation=0.1) |
| Epochs | 10 |
| Batch size | 64 |

**Scripts**: [`onvvinx.py`](onvvinx.py) (CLI) / [`xr-mobilenet.ipynb`](xr-mobilenet.ipynb) (interactive)

---

## ONNX Export for Unity

The trained MobileNetV3-Small checkpoint is exported to ONNX with specific constraints to ensure full compatibility with **Unity's Inference Engine**:

```python
torch.onnx.export(
    model.eval(),
    torch.randn(1, 3, 224, 224, device=device),
    "emotion_mnv3_4cls_unity.onnx",
    export_params   = True,
    opset_version   = 13,          # Unity Inference Engine sweet spot
    do_constant_folding = True,
    input_names     = ["input"],   # NCHW float32
    output_names    = ["logits"],  # (1, 4) raw logits
    dynamo          = False,       # force legacy exporter for stability
    # NO dynamic_axes — Unity prefers fully static shapes
)
```

### Key Export Decisions

| Decision | Rationale |
|----------|-----------|
| **Opset 13** | Maximum operator coverage that Unity Inference Engine supports reliably |
| **Fixed batch = 1** | Avoids dynamic shape issues in Unity's tensor runtime |
| **No `dynamic_axes`** | Unity's model importer handles static shapes most robustly |
| **Legacy exporter** (`dynamo=False`) | Prevents complex graph transformations that may introduce unsupported ops |
| **Constant folding** | Reduces graph complexity and improves inference speed |

Alongside the `.onnx` file, two metadata files are exported for the Unity side:

- **`labels.json`** — class names: `["happy", "sad", "angry", "fear"]`
- **`preprocess.json`** — input size, ImageNet mean/std, layout (`NCHW`), output tensor name

---

## Unity Integration

### Face Detection with BlazeFace

EmoCube uses Google's **BlazeFace short-range** model (`blaze_face_short_range.onnx`) for real-time face detection, running entirely through Unity's Inference Engine.

**Implementation** ([`BlazeFaceOfficialOnQuad.cs`](emocube/Assets/Scripts/BlazeFaceOfficialOnQuad.cs)):

1. **Webcam sampling**: A GPU compute shader (`BlazeUtils.SampleImageAffine`) samples the live webcam texture into a `128×128×3` input tensor using an affine transformation matrix that handles arbitrary aspect ratios.

2. **In-graph preprocessing**: The input is rescaled to `[-1, 1]` directly in the computational graph (`2 * input - 1`), matching BlazeFace's expected range.

3. **NMS in the graph**: Non-Maximum Suppression is compiled into the inference graph itself via `BlazeUtils.NMSFiltering`, avoiding post-processing on the CPU.

4. **Anchor-based decoding**: 896 pre-computed anchor boxes (loaded from [`anchors.csv`](emocube/Assets/Data/anchors.csv)) are used to decode bounding box offsets and 6 facial keypoints (eyes, ears, nose, mouth).

5. **Output**: Normalized bounding box (`Rect` in 0–1 coordinates) and 6 keypoint positions, rendered as overlay lines on the webcam quad via [`QuadOverlayDrawerOfficial.cs`](emocube/Assets/Scripts/QuadOverlayDrawerOfficial.cs).

### Emotion Inference at Runtime

The core emotion classification runs in [`EmotionWebcamCube.cs`](emocube/Assets/Scripts/EmotionWebcamCube.cs):

```
Webcam Frame → Blit to 224×224 → NCHW Tensor (ImageNet norm) → MobileNetV3 ONNX → Softmax → Emotion
```

**Step-by-step**:

1. **Frame capture**: Every `N` seconds, the current webcam frame is blitted to a `224×224` `RenderTexture`, then read back to a CPU `Texture2D`.

2. **Tensor construction**: Pixel data is manually packed into an NCHW `Tensor<float>` with per-channel ImageNet normalization:
   ```
   channel[c, y, x] = (pixel[c] / 255.0 − mean[c]) / std[c]
   ```

3. **Inference**: The tensor is fed to the Unity `Worker` (GPU compute backend) with input name `"input"`, and the output `"logits"` tensor is read back synchronously.

4. **Post-processing**: A numerically stable softmax is applied to the 4-element logit vector; the argmax class is selected. A configurable confidence threshold (`unknownThreshold = 0.55`) gates low-confidence predictions to an `Unknown` state.

5. **Visualization**: The detected emotion maps to a cube material color:

   | Emotion | Color |
   |---------|-------|
   | Happy | Pink `(1.0, 0.4, 0.7)` |
   | Sad | Blue `(0.2, 0.4, 1.0)` |
   | Angry | Red `(1.0, 0.15, 0.15)` |
   | Fear | Black `(0, 0, 0)` |
   | Unknown | Pink (default) |

---

## Demo

<div align="center">

![EmoCube Demo](emocube.gif)

*Real-time emotion recognition driving a 3D cube's color in Unity.*

</div>

---

## Getting Started

### Prerequisites

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| torchvision | 0.15+ |
| ONNX | 1.14+ |
| Unity | 6000.0 (Unity 6) |
| Unity Inference Engine | Built-in (via Package Manager) |

### Training & Export

```bash
# Clone the repository
git clone https://github.com/<your-username>/EmoCube.git
cd EmoCube

# Install Python dependencies
pip install torch torchvision onnx pandas scikit-learn pillow

# Train MobileNetV3-Small and export to ONNX
python onvvinx.py \
    --base_real_path DiffusionEmotion_S/cropped \
    --csv_path DiffusionEmotion_S/dataset_sheet.csv \
    --epochs 10 \
    --out_dir export_out
```

This produces:
```
export_out/
├── best_emotion_mnv3.pth          # PyTorch checkpoint
├── emotion_mnv3_4cls_unity.onnx   # ONNX model for Unity
├── labels.json                    # Class names
└── preprocess.json                # Preprocessing metadata
```

### Unity Setup

1. Open the `emocube/` folder in **Unity 6** (6000.0+).
2. Copy the exported `.onnx`, `labels.json`, and `preprocess.json` into `Assets/Models/`.
3. Ensure the **Inference Engine** package is installed (should be included by default in Unity 6).
4. Open `Assets/Scenes/SampleScene.unity`.
5. In the Inspector, assign:
   - `EmotionWebcamCube` → `onnxModel` field → your emotion ONNX model
   - `BlazeFaceOfficialOnQuad` → `faceDetector` field → `blaze_face_short_range.onnx`
   - `BlazeFaceOfficialOnQuad` → `anchorsCSV` field → `anchors.csv`
6. Press **Play** — the webcam feed will appear, faces will be detected, and the cube will change color based on the recognized emotion.

---

## Project Structure

```
EmoCube/
├── onvvinx.py                     # CLI: train MobileNetV3-Small + export ONNX
├── xr-mobilenet.ipynb             # Interactive training notebook + webcam demos
├── DINOV3+SVM.ipynb               # DINOv3 + SVM baseline experiments
├── README.md
│
└── emocube/                       # Unity 6 project
    ├── Assets/
    │   ├── Scripts/
    │   │   ├── EmotionWebcamCube.cs           # Core: emotion inference → cube color
    │   │   ├── BlazeFaceOfficialOnQuad.cs     # BlazeFace face detection (GPU)
    │   │   ├── BlazeFaceOnQuad.cs             # BlazeFace face detection (CPU fallback)
    │   │   ├── BlazeUtils.cs                  # Shared utilities (affine, NMS, anchors)
    │   │   ├── WebcamManager.cs               # Singleton webcam manager
    │   │   ├── WebcamToMaterial.cs             # Webcam → Quad material
    │   │   ├── QuadOverlayDrawerOfficial.cs   # Face box + keypoint overlay
    │   │   ├── QuadFaceBoxDrawer.cs           # Bounding box renderer
    │   │   └── QuadFaceKeypointDrawer.cs      # Keypoint renderer
    │   ├── Models/
    │   │   ├── blaze_face_short_range.onnx    # BlazeFace detection model
    │   │   ├── emotion_mnv3_4cls_unity.onnx   # MobileNetV3 emotion model
    │   │   ├── labels.json                    # Class definitions
    │   │   └── preprocess.json                # Preprocessing config
    │   ├── Data/
    │   │   └── anchors.csv                    # 896 BlazeFace anchor boxes
    │   └── Scenes/
    │       └── SampleScene.unity
    ├── Packages/
    └── ProjectSettings/
```

---

## Acknowledgements

- [BlazeFace](https://arxiv.org/abs/1907.05047) — Google Research, sub-millisecond face detection
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Howard et al., efficient mobile architectures
- [DINOv3](https://arxiv.org/abs/2304.07193) — Meta AI, self-supervised vision transformers
- [DiffusionEmotion_S](https://huggingface.co/datasets/) — Emotion recognition dataset
- [Unity Inference Engine](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest) — On-device neural network inference
- [ONNX](https://onnx.ai/) — Open Neural Network Exchange format

---

<div align="center">

**Built with PyTorch, ONNX, and Unity 6**

</div>
