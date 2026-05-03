# PROJECT PLAN: Hand Reaching Intention & Position Prediction

**Project Title**: Real-time Anticipatory Perception System for Physical Human-Robot Handshake Interaction  
**Version**: 1.0  
**Last Updated**: May 3, 2026  
**Status**: Implementation Phase – Stage 0 (Data Preparation)

---

## Table of Contents

1. [Project Vision](#project-vision)
2. [Problem Statement](#problem-statement)
3. [Model Architecture](#model-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Training Strategy](#training-strategy)
6. [Preprocessing Specifications](#preprocessing-specifications)
7. [Coordinate Frames & Transformations](#coordinate-frames--transformations)
8. [Module Architecture](#module-architecture)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Evaluation Metrics & Success Criteria](#evaluation-metrics--success-criteria)
11. [Key Design Decisions](#key-design-decisions)
12. [Configuration & Hyperparameters](#configuration--hyperparameters)
13. [References & Citations](#references--citations)

---

## Project Vision

Enable **proactive robot arm pre-positioning** before human hand arrival during physical human-robot interaction (pHRI) by predicting hand reaching intentions and 3D endpoints in real-time using egocentric depth vision.

**Key Innovation**: Achieves >500ms lead time for robot response (sufficient for 200ms mechanical latency + 300ms motion planning).

---

## Problem Statement

### Scenario
A collaborative robot with an egocentric RGB-D camera mounted on its gripper observes a human approaching for a handshake. The system must:

1. **Recognize the human's intention** (Idle/Reaching/Grabbing) from the initial hand motion
2. **Predict the final hand endpoint** in 3D space with <5cm error
3. **Execute a robot response** before the human hand arrives (~1 second prediction horizon)

### Constraints
- **Real-time budget**: <50ms per inference frame (to maintain 30fps processing)
- **Custom data scarcity**: Only ~200 domain-specific clips available for fine-tuning
- **Sensor modality**: Fixed Intel RealSense D435 camera (640×480 @ 30fps, working range 0.5-1.5m)
- **Deployment environment**: Controlled industrial setting with consistent lighting

### Why This Matters
- Current industrial robots react **passively** after hand contact is detected
- Predictive perception enables **safer, more natural** human-robot interaction
- Anticipatory behavior improves **trust and comfort** in collaborative scenarios

---

## Model Architecture

### Design Philosophy

**Primary Method: Depth Map-based Transformer Backbone**

We use raw depth maps rather than skeleton-based approaches because:

| Criterion | Skeleton (skeleton_based) | Depth Maps (depth_based) |
|-----------|--------------------------|-------------------------|
| **Input size** | (T, 21, 3) = 63 dims/frame | (T, H, W, 1) = dense spatial |
| **Preprocessing** | MediaPipe detection (can fail) | Direct sensor output (always available) |
| **Robustness** | Fails on occlusion, motion blur, poor lighting | Works in all conditions + partial occlusion |
| **Domain gap** | MediaPipe pretraining ≠ RealSense deployment | SHREC 2017 depth → RealSense D435 (exact match) |
| **Spatial context** | No background information | Includes scene context for intention inference |
| **SOTA validation** | [TransformerBasedGestureRecognition](https://github.com/aimagelab/TransformerBasedGestureRecognition) | Depth+Transformer (3DV 2020) achieves SOTA on NVGestures |
| **Implementation complexity** | Lower (< MPS) | Moderate (CNN + Transformer) |

**Decision**: Primary = Depth (deployed) | Baseline = Skeleton (for comparison)

### Architecture Overview

```
INPUT (Depth Sequence)
  Shape: (B, T=16, H=64, W=64, C=1)
  - B = batch size
  - T = temporal window at 30fps ≈ 0.53 seconds
  - 64×64 = cropped hand ROI (from full 640×480)
  - Depth range: [0, 1] normalized

           ↓
    
[SHARED BACKBONE — 3D CNN + Transformer Encoder]
  
  ┌─ Conv3D(1→32, 3×3×3) → BN → ReLU → MaxPool
  │  Spatial: 64×64 → 32×32 → 16×16
  │  Temporal: 16 → preserved
  │
  ├─ Conv3D(32→64, 3×3×3) → BN → ReLU → MaxPool
  │  Spatial: 16×16 → 8×8
  │
  ├─ Flatten spatial dims: (B, T, 8×8×64) = (B, 16, 4096)
  │
  ├─ Linear projection: (B, 16, 128)  [d_model=128]
  │
  ├─ Sinusoidal positional encoding: (B, 16, 128)
  │
  ├─ Transformer Encoder:
  │    - Num layers: 2-4
  │    - Num heads: 4
  │    - Dropout: 0.1
  │    - Output: (B, 16, 128)
  │
  └─ Temporal aggregation (mean pool): (B, 128)
  
           ↓
         FEAT
      (B, 128)

           ↓↓↓  —— SPLIT INTO TWO TASK-SPECIFIC HEADS ——
          
     BRANCH A                           BRANCH B
(Intention Classifier)          (3D Position Regressor)

  Linear(128→64)                Linear(128→64)
  ReLU + Dropout(0.2)           ReLU
  Linear(64→3)                  Linear(64→6)
  
  Output: (B, 3)                Output: (B, 6)
  - [idle, reach, grab]         - [x, y, z, n_x, n_y, n_z]
    logits                         meters in camera frame
```

### Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **CNN backbone** | Lightweight 3D CNN (2-3 layers) | Efficient; avoids overfitting on 200 custom samples |
| **Transformer** | 4-head, 2-layer encoder | Captures temporal motion patterns; proven on gesture recognition |
| **d_model** | 128 dimensions | Balance between expressiveness and parameter efficiency |
| **Temporal window** | T=16 frames (0.53s @ 30fps) | Sufficient for ~1s prediction; input dimension manageable |
| **Input resolution** | 64×64 depth ROI | Hand structure visible; CNN receptive field adequate |
| **Pooling strategy** | Mean aggregation | Simpler than max; prevents single-frame dominance |

---

## Data Pipeline

### Data Source Hierarchy

```
STAGE 0: RAW DATA ACQUISITION
│
├─ Primary: SHREC 2017 Hand Gesture Dataset
│  └─ Path: dataset/raw/HandGestureDataset_SHREC2017/
│     ├─ 2,800 RGB-D gesture sequences
│     ├─ 14 gesture classes (hand gestures)
│     ├─ 27 subjects
│     ├─ Intel RealSense SR300 camera (640×480 @ 30fps)
│     ├─ Train split: 2,000 clips (train_gestures.txt)
│     └─ Test split: 800 clips (test_gestures.txt)
│
├─ Secondary: Custom Domain-Specific Dataset
│  └─ ~200 clips (recorded live with RealSense D435)
│     ├─ Idle: 50 clips
│     ├─ Reaching: 100 clips
│     ├─ Grabbing: 50 clips
│     └─ Purpose: Fine-tuning Branches A & B
│
└─ Label Format:
   ├─ Intention labels: {0: Idle, 1: Reaching, 2: Grabbing}
   └─ Position labels: [xyz_meters, normal_vector_3d]
```

### SHREC 2017 Directory Structure

```
dataset/raw/HandGestureDataset_SHREC2017/
├── train_gestures.txt              [manifest: gesture, finger, subject, essai]
├── test_gestures.txt               [manifest: gesture, finger, subject, essai]
├── display_gesture.m               [MATLAB visualization utility]
├── display_sequence.py             [Python visualization utility]
│
└── gesture_{1..14}/                [14 gesture classes]
    └── finger_{1,2}/               [Thumb, Index finger splits]
        └── subject_{1..27}/        [up to 27 subjects]
            ├── essai_1/
            │   ├── skeletons_world.txt    (N_frames × 22 joints × 3 coords)
            │   ├── 0_depth.png ... N_depth.png    (RGB-D depth frames)
            │   └── general_informations.txt       (per-frame bounding boxes)
            ├── essai_2/
            └── ... (up to 5 trials per subject)
```

#### File Format Specifications

**`skeletons_world.txt`** — Hand skeleton trajectory
```
Structure: N_frames × 22 joints × 3 coordinates
           (N, 22, 3) with whitespace-separated values
           
Example (3 frames):
  0.123 0.456 0.789 0.125 0.458 0.791 ... [22 joints × 3 coords]
  0.124 0.457 0.790 0.126 0.459 0.792 ...
  0.125 0.458 0.791 0.127 0.460 0.793 ...

Units: Meters (3D camera frame)
Joints: MediaPipe 21 + 1 extra (typically palm center)
```

**`{i}_depth.png`** — Per-frame depth image
```
Format: 16-bit PNG (single channel)
Dimensions: 640 × 480 pixels
Encoding: Depth in millimeters (0-65535 mm range)
           0 = invalid pixel, 65535 = out-of-range
           
Usage: Load with PIL/OpenCV, normalize to [0,1] by dividing by 3000mm
```

**`general_informations.txt`** — Per-frame bounding boxes
```
Format: N_frames × 4 values (x, y, width, height)
        
Example:
  150 180 120 140
  151 181 121 141
  ...
  
Units: Pixels in 640×480 frame
Purpose: Crop hand ROI before resizing to 64×64
```

### Data Preparation Stages

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 0: DATA PREPARATION (src/data/)                          │
└─────────────────────────────────────────────────────────────────┘

INPUT: dataset/raw/HandGestureDataset_SHREC2017/
       + custom_dataset/ (future)

    ↓↓↓ shrec_io.py (Low-level I/O)

┌──────────────────────────────────────────┐
│ SHREC_DATASET (Class)                    │
├──────────────────────────────────────────┤
│ - parse_manifest(file)                   │
│   → List[(gesture, finger, subject, essai)]│
│                                          │
│ - iter_clips()                           │
│   → Yields per-clip directory paths      │
│                                          │
│ - load_skeleton_world(clip_dir)          │
│   → (N_frames, 22, 3) numpy array       │
│                                          │
│ - load_depth_frames(clip_dir)            │
│   → (N_frames, 480, 640) numpy array    │
│                                          │
│ - load_bboxes(clip_dir)                  │
│   → (N_frames, 4) bounding boxes        │
└──────────────────────────────────────────┘

    ↓↓↓ preprocess_skeleton.py + preprocess_depth.py

┌──────────────────────────────────────────┐
│ NORMALIZED SEQUENCES                     │
├──────────────────────────────────────────┤
│ Per clip:                                │
│  - Root-normalize (wrist-centered)       │
│  - Scale-normalize (hand-size invariant) │
│  - Augment (noise, flip, rotation)       │
│  - Window to 16 frames                   │
└──────────────────────────────────────────┘

    ↓↓↓ dataset.py (PyTorch DataLoader)

OUTPUT: dataset/processed/
        ├── skeleton_train.npz   (M_train, 16, 22, 3)
        ├── skeleton_val.npz
        ├── skeleton_test.npz
        ├── depth_train.npz      (M_train, 16, 64, 64)
        ├── depth_val.npz
        ├── depth_test.npz
        ├── labels_train.npy     (M_train,)
        ├── labels_val.npy
        └── labels_test.npy
```

### Label Definitions

#### Definition 1: Gesture Classification (SHREC 14-class or 83-class for EgoGesture)
```
Classes: {0: Pinch, 1: Swipe left, ..., 13: Ok}
         (14 SHREC gestures or 83 EgoGesture classes)

Purpose: Pretraining backbone on large public dataset
Training set: 2,000 SHREC clips
Test set: 800 SHREC clips
```

#### Definition 2: Intention Classification (Recommended ⭐)
```
Classes: {0: Idle, 1: Reaching, 2: Grabbing}

Label assignment:
  - Idle: Hand at rest, no approach motion
  - Reaching: Hand moving toward robot/target location
  - Grabbing: Hand opened/closing in preparation for grip

Ground truth: Assigned from video annotation during custom data collection
             OR inferred from skeleton trajectory (if reaching trajectory detected)

Training set: ~200 custom clips
             Unbalanced → Use class weights in loss
```

#### Definition 3: 3D Endpoint Position (Recommended ⭐)
```
Target: Palm centroid position in camera frame at gesture end

Extraction (from frame T_final = frame 90 of full sequence):
  palm_landmarks = [skeleton[T_final, joint_i] for i in [0, 5, 9, 13, 17]]
                   # 0=wrist, 5=index_mcp, 9=middle_mcp, 13=ring_mcp, 17=pinky_mcp
  
  position_xyz = mean(palm_landmarks)  # (3,) in meters
  
  # Palm normal via cross product
  P0, P1, P2 = skeleton[T_final, 0], skeleton[T_final, 5], skeleton[T_final, 17]
  v1 = P1 - P0
  v2 = P2 - P0
  normal = cross(v1, v2) / norm(cross(v1, v2))  # unit vector (3,)

Label = [position_xyz, normal_vector]  # stacked: (6,)

Error metric: Euclidean distance in meters
             Target: < 0.05m (5cm) @ 0.5m distance
```

### Training Window Format

```
INPUT WINDOW:
  Frames 0-15 (first 0.53s of gesture)
  
LABEL WINDOW:
  Frame 90 (~3s later, at gesture completion)
  
Prediction horizon: ~2.5 seconds
  (Achieves >500ms robot lead time when inference latency <50ms)

Window extraction algorithm:
  For each full clip (90-120 frames):
    - Extract frames [0:16] as X_input
    - Use frame [90] for y_label
    - If clip too short: skip or zero-pad
    - If clip too long: use all valid 16-frame windows
```

---

## Training Strategy

### 3-Stage Sequential Training

This approach maximizes transfer from public data while avoiding gradient interference.

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: BACKBONE PRETRAINING (2-4 weeks, ~50-100 epochs)      │
├─────────────────────────────────────────────────────────────────┤
│ Dataset: SHREC 2017 depth maps (2,800 clips, 14 classes)       │
│                                                                 │
│ Components:                                                     │
│   ✓ Trainable: CNN (conv3d) + Transformer backbone + head      │
│   ✗ Frozen: None                                               │
│                                                                 │
│ Task: Multi-class gesture classification (14 classes)          │
│ Loss: CrossEntropyLoss(pred_gesture_class, label)              │
│                                                                 │
│ Optimizer: AdamW(lr=1e-3)                                      │
│   Scheduler: CosineAnnealingLR(T_max=50, eta_min=1e-5)         │
│                                                                 │
│ Batch size: 32 (may reduce to 16 on low-VRAM systems)          │
│ Gradient accumulation steps: 2 (effective batch = 64)           │
│                                                                 │
│ Early stopping: patience=10 on validation accuracy             │
│                                                                 │
│ Output artifact: pretrained_backbone.pth                       │
│   - Contains: backbone weights (CNN + Transformer encoder)     │
│   - Discarded: classification head                             │
│                                                                 │
│ Success metric: Top-1 Accuracy ≥ 85%                           │
│   (SOTA on SHREC: 95-97%, we target conservative baseline)     │
│                                                                 │
│ Validation checks:                                             │
│   ✓ 14×14 confusion matrix per class                           │
│   ✓ Per-class precision, recall, F1                            │
│   ✓ Training/validation loss curves smooth                     │
│   ✓ No class-wise underfitting (< 70% for any class)          │
└─────────────────────────────────────────────────────────────────┘

        ↓ (Load pretrained_backbone.pth)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: BRANCH A FINE-TUNE (1-2 weeks, ~30-50 epochs)         │
├─────────────────────────────────────────────────────────────────┤
│ Dataset: Custom data (200 clips: 100 reach, 100 grab, 50 idle) │
│          Split: 70% train (140), 15% val (30), 15% test (30)   │
│                                                                 │
│ Components:                                                     │
│   ✓ Trainable: Branch A only (Linear 128 → 64 → 3)            │
│   ✗ Frozen: Backbone (CNN + Transformer encoder)               │
│                                                                 │
│ Task: Intention classification (3 classes)                     │
│ Loss: CrossEntropyLoss(pred_intention, label)                  │
│       With class weights (if imbalanced):                       │
│         weights = [1.0, 0.5, 0.5]  (reach is main target)     │
│                                                                 │
│ Optimizer: Adam(lr=5e-4)  [Lower LR to preserve backbone]      │
│   Scheduler: Linear warmup → ReduceLROnPlateau                 │
│              factor=0.5, patience=5                            │
│                                                                 │
│ Batch size: 8 (small dataset → prevent overfitting)            │
│ Gradient accumulation: off (small batch sufficient)             │
│                                                                 │
│ Early stopping: patience=15 on validation F1-score             │
│                                                                 │
│ Optional unfreezing (after epoch 10):                           │
│   - Unfreeze last 2 Transformer layers                         │
│   - Lower LR to 1e-4                                           │
│   - Monitor validation loss for instability                    │
│                                                                 │
│ Output artifact: stage2_model.pth (checkpoint before Stage 3)  │
│   - Contains: backbone + Branch A                              │
│                                                                 │
│ Success metrics:                                               │
│   ✓ Top-1 Accuracy ≥ 90%                                       │
│   ✓ Per-class F1 ≥ 0.85                                        │
│   ✓ 3×3 confusion matrix shows no systematic error             │
│                                                                 │
│ Validation checks:                                             │
│   ✓ Per-class precision/recall/F1 via classification_report    │
│   ✓ ROC-AUC per class (one-vs-rest)                            │
│   ✓ t-SNE visualization: clusters separate by intention        │
│   ✓ No class collapse (softmax output entropy > 0.5)           │
└─────────────────────────────────────────────────────────────────┘

        ↓ (Load stage2_model.pth)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: BRANCH B TRAINING (2-3 weeks, ~50-80 epochs)          │
├─────────────────────────────────────────────────────────────────┤
│ Dataset: Custom data (200 clips, same as Stage 2)              │
│          Filtered: Only reach + grab (drop idle)               │
│          Effective: ~150 clips for regression                  │
│          Split: 70% train (105), 15% val (22), 15% test (23)   │
│                                                                 │
│ Components:                                                     │
│   ✓ Trainable: Branch B only (Linear 128 → 64 → 6)            │
│   ✗ Frozen: Backbone + Branch A                                │
│                                                                 │
│ Task: 3D endpoint prediction (position + orientation)          │
│ Loss: SmoothL1Loss(pred_xyz, label_xyz, beta=0.1)              │
│       + MSELoss(pred_normal, label_normal)                     │
│       Total = α * SmoothL1 + (1 - α) * MSE  [α=0.7]            │
│                                                                 │
│       Rationale: SmoothL1 robust to depth noise outliers;      │
│                  MSE keeps orientation predictions smooth      │
│                                                                 │
│ Optimizer: Adam(lr=1e-3)                                       │
│   Scheduler: StepLR(step_size=20, gamma=0.5)                   │
│                                                                 │
│ Batch size: 8                                                  │
│ Gradient accumulation: off                                     │
│                                                                 │
│ Early stopping: patience=20 on validation MAE                  │
│                                                                 │
│ Output artifact: final_model.pth                               │
│   - Contains: backbone + Branch A + Branch B                   │
│   - Ready for: inference in production                         │
│                                                                 │
│ Success metrics:                                               │
│   ✓ Mean Euclidean Error < 0.05m (5cm @ 0.5m distance)         │
│   ✓ Per-axis MAE: < 0.03m (x, y), < 0.07m (z)                 │
│   ✓ Orientation error: cosine similarity > 0.95 with label     │
│                                                                 │
│ Validation checks:                                             │
│   ✓ Plot pred vs label scatter (should be diagonal)            │
│   ✓ Error distribution histogram (check for outliers)          │
│   ✓ Ablation: compare vs Baseline B (frame-16 centroid)        │
│   ✓ Temporal ablation: test with T ∈ {8, 12, 16, 24}         │
│                                                                 │
│ Baseline comparisons:                                          │
│   - Baseline A: Oracle (use actual frame 90 endpoint)          │
│                 Error = 0 (ceiling)                            │
│   - Baseline B: Use frame 16 endpoint (t=0 prediction)         │
│                 Error = typical 0.15-0.30m (floor)             │
│   - Model should: outperform B, approach A                     │
└─────────────────────────────────────────────────────────────────┘

        ↓ (Deploy final_model.pth)

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: VALIDATION & DEPLOYMENT (1-2 weeks)                   │
├─────────────────────────────────────────────────────────────────┤
│ End-to-end system test on robot                                │
│ Measure: Robot lead time before hand arrival                   │
│ Target: > 500ms (proof of anticipatory perception)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Preprocessing Specifications

### Skeleton Preprocessing (`preprocess_skeleton.py`)

**Input**: `(N_frames, 22, 3)` raw hand skeleton from `skeletons_world.txt`

```python
def preprocess_skeleton(skeleton, augment=True):
    """
    Normalize & augment hand skeleton for training.
    
    Args:
        skeleton: (N, 22, 3) – raw 3D keypoints in camera frame
        augment: bool – apply data augmentation
    
    Returns:
        skeleton_normalized: (N, 22, 3) – ready for model input
    """
    
    # Step 1: Root-normalize (translation invariance)
    #   Problem: Wrist distance from camera varies
    #   Solution: Subtract wrist (joint 0) position
    wrist = skeleton[:, 0, :]  # (N, 3)
    skeleton = skeleton - wrist[:, None, :]  # broadcast: (N, 1, 3)
    
    # Step 2: Scale-normalize (hand-size invariance)
    #   Problem: Hand size varies (8cm → 12cm depending on age, gender)
    #   Solution: Divide by max pairwise joint distance
    N = len(skeleton)
    for i in range(N):
        frame = skeleton[i]  # (22, 3)
        distances = []
        for j1 in range(22):
            for j2 in range(j1+1, 22):
                dist = np.linalg.norm(frame[j1] - frame[j2])
                distances.append(dist)
        max_dist = np.max(distances)
        skeleton[i] /= (max_dist + 1e-8)  # avoid division by zero
    
    # Step 3: Window to 16 frames
    #   Problem: SHREC sequences vary in length (60-120 frames)
    #   Solution: Extract first 16 frames, zero-pad if shorter
    if len(skeleton) >= 16:
        skeleton = skeleton[:16]
    else:
        pad_length = 16 - len(skeleton)
        skeleton = np.vstack([skeleton, np.zeros((pad_length, 22, 3))])
    
    # Step 4: Data augmentation (if training)
    if augment:
        # Gaussian noise on joint positions (σ=0.01)
        #   Simulates RealSense joint tracking jitter (~1cm)
        noise = np.random.normal(0, 0.01, skeleton.shape)
        skeleton = skeleton + noise
        
        # Horizontal flip (p=0.5)
        #   Simulates left/right hand symmetry & camera pose variation
        if np.random.rand() < 0.5:
            skeleton[:, :, 0] *= -1  # flip X coordinate
        
        # Y-axis rotation ±15° (p=1.0, random angle)
        #   Simulates camera viewing angle variation
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        skeleton = skeleton @ rotation_matrix.T  # apply to all frames & joints
        
        # Temporal dropout (p=0.1)
        #   Simulates momentary MediaPipe tracking loss
        if np.random.rand() < 0.1:
            drop_frame_idx = np.random.randint(0, 16)
            skeleton[drop_frame_idx] = 0  # zero out entire frame
    
    return skeleton  # (16, 22, 3)
```

### Depth Preprocessing (`preprocess_depth.py`)

**Input**: `(N_frames, 480, 640)` raw depth frames from `{i}_depth.png` files + bounding boxes from `general_informations.txt`

```python
def preprocess_depth(depth_frames, bboxes, augment=True):
    """
    Crop, resize, normalize & augment depth frames.
    
    Args:
        depth_frames: (N, 480, 640) – raw 16-bit depth images
        bboxes: (N, 4) – per-frame bounding boxes [x, y, w, h]
        augment: bool – apply data augmentation
    
    Returns:
        depth_normalized: (N, 64, 64) – ready for CNN input
    """
    
    # Step 0: Validate and clip out-of-range values
    #   RealSense D435 valid range: 0-3000mm
    #   0 = invalid pixel (hole), 65535 = out-of-range
    DEPTH_MAX_MM = 3000
    depth_frames = np.clip(depth_frames, 0, DEPTH_MAX_MM)
    
    # Step 1: Crop hand ROI per frame
    #   Input: full 640×480 frame (95% is background)
    #   Output: ~150×150 ROI containing hand
    N = len(depth_frames)
    depth_cropped = []
    
    for i in range(N):
        x, y, w, h = bboxes[i]
        
        # Add 15% padding to avoid cropping fingers at edges
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(640, x + w + pad_x)
        y2 = min(480, y + h + pad_y)
        
        roi = depth_frames[i, y1:y2, x1:x2]  # crop
        depth_cropped.append(roi)
    
    # Step 2: Resize all ROIs to 64×64
    #   Justification: 64×64 preserves finger silhouettes
    #                  (32×32 too small, 112×112 too memory-hungry)
    from PIL import Image
    
    depth_resized = []
    for roi in depth_cropped:
        # PIL expects (H, W) for Image.fromarray
        pil_img = Image.fromarray(roi.astype(np.uint16))
        resized = pil_img.resize((64, 64), Image.BILINEAR)
        depth_resized.append(np.array(resized))
    
    depth_resized = np.array(depth_resized)  # (N, 64, 64)
    
    # Step 3: Normalize to [0, 1]
    #   Converting mm → normalized depth
    depth_normalized = depth_resized.astype(np.float32) / DEPTH_MAX_MM
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Step 4: Window to 16 frames
    if len(depth_normalized) >= 16:
        depth_normalized = depth_normalized[:16]
    else:
        pad_length = 16 - len(depth_normalized)
        depth_normalized = np.vstack([
            depth_normalized,
            np.zeros((pad_length, 64, 64))
        ])
    
    # Step 5: Data augmentation (if training)
    if augment:
        # Depth noise injection (σ ≈ 15mm = 0.005 normalized)
        #   Simulates RealSense sensor noise at arm distance
        noise = np.random.normal(0, 0.005, depth_normalized.shape)
        depth_normalized = np.clip(depth_normalized + noise, 0, 1)
        
        # Brightness jitter (±5% of max depth)
        #   Simulates distance variation (50cm → 60cm)
        scale = np.random.uniform(0.95, 1.05)
        depth_normalized = np.clip(depth_normalized * scale, 0, 1)
        
        # Spatial jitter (±4 pixels shift)
        #   Simulates camera vibration from robot motion
        shift_x = np.random.randint(-4, 5)
        shift_y = np.random.randint(-4, 5)
        depth_normalized = np.roll(depth_normalized, (shift_x, shift_y), axis=(1, 2))
        
        # Temporal dropout (p=0.1)
        #   Simulates dropped frames
        if np.random.rand() < 0.1:
            drop_frame_idx = np.random.randint(0, 16)
            depth_normalized[drop_frame_idx] = 0
        
        # Horizontal flip (p=0.5)
        if np.random.rand() < 0.5:
            depth_normalized = np.flip(depth_normalized, axis=2)
    
    return depth_normalized  # (16, 64, 64), float32, [0, 1]
```

### Augmentation Strategy Summary

Both modalities use complementary augmentation to simulate deployment variability:

| Augmentation | Skeleton | Depth | Rationale |
|---|---|---|---|
| Gaussian noise (σ) | 0.01m (~1cm) | 0.005 (~15mm) | Sensor tracking jitter |
| Horizontal flip | ✓ | ✓ | Left/right hand symmetry |
| Rotation ±15° | Y-axis rotation | N/A | Camera angle variation |
| Brightness jitter | N/A | ±5% | Subject-to-camera distance |
| Spatial jitter | N/A | ±4px | Robot arm vibration |
| Temporal dropout | ✓ | ✓ | Momentary detection loss |

---

## Coordinate Frames & Transformations

### Frame Definitions

```
┌────────────────────────────────────────────┐
│ FRAME 0: MediaPipe Normalized              │
├────────────────────────────────────────────┤
│ Range: [0, 1] × [0, 1]                    │
│ Origin: Top-left of image                 │
│ Usage: MediaPipe hand detection output    │
│                                           │
│      (0,0) ──────────────→ (1,0)         │
│        │                                  │
│        │      Hand                        │
│        ▼                                  │
│      (0,1) ──────────────→ (1,1)         │
│                                           │
│ Joints: [0: wrist, 1-4: thumb,           │
│          5-8: index, 9-12: middle, ...]   │
└────────────────────────────────────────────┘

        ↓ Project via camera intrinsics

┌────────────────────────────────────────────┐
│ FRAME 1: Camera 3D (RealSense)             │
├────────────────────────────────────────────┤
│ Origin: Optics center of depth sensor    │
│ X-axis: Rightward in image               │
│ Y-axis: Downward in image                │
│ Z-axis: Outward from camera (depth)      │
│                                           │
│ Units: Meters (m)                        │
│ Range: [0.5m, 1.5m] typical              │
│                                           │
│    Y (down)                               │
│    ↑                                      │
│    │     Z (toward camera)                │
│    └──→  ↓                                │
│   X (right)                               │
│                                           │
│ Projection formula:                      │
│   X_cam = (u_px - cx) * depth / fx       │
│   Y_cam = (v_px - cy) * depth / fy       │
│   Z_cam = depth                          │
│                                           │
│ where:                                    │
│   (u_px, v_px) = pixel coords in image   │
│   (cx, cy) = principal point (optical center) │
│   (fx, fy) = focal lengths in pixels     │
│   depth = per-pixel depth from sensor    │
│                                           │
│ Example (RealSense D435 intrinsics):     │
│   fx = 615.2, fy = 614.8                │
│   cx = 320.0, cy = 240.0                │
│   Resolution: 640×480                    │
└────────────────────────────────────────────┘

        ↓ Apply extrinsic calibration matrix

┌────────────────────────────────────────────┐
│ FRAME 2: Robot Base Frame                  │
├────────────────────────────────────────────┤
│ Origin: Robot base (UR, ABB, etc.)       │
│ X-axis: Forward (along table)            │
│ Y-axis: Leftward from robot perspective  │
│ Z-axis: Upward (vertical)                │
│                                           │
│ Units: Meters (m)                        │
│                                           │
│ Transformation:                          │
│   P_robot = T_cam_to_robot @ P_camera    │
│                                           │
│   where:                                  │
│   T_cam_to_robot = [R | t]  (4×4 SE(3)) │
│                    [0 | 1]               │
│                                           │
│   R = 3×3 rotation matrix                 │
│   t = 3×1 translation vector (m)         │
│                                           │
│ Calibration: ArUco board + PnP solver    │
│   - Mount ArUco pattern at known pose    │
│   - Detect in camera frame               │
│   - Compute T_cam_to_robot once offline  │
│   - Cache in robot config file           │
└────────────────────────────────────────────┘
```

### 2D→3D Projection Algorithm

```python
def project_2d_to_3d(landmarks_2d, depth_frame, intrinsics):
    """
    Convert MediaPipe 2D landmarks + RealSense depth → 3D world coords.
    
    Args:
        landmarks_2d: (21, 2) – normalized [0, 1] × [0, 1]
        depth_frame: (480, 640) – depth in millimeters
        intrinsics: dict – {fx, fy, cx, cy}
    
    Returns:
        landmarks_3d: (21, 3) – 3D points in camera frame (meters)
    """
    
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    img_width, img_height = 640, 480
    
    landmarks_3d = []
    
    for i, (x_norm, y_norm) in enumerate(landmarks_2d):
        # Denormalize to pixel coordinates
        u_px = x_norm * img_width
        v_px = y_norm * img_height
        
        # Get depth at this pixel
        u_int = int(np.round(u_px))
        v_int = int(np.round(v_px))
        
        # Clamp to image bounds
        u_int = np.clip(u_int, 0, img_width - 1)
        v_int = np.clip(v_int, 0, img_height - 1)
        
        d_mm = depth_frame[v_int, u_int]  # depth in mm
        
        # Convert to meters
        d = d_mm / 1000.0
        
        # Handle invalid depth
        if d == 0 or d > 3.0:  # out of range
            # Use interpolation or previous frame
            d = np.nan
        
        # Project to 3D (camera frame)
        X = (u_px - cx) * d / fx
        Y = (v_px - cy) * d / fy
        Z = d
        
        landmarks_3d.append([X, Y, Z])
    
    return np.array(landmarks_3d)  # (21, 3) in meters
```

### Extrinsic Calibration Procedure

```python
def calibrate_camera_to_robot(aruco_poses_camera, aruco_poses_robot):
    """
    Compute T_cam_to_robot using ArUco board calibration.
    
    Procedure:
    1. Mount ArUco board at known robot pose
    2. Capture from camera, detect ArUco markers
    3. Estimate camera pose relative to ArUco
    4. Solve: T_cam_to_robot = T_cam_to_aruco @ T_aruco_to_robot^-1
    
    Args:
        aruco_poses_camera: (4, 4) – ArUco pose in camera frame (from cv2.solvePnP)
        aruco_poses_robot: (4, 4) – ArUco pose in robot frame (from calibration board placement)
    
    Returns:
        T_cam_to_robot: (4, 4) SE(3) transformation matrix
    """
    
    import cv2
    
    # T_cam_to_robot = T_cam_to_aruco @ inv(T_aruco_to_robot)
    T_cam_to_aruco = aruco_poses_camera
    T_aruco_to_robot = aruco_poses_robot
    T_robot_to_aruco = np.linalg.inv(T_aruco_to_robot)
    
    T_cam_to_robot = T_cam_to_aruco @ T_robot_to_aruco
    
    return T_cam_to_robot  # (4, 4)

def apply_transform(point_3d_camera, T_cam_to_robot):
    """Apply transformation during inference."""
    p_homogeneous = np.append(point_3d_camera, 1)  # (4,)
    p_robot = T_cam_to_robot @ p_homogeneous  # (4,)
    return p_robot[:3]  # (3,)
```

---

## Module Architecture

### File Organization

```
comp4471-project/
├── PROJECT_PLAN.md                     [← This file — single source of truth]
├── README.md                           [Setup instructions]
├── pyproject.toml                      [Dependencies: PyTorch, torchvision, etc.]
├── environment_check.ipynb             [Existing: verify GPU/environment]
│
├── src/
│   ├── __init__.py
│   ├── device.py                       [Existing: CUDA/MPS/CPU device selection]
│   │
│   ├── config.py                       [NEW: Centralized configuration]
│   │   - SHREC_RAW_DIR
│   │   - CUSTOM_DATA_DIR
│   │   - PROCESSED_DATA_DIR
│   │   - Model hyperparameters (d_model, num_layers, etc.)
│   │   - Training hyperparameters (lr, batch_size, etc.)
│   │   - Evaluation thresholds
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── shrec_io.py                [NEW: SHREC 2017 low-level I/O]
│   │   │   - SHREC_DATASET class
│   │   │   - parse_manifest()
│   │   │   - iter_clips()
│   │   │   - load_skeleton_world()
│   │   │   - load_depth_frames()
│   │   │   - load_bboxes()
│   │   │
│   │   ├── preprocess_skeleton.py     [NEW: Skeleton normalization & augmentation]
│   │   │   - preprocess_skeleton()
│   │   │   - SkeletonProcessor class
│   │   │
│   │   ├── preprocess_depth.py        [NEW: Depth cropping, resizing, augmentation]
│   │   │   - preprocess_depth()
│   │   │   - DepthProcessor class
│   │   │
│   │   └── dataset.py                 [NEW: PyTorch DataLoaders]
│   │       - SkeletonDataset class
│   │       - DepthDataset class
│   │       - create_dataloaders()
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                [NEW: Shared encoder]
│   │   │   - DepthCNN3DEncoder
│   │   │   - TransformerEncoder
│   │   │   - SkeletonTransformer (baseline)
│   │   │
│   │   ├── heads.py                   [NEW: Task-specific heads]
│   │   │   - IntentionHead (Branch A)
│   │   │   - PositionHead (Branch B)
│   │   │
│   │   └── model.py                   [NEW: Full architecture]
│   │       - HandReachingModel (multi-task)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── stage1_pretraining.py      [NEW: SHREC backbone training]
│   │   ├── stage2_intention.py        [NEW: Branch A fine-tuning]
│   │   ├── stage3_position.py         [NEW: Branch B training]
│   │   └── utils.py                   [Logging, checkpointing, etc.]
│   │
│   └── inference/
│       ├── __init__.py
│       ├── real_time_loop.py          [NEW: Inference pipeline]
│       ├── camera_interface.py        [NEW: RealSense camera wrapper]
│       └── ros_publisher.py           [NEW: optional – ROS integration]
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      [NEW: SHREC loading & visualization]
│   ├── 02_eda_skeleton.ipynb          [NEW: Exploratory data analysis]
│   ├── 03_eda_depth.ipynb             [NEW: Depth map visualization]
│   ├── 10_stage1_training.ipynb       [NEW: Backbone pretraining]
│   ├── 20_stage2_training.ipynb       [NEW: Branch A fine-tuning]
│   ├── 30_stage3_training.ipynb       [NEW: Branch B training]
│   └── 40_evaluation.ipynb            [NEW: End-to-end evaluation]
│
├── dataset/
│   ├── raw/
│   │   └── HandGestureDataset_SHREC2017/   [Existing: SHREC data]
│   │       ├── train_gestures.txt
│   │       ├── test_gestures.txt
│   │       └── gesture_{1..14}/finger_{1,2}/subject_{1..27}/...
│   │
│   └── processed/                    [NEW: Preprocessed .npz files]
│       ├── skeleton_train.npz
│       ├── skeleton_val.npz
│       ├── skeleton_test.npz
│       ├── depth_train.npz
│       ├── depth_val.npz
│       ├── depth_test.npz
│       ├── labels_train.npy
│       ├── labels_val.npy
│       └── labels_test.npy
│
└── models/                            [NEW: Trained checkpoints]
    ├── stage1_pretrained_backbone.pth
    ├── stage2_intention_model.pth
    └── final_model.pth
```

### Module Dependencies & Imports

```
config.py
  ↓ (imports nothing from src/)

shrec_io.py
  ← depends on: config
  
preprocess_skeleton.py
  ← depends on: config, numpy, PIL, sklearn (for preprocessing)
  
preprocess_depth.py
  ← depends on: config, numpy, PIL, scipy (for bilateral filtering)
  
dataset.py
  ← depends on: config, preprocess_skeleton, preprocess_depth, torch, numpy

backbone.py
  ← depends on: config, torch, torch.nn

heads.py
  ← depends on: config, torch, torch.nn

model.py
  ← depends on: backbone, heads, torch

stage1_pretraining.py
  ← depends on: model, dataset, device, torch, config

stage2_intention.py
  ← depends on: model, dataset, device, torch, config

stage3_position.py
  ← depends on: model, dataset, device, torch, config

real_time_loop.py
  ← depends on: model, camera_interface, device, torch, numpy

camera_interface.py
  ← depends on: pyrealsense2, numpy, config

Notebook imports:
  01_data_preparation.ipynb
    ← shrec_io, matplotlib, numpy, PIL
  
  10_stage1_training.ipynb
    ← model, dataset, training.stage1_pretraining, config
```

---

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1-2)

**Milestone 1.1: Configuration System**
- [ ] Create `src/config.py` with all hardcoded constants
  - SHREC path, processed data path, model hyperparams, training hyperparams
  - Reference: Single source of truth for all parameters
  - Validation: Run `config._validate()` to check all paths exist

**Milestone 1.2: SHREC I/O Layer**
- [ ] Implement `src/data/shrec_io.py`
  - `SHREC_DATASET` class with `iter_clips()` generator
  - `load_skeleton_world()`, `load_depth_frames()`, `load_bboxes()`
  - Validation: Can iterate 100 random clips without errors
  - Test: `python -m src.data.shrec_io`

**Milestone 1.3: Data Preparation Notebooks**
- [ ] Create `notebooks/01_data_preparation.ipynb`
  - Load 10 clips, visualize skeleton + depth overlay
  - Validate coordinate projections
  - Reference: Explore SHREC structure

### Phase 2: Data Processing (Week 2-3)

**Milestone 2.1: Preprocessing Modules**
- [ ] Implement `src/data/preprocess_skeleton.py`
  - Root normalization + scale normalization + windowing
  - Augmentation (noise, flip, rotation, dropout)
  - Validation: Check output shape (16, 22, 3), values in [-1, 1]

- [ ] Implement `src/data/preprocess_depth.py`
  - ROI cropping, resizing to 64×64, normalization
  - Augmentation (noise, jitter, brightness, dropout)
  - Validation: Output shape (16, 64, 64), values in [0, 1]

**Milestone 2.2: PyTorch Dataset Wrappers**
- [ ] Implement `src/data/dataset.py`
  - `SkeletonDataset`, `DepthDataset` classes
  - Lazy loading from .npz files
  - `create_dataloaders()` factory
  - Validation: DataLoader yields batches of correct shape

**Milestone 2.3: Batch Processing**
- [ ] Generate processed datasets (train/val/test splits)
  - SHREC: 2000 train / 800 test
  - Custom: 140 train / 30 val / 30 test (future)
  - Output: `.npz` files in `dataset/processed/`
  - Validation: Disk space ~10GB total, checksums logged

### Phase 3: Model Architecture (Week 3-4)

**Milestone 3.1: Backbone Network**
- [ ] Implement `src/models/backbone.py`
  - `DepthCNN3DEncoder`: Conv3D layers for spatial feature extraction
  - `TransformerEncoder`: MultiheadAttention + feedforward
  - `SkeletonTransformer`: Baseline (no CNN, direct Transformer)
  - Validation: Forward pass with dummy input (B=2, T=16, H=64, W=64)

**Milestone 3.2: Task-Specific Heads**
- [ ] Implement `src/models/heads.py`
  - `IntentionHead`: Linear 128→64→3
  - `PositionHead`: Linear 128→64→6
  - Validation: Output shapes correct, logits/predictions reasonable

**Milestone 3.3: Multi-Task Model**
- [ ] Implement `src/models/model.py`
  - `HandReachingModel` class with `.forward()` → (intention_logits, position_pred)
  - State dict management for freezing/unfreezing
  - Validation: Reproducible outputs, parameter count < 10M

### Phase 4: Stage 1 Pretraining (Week 4-6)

**Milestone 4.1: Training Loop & Logging**
- [ ] Implement `src/training/stage1_pretraining.py`
  - Training loop with loss computation, backward pass, optimizer step
  - Validation loop with metrics (accuracy, loss)
  - TensorBoard logging, checkpoint saving
  - Early stopping logic
  - Validation: Runs for 5 epochs without error, logs appear in tensorboard

**Milestone 4.2: Experimentation Notebook**
- [ ] Create `notebooks/10_stage1_training.ipynb`
  - Train on SHREC for 50 epochs
  - Plot train/val loss curves
  - Generate 14×14 confusion matrix
  - Target: ≥85% top-1 accuracy
  - Reference: Baseline to beat with custom data

**Deliverable**: `models/stage1_pretrained_backbone.pth` (backbone weights only)

### Phase 5: Stage 2 Fine-tuning (Week 6-7)

**Milestone 5.1: Branch A Fine-tuning Loop**
- [ ] Implement `src/training/stage2_intention.py`
  - Load pretrained backbone from Stage 1
  - Freeze backbone, train only Branch A head
  - Optional: Unfreeze after epoch 10
  - Loss: CrossEntropyLoss with class weights
  - Validation: Notebook runs for 30 epochs

**Milestone 5.2: Evaluation Notebook**
- [ ] Create `notebooks/20_stage2_training.ipynb`
  - Train on custom data (when available)
  - Plot 3×3 confusion matrix
  - Calculate per-class F1, precision, recall
  - t-SNE visualization of features
  - Target: ≥90% top-1 accuracy

**Deliverable**: `models/stage2_intention_model.pth` (backbone + Branch A)

### Phase 6: Stage 3 Position Regression (Week 7-8)

**Milestone 6.1: Branch B Training Loop**
- [ ] Implement `src/training/stage3_position.py`
  - Load Stage 2 model
  - Freeze backbone + Branch A, train only Branch B
  - Loss: SmoothL1 + MSE (orientation)
  - Validator: Euclidean error in meters

**Milestone 6.2: Regression Evaluation**
- [ ] Create `notebooks/30_stage3_training.ipynb`
  - Train for 50 epochs
  - Plot pred vs label scatter (check if diagonal)
  - Compute MAE per axis (x, y, z)
  - Compare vs Baseline B (frame-16 centroid)
  - Temporal ablation: test T ∈ {8, 12, 16, 24}
  - Target: <5cm MEE, outperform Baseline B by >50%

**Deliverable**: `models/final_model.pth` (full model)

### Phase 7: Evaluation & Deployment (Week 8-9)

**Milestone 7.1: Real-Time Inference**
- [ ] Implement `src/inference/real_time_loop.py`
  - Rolling buffer for 16-frame window
  - Model.eval() + no_grad() inference
  - Latency profiling (<50ms target)
  - Publish predictions (intention + endpoint)

- [ ] Implement `src/inference/camera_interface.py`
  - RealSense D435 frame capture
  - Camera intrinsics loading
  - Depth frame preprocessing on-the-fly

**Milestone 7.2: End-to-End Evaluation**
- [ ] Create `notebooks/40_evaluation.ipynb`
  - Load final_model.pth
  - Test on held-out test set
  - Generate all evaluation metrics
  - Benchmark: robot lead time demo

**Deliverable**: End-to-end system ready for deployment

---

## Evaluation Metrics & Success Criteria

### Stage 1: Backbone Pretraining

| Metric | Target | How to Compute |
|--------|--------|---|
| **Top-1 Accuracy** | ≥85% | `accuracy_score(y_true, y_pred)` |
| **Macro-Averaged F1** | ≥0.80 | `f1_score(y_true, y_pred, average='macro')` |
| **Per-class Accuracy** | >70% each | Min accuracy across 14 classes |
| **Confusion Matrix** | Diagonal dominant | No systematic off-diagonal pattern |

**Validation checks**:
- Loss curves: smooth, no sudden spikes
- No class collapse: all classes predicted with >1% frequency
- Reproducibility: same model loaded twice gives identical outputs

### Stage 2: Intention Classification

| Metric | Target | How to Compute |
|--------|--------|---|
| **Top-1 Accuracy** | ≥90% | `accuracy_score(y_true, y_pred)` |
| **Macro F1** | ≥0.85 | `f1_score(..., average='macro')` |
| **Per-class F1** | ≥0.85 each | Must hold for all 3 classes |
| **Reaching Recall** | ≥0.95 | TP/(TP+FN) for reaching class — critical for robot response |
| **3×3 Confusion Matrix** | Minimal off-diagonal | Acceptable: ≤ 10% misclassification |

**Validation checks**:
- ROC-AUC: >0.95 for each class (one-vs-rest)
- Invalid output check: No NaN/Inf during inference
- Latency: <50ms per batch on deployment hardware

### Stage 3: 3D Position Regression

| Metric | Target | How to Compute |
|--------|--------|---|
| **Mean Euclidean Error** | <0.05m @ 0.5m | `mean(norm(pred_xyz - label_xyz, axis=1))` |
| **MAE (X-axis)** | <0.03m | `mean(abs(pred_x - label_x))` |
| **MAE (Y-axis)** | <0.03m | `mean(abs(pred_y - label_y))` |
| **MAE (Z-axis)** | <0.07m | Depth error typically larger |
| **Orientation Error** | CosSim >0.95 | `mean(cosine_similarity(pred_normal, label_normal))` |

**Baseline comparisons**:
- **Baseline A (Oracle)**: Use actual frame-90 endpoint → Error = 0
- **Baseline B (Naive)**: Use frame-16 endpoint → Error = ~0.15-0.30m
- **Model**: Should achieve `Error_model << Error_B`

**Ablation studies**:
- Temporal window: Accuracy vs T ∈ {8, 12, 16, 24}
  - Expect: Error decreases as T increases (proof of temporal usage)
- Architecture: Depth-CNN vs Skeleton-Transformer
  - Expect: Depth more robust to occlusion

**Validation checks**:
- Scatter plot (pred vs label): approximately diagonal (y=x line)
- Error distribution: few outliers (>3σ), mostly <0.1m
- Per-class: reaching vs grabbing error difference <20%

### Stage 4: Robot Demo

| Metric | Target | Meaning |
|--------|--------|---------|
| **Robot Lead Time** | >500ms | Robot starts motion before hand arrives |
| **Hand Arrival Error** | <5cm | End-effector within grasp distance |
| **Success Rate** | ≥80% over 10 trials | Reproducibility |

---

## Key Design Decisions

### 1. Depth Maps vs Skeleton Keypoints

**Decision**: Primary = Depth, Baseline = Skeleton

| Factor | Depth | Skeleton |
|--------|-------|----------|
| **Robustness** | ✓ Works in occlusion, blur, poor lighting | ✗ Fails when tracked joints are occluded |
| **Domain gap** | ✓ SHREC depth → RealSense D435 (direct) | ✗ MediaPipe training ≠ MediaPipe inference modalities |
| **Information richness** | ✓ Spatial context from background | ✗ 63-dim features per frame (sparse) |
| **Computational cost** | ⚠ CNN preprocessing needed | ✓ Lower MPS (21 joints vs 64×64 pixels) |
| **Data requirements** | ⚠ Needs more training samples | ✓ Smaller feature space |
| **SOTA validation** | ✓ [Mercier et al., 3DV 2020] achieved SOTA | ✓ Standard baseline in skeleton-based recognition |

**Rationale**: Your deployment uses a fixed RealSense camera in controlled lighting. Training on depth maps eliminates a failure mode (MediaPipe detection) and ensures exact input-output modality matching.

### 2. Sequential 3-Stage Training vs Joint Multi-Task

**Decision**: Sequential stages (1 → 2 → 3)

| Aspect | Sequential | Joint Multi-Task |
|--------|-----------|------------------|
| **Gradient interference** | ✗ None (stages frozen) | ✓ Potential conflicts between intention & position losses |
| **Public dataset usage** | ✓ Full SHREC pretraining on 14 classes | ✗ Limited to 3-class + regression |
| **Customization flexibility** | ✓ Can use different optimizers per stage | ✗ Single optimization strategy |
| **Debugging** | ✓ Easy (isolate which stage breaks) | ✗ Hard (entangled losses) |
| **Empirical success rate** | ✓ Proven in transfer learning literature | ✓ Also works, but less stable with small custom data |

**Rationale**: You have 2,800 SHREC clips but only 200 custom clips. Sequential training maximizes transfer from public data. Stage 1 learns generalizable spatio-temporal patterns (gesture recognition). Stages 2-3 specialize to your 3-class task without forgetting.

### 3. Input Resolution: 64×64 Depth ROI

**Decision**: Crop hand ROI to 64×64

| Resolution | VRAM/Sample | Hand Details | Used |
|---|---|---|---|
| 32×32 | 0.06 MB | Finger boundaries blurred | Too small |
| **64×64** | **0.48 MB** | **Finger silhouettes clear** | **✓ Selected** |
| 112×112 | 1.5 MB | Excessive context (mostly background) | Too large |
| 224×224 | 6.0 MB | ImageNet-scale, 12× VRAM overhead | Overkill |

**Rationale**: At 64×64, the hand occupies ~70% of the frame (sufficient to preserve structure), and a batch of 32 fits in <1GB VRAM for inference.

### 4. Loss Function for Position Regression

**Decision**: SmoothL1 + MSE hybrid

```python
Loss = 0.7 * SmoothL1Loss(pred_xyz, label_xyz, beta=0.1)
     + 0.3 * MSELoss(pred_normal, label_normal)
```

| Loss | Property | Use Case |
|-----|----------|----------|
| **SmoothL1** | Robust to outliers (depth noise >15cm) | Position branch (sensitive to sensor noise) |
| **MSE** | Smooth gradients | Orientation branch (fine-grained alignment) |
| **Weights** | 70/30 split | Position more important than orientation for grasping |

**Rationale**: RealSense D435 depth error is heteroscedastic (~±1mm @ 0.5m, but +15mm @ 1.5m). SmoothL1 remains stable with depth noise outliers. MSE keeps orientation predictions from drifting.

### 5. Temporal Window: T=16 Frames

**Decision**: Fixed 16 frames at 30fps = 0.53 seconds

| Window | Pros | Cons |
|--------|------|------|
| T=8 (0.27s) | Fast inference | Insufficient motion context |
| **T=16 (0.53s)** | **Sweet spot: >500ms lead time, CNN + Transformer effective** | **Moderate latency** |
| T=24 (0.80s) | More context | Delayed response by 267ms |
| T=32 (1.07s) | Full gesture | Too slow for real-time response |

**Rationale**: At T=16, the model sees ~half a second of motion (enough to detect reaching vs grabbing), and predicts 1 second ahead (frame 90/30fps ≈ 3s into gesture, minus 0.53s input window = 2.5s prediction). This gives robot >500ms lead time.

---

## Configuration & Hyperparameters

### Configuration Management

**File**: `src/config.py` — Single source of truth

```python
# ===== Paths =====
SHREC_RAW_DIR = r"D:\course\comp4471\comp4471-project\dataset\raw\HandGestureDataset_SHREC2017"
CUSTOM_DATA_DIR = r"D:\course\comp4471\comp4471-project\dataset\custom"  # (future)
PROCESSED_DATA_DIR = r"D:\course\comp4471\comp4471-project\dataset\processed"
MODELS_DIR = r"D:\course\comp4471\comp4471-project\models"
LOGS_DIR = r"D:\course\comp4471\comp4471-project\logs"

# ===== Data =====
DEPTH_MAX_MM = 3000  # RealSense max valid range
DEPTH_NORMALIZED_MAX = 1.0  # Depth normalization range
IMG_WIDTH, IMG_HEIGHT = 640, 480  # RealSense resolution
SKELETON_NUM_JOINTS = 22  # SHREC format
TEMPORAL_WINDOW = 16  # Frames per sequence @ 30fps
DEPTH_ROI_SIZE = 64  # Cropped depth frame size (64×64)

# ===== Class Labels =====
SHREC_GESTURE_CLASSES = {
    0: "Pinch", 1: "Swipe left", ..., 13: "Ok"
}  # 14 classes
INTENTION_CLASSES = {
    0: "Idle", 1: "Reaching", 2: "Grabbing"
}  # 3 classes

# ===== Model Hyperparameters =====
D_MODEL = 128  # Transformer embedding dimension
NUM_TRANSFORMER_LAYERS = 2  # Encoder depth
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1
FEEDFORWARD_DIM = 512  # MLP hidden dimension

# ===== Stage 1: Pretraining =====
STAGE1_LEARNING_RATE = 1e-3
STAGE1_BATCH_SIZE = 32
STAGE1_NUM_EPOCHS = 50
STAGE1_GRADIENT_ACCUMULATION_STEPS = 2
STAGE1_EARLY_STOPPING_PATIENCE = 10

# ===== Stage 2: Intention =====
STAGE2_LEARNING_RATE = 5e-4
STAGE2_BATCH_SIZE = 8
STAGE2_NUM_EPOCHS = 30
STAGE2_UNFREEZE_EPOCH = 10
STAGE2_UNFREEZE_LR = 1e-4
STAGE2_EARLY_STOPPING_PATIENCE = 15

# ===== Stage 3: Position =====
STAGE3_LEARNING_RATE = 1e-3
STAGE3_BATCH_SIZE = 8
STAGE3_NUM_EPOCHS = 50
STAGE3_SMOOTHL1_BETA = 0.1
STAGE3_LOSS_WEIGHTS = (0.7, 0.3)  # (position, orientation)

# ===== Augmentation =====
SKELETON_NOISE_SIGMA = 0.01  # meters
DEPTH_NOISE_SIGMA = 0.005  # normalized [0,1]
SPATIAL_JITTER_PIXELS = 4
BRIGHTNESS_JITTER_RANGE = (0.95, 1.05)
ROTATION_RANGE_DEG = 15
TEMPORAL_DROPOUT_PROB = 0.1
HORIZONTAL_FLIP_PROB = 0.5

# ===== Inference =====
INFERENCE_INTENTION_THRESHOLD = 0.7  # Confidence threshold
INFERENCE_LATENCY_BUDGET_MS = 50  # Target <50ms per frame

# ===== Evaluation =====
POSITION_ERROR_THRESHOLD_M = 0.05  # Target <5cm
MAE_X_THRESHOLD_M = 0.03
MAE_Y_THRESHOLD_M = 0.03
MAE_Z_THRESHOLD_M = 0.07
ORIENTATION_COSINE_SIM_THRESHOLD = 0.95

# ===== Device =====
DEVICE = get_device()  # Imports from src.device
TORCH_NUM_THREADS = 4  # CPU parallelism

def _validate():
    """Validate all paths exist and all values are sensible."""
    assert os.path.exists(SHREC_RAW_DIR), f"SHREC dir not found: {SHREC_RAW_DIR}"
    assert (TEMPORAL_WINDOW > 0), "TEMPORAL_WINDOW must be positive"
    assert (D_MODEL % NUM_ATTENTION_HEADS == 0), "d_model must be divisible by num_heads"
    # ... more assertions
```

### Hyperparameter Tuning Guidance

| Parameter | Easy to Tune? | Impact | Recommendation |
|-----------|---|---|---|
| Learning rate | ✓ Fast (5 min/trial) | High (2-3× difference) | Grid search: [1e-5, 1e-4, 1e-3, 1e-2] |
| Batch size | ✓ Fast (just reload data) | Medium (convergence vs VRAM) | Largest batch that fits in GPU |
| d_model | ⚠ Slow (recompile model) | Medium (capacity trade-off) | Start 128, ablate to 256 if underfitting |
| num_transformer_layers | ⚠ Slow | Low-Medium | Usually 2-4 sufficient; 6+ overfits small data |
| dropout_rate | ✓ Fast | Medium (regularization) | Default 0.1, increase to 0.2-0.3 if overfitting |
| warmup_epochs | ✓ Fast | Low | Default 5, rarely needs tuning |

### Environment Variables (Optional)

```bash
# .env file (for sensitive paths or credentials)
SHREC_RAW_DIR=/mnt/data/SHREC2017/
CUDA_VISIBLE_DEVICES=0  # If multi-GPU
WANDB_PROJECT=comp4471-project  # If using W&B logging
```

---

## References & Citations

### Key Papers

1. **[Mercier et al., 3DV 2020]** "A Transformer-Based Network for Dynamic Hand Gesture Recognition"
   - Demonstrates Transformer + depth maps for gesture recognition
   - Achieves SOTA on NVGestures and Briareo datasets
   - URL: https://iris.unimore.it/bitstream/11380/1212263/1/3DV_2020.pdf

2. **[SHREC 2017 Dataset]** "Hand Gesture Recognition Challenge"
   - 2,800 RGB-D gesture sequences from 27 subjects
   - Intel RealSense SR300 camera
   - URL: http://www-rech.telecom-lille.fr/shrec/shrec2017/

3. **[MediaPipe Hands]** "On-device Real-time Hand Tracking with MediaPipe"
   - 21-joint hand skeleton detection from RGB
   - Baseline for skeleton-based approaches
   - URL: https://github.com/google/mediapipe

4. **[Transfer Learning]** "A Survey on Transfer Learning"
   - Justification for 3-stage sequential training
   - Public pretraining → custom fine-tuning best practices

### Datasets

- **Primary**: SHREC 2017 Hand Gesture (2,800 clips, 14 classes)
- **Secondary**: Custom domain-specific (~200 clips, 3 classes)
- **Alternative**: EgoGesture (2,081 RGB-D videos, 83 classes) — if SHREC insufficient

### Reproduction Code

- GitHub: https://github.com/aimagelab/TransformerBasedGestureRecognition
  - Reference implementation of depth-based Transformer
  - Directly adaptable for our 3→6 output head modification

---

## Checkpoints & Artifacts

### Expected Outputs

| Stage | Artifact | Size | Purpose |
|-------|----------|------|---------|
| Stage 0 | `dataset/processed/*.npz` | ~2-5 GB | Train/val/test splits |
| Stage 1 | `models/stage1_pretrained_backbone.pth` | ~15-20 MB | Reusable backbone for Stages 2-3 |
| Stage 2 | `models/stage2_intention_model.pth` | ~20-25 MB | Backbone + Branch A (checkpoint) |
| Stage 3 | `models/final_model.pth` | ~20-25 MB | Full pipeline (deployment ready) |
| Logs | `logs/stage{1,2,3}/*.tensorboard` | ~50-100 MB | Training curves, confusion matrices |

### Reproducibility

- All random seeds fixed in `config.py`
- Model weights saved to allow resuming interrupted training
- Validation metrics logged per epoch
- Best model identified by early stopping

---

## Next Steps

1. **Immediately**: Review this plan with team, collect feedback
2. **This week**: Implement Phase 1 (config, SHREC I/O)
3. **Next week**: Implement Phase 2 (preprocessing)
4. **Week 3**: Implement Phase 3 (model architecture)
5. **Week 4-6**: Stage 1 pretraining on SHREC
6. **Week 7**: Stage 2 & 3 training (awaiting custom data collection)
7. **Week 8**: End-to-end evaluation & deployment

---

**Document Version Control**:
- v1.0 (May 3, 2026): Initial specification from PDF + reference docs
- Updates: Any changes to architecture/training strategy must update this file first

**Questions?** Refer to relevant section above. If unclear, post in project discussion before implementing.

