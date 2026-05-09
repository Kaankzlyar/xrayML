# xrayML — X-ray Body Part Classification

A deep-learning project that classifies grayscale musculoskeletal X-ray images into one of four body parts using **VGG16 transfer learning** in TensorFlow / Keras.

| Class (TR) | Class (EN) |
| ---------- | ---------- |
| `bilek`    | wrist      |
| `dirsek`   | elbow      |
| `el`       | hand       |
| `omuz`     | shoulder   |

---

## Table of contents

1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Dataset](#dataset)
4. [Pipeline](#pipeline)
5. [Model architecture](#model-architecture)
6. [Training workflow](#training-workflow)
7. [Getting started](#getting-started)
8. [Results](#results)
9. [Roadmap](#roadmap)

---

## Overview

The goal is to take a single X-ray image and predict which of the four anatomical regions it depicts. The repository ships two trained artifacts:

- `xray_model.h5` — initial baseline (custom 3-block CNN, grayscale).
- `xray_vgg16.h5` / `best_model.h5` — current model based on VGG16 transfer learning (RGB).

```mermaid
flowchart LR
    A[X-ray image<br/>PNG] --> B[Preprocess<br/>resize 128×128, rescale]
    B --> C[VGG16 backbone<br/>frozen ImageNet weights]
    C --> D[Flatten + Dense 256 + Dropout]
    D --> E[Softmax<br/>4 classes]
    E --> F[Predicted body part<br/>bilek / dirsek / el / omuz]
```

---

## Project structure

```
xrayML/
├── datasets/
│   ├── parts_train/         # training images, one folder per class
│   │   ├── bilek/
│   │   ├── dirsek/
│   │   ├── el/
│   │   └── omuz/
│   └── parts_test/          # held-out test images, same class folders
├── xray_model.py            # training script (VGG16 transfer learning)
├── xray_model.h5            # legacy custom-CNN weights
├── xray_vgg16.h5            # VGG16 final weights
├── best_model.h5            # best checkpoint by val_accuracy
├── training_results.png     # accuracy + loss curves
└── README.md
```

```mermaid
graph TD
    Root[xrayML/] --> DS[datasets/]
    Root --> Script[xray_model.py]
    Root --> M1[xray_model.h5]
    Root --> M2[xray_vgg16.h5]
    Root --> M3[best_model.h5]
    Root --> Plot[training_results.png]
    DS --> Train[parts_train/]
    DS --> Test[parts_test/]
    Train --> T1[bilek/]
    Train --> T2[dirsek/]
    Train --> T3[el/]
    Train --> T4[omuz/]
    Test --> E1[bilek/]
    Test --> E2[dirsek/]
    Test --> E3[el/]
    Test --> E4[omuz/]
```

---

## Dataset

The images are organized into class-named subdirectories so Keras' `flow_from_directory` can infer labels automatically. Filenames follow `studyN_{positive|negative}_imageM.png`.

| Class    | Train images | Test images |
| -------- | -----------: | ----------: |
| bilek    |           27 |           8 |
| dirsek   |           18 |           7 |
| el       |           14 |           7 |
| omuz     |           20 |           5 |
| **Total**|       **79** |      **27** |

> The dataset is intentionally small — this project is set up for transfer learning and aggressive augmentation rather than training from scratch.

```mermaid
pie title Training image distribution
    "bilek (wrist)" : 27
    "dirsek (elbow)" : 18
    "el (hand)" : 14
    "omuz (shoulder)" : 20
```

---

## Pipeline

```mermaid
flowchart TD
    Raw[Raw PNG images<br/>parts_train / parts_test] --> Gen[ImageDataGenerator<br/>rescale 1/255<br/>rotation ±10°<br/>zoom ±0.1<br/>horizontal flip<br/>validation_split=0.2]
    Gen --> Train[train_gen<br/>80% of parts_train]
    Gen --> Valid[valid_gen<br/>20% of parts_train]
    Gen --> Test[test_gen<br/>parts_test]
    Train --> Fit[model.fit]
    Valid --> Fit
    Fit --> CB{Callbacks}
    CB --> Ckpt[ModelCheckpoint<br/>best_model.h5]
    CB --> ES[EarlyStopping<br/>patience=5]
    Fit --> Eval[model.evaluate]
    Test --> Eval
    Eval --> Report[Test accuracy / loss]
    Fit --> Save[xray_vgg16.h5]
    Fit --> Plot[training_results.png]
```

---

## Model architecture

A frozen VGG16 ImageNet backbone is used as a feature extractor, followed by a small dense classification head.

```mermaid
flowchart TB
    In["Input: 128×128×3 RGB"] --> VGG["VGG16 base<br/>weights=imagenet<br/>include_top=False<br/>trainable=False"]
    VGG --> F["Flatten"]
    F --> D1["Dense 256, ReLU"]
    D1 --> Drop["Dropout 0.3"]
    Drop --> Out["Dense 4, Softmax"]
    Out --> P["Class probabilities"]
```

**Compile config**

| Setting       | Value                       |
| ------------- | --------------------------- |
| Optimizer     | Adam, `lr=1e-4`             |
| Loss          | `categorical_crossentropy`  |
| Metric        | `accuracy`                  |
| Epochs (max)  | 30 (early-stopped)          |
| Batch size    | 32                          |
| Input shape   | 128 × 128 × 3               |

---

## Training workflow

```mermaid
sequenceDiagram
    participant User
    participant Script as xray_model.py
    participant Keras as tf.keras
    participant FS as Filesystem

    User->>Script: python xray_model.py
    Script->>FS: read parts_train/, parts_test/
    Script->>Keras: build VGG16 + custom head
    Keras-->>Script: model.summary()
    loop up to 30 epochs
        Script->>Keras: fit on train_gen
        Keras-->>Script: train/val accuracy + loss
        alt val_accuracy improved
            Keras->>FS: save best_model.h5
        end
        alt val_loss stagnates ≥ 5 epochs
            Keras-->>Script: EarlyStopping triggered
        end
    end
    Script->>FS: save xray_vgg16.h5
    Script->>Keras: evaluate on test_gen
    Keras-->>Script: test accuracy / loss
    Script->>FS: save training_results.png
```

---

## Getting started

### 1. Prerequisites

- Python 3.10+
- A working TensorFlow install (CPU is fine for this dataset size, GPU is faster)

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib pillow
```

### 3. Configure dataset paths

`xray_model.py` currently uses hard-coded Windows paths:

```python
train_dir = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_train"
test_dir  = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_test"
```

Update these to point at the local `datasets/parts_train` and `datasets/parts_test` folders, e.g.:

```python
train_dir = "datasets/parts_train"
test_dir  = "datasets/parts_test"
```

### 4. Train

```bash
python xray_model.py
```

Outputs:

- `best_model.h5` — best checkpoint by validation accuracy
- `xray_vgg16.h5` — final weights at end of training
- `training_results.png` — accuracy / loss curves

### 5. Predict on a single image

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASSES = ["bilek", "dirsek", "el", "omuz"]

model = load_model("best_model.h5")

img = image.load_img("path/to/xray.png", target_size=(128, 128), color_mode="rgb")
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

probs = model.predict(x)[0]
print(dict(zip(CLASSES, probs.tolist())))
print("Prediction:", CLASSES[int(np.argmax(probs))])
```

---

## Results

Training and validation curves are saved to `training_results.png` after each run:

![training results](training_results.png)

The exact test accuracy is printed at the end of `xray_model.py` and depends on the random split.

---

## Roadmap

```mermaid
flowchart LR
    A[Baseline custom CNN] --> B[VGG16 transfer learning]
    B --> C[Fine-tune top VGG16 blocks]
    C --> D[Larger / balanced dataset]
    D --> E[Add positive/negative<br/>fracture detection head]
    E --> F[Serve as REST API]
```

Planned next steps:

- Unfreeze the last VGG16 block and fine-tune at a smaller learning rate.
- Rebalance classes (currently `el` and `dirsek` are under-represented).
- Use the `study*_positive_*` / `study*_negative_*` filename convention to add a second head for **abnormality detection**, not just body-part classification.
- Wrap the trained model in a small FastAPI / Flask service for inference.
