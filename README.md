# CFMANet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The official PyTorch implementation of **CFMANet** for **few-shot industrial defect segmentation**. This repository is a complete, runnable training and evaluation framework: the model and its core modules are implemented in `model/cfmanet.py`, which ships together with the ResNet/VGG backbones, data loaders, configuration files, and the train/validation entry points.

---

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- numpy
- opencv-python
- PyYAML
- tqdm
- tensorboard

The backbone constructors rely on torchvision's `model_urls` and ImageNet weights.

---

## Repository Structure

```text
.
├── train.py                  # Training entry point
├── val.py                    # Standalone evaluation entry point
├── config/
│   └── fold0_resnet50.yaml    # Default configuration (ResNet-50, fold 0)
├── model/
│   ├── cfmanet.py              # CFMANet model and core modules
│   ├── mfanet.py              # MFANet comparison model
│   ├── resnet.py              # ResNet backbones
│   └── vgg.py                 # VGG16-BN backbone
├── util/
│   ├── config.py              # YAML configuration loader
│   ├── dataset_fss.py         # FSSD dataset / episodic loader
│   ├── dataset_cdfss.py       # CGFSDS dataset / episodic loader
│   ├── transform.py           # Data augmentation and transforms
│   └── util.py                # Logging, initialization, and utilities
├── data_list/                 # Image/mask list files for every split
│   ├── defect_cdfssd/         # CGFSDS-9 lists (train / val)
│   └── defect_fssd/           # FSSD-12 lists (fold 0 / 1 / 2)
└── LICENSE                    # MIT License
```

---

## Data Preparation

The repository ships with the **list files** under `data_list/`, but not the raw images. Place the datasets under `data/` and point `DATA.data_root` in the YAML config to them:

```text
./data/CGFSDS-9     # default in config/fold0_resnet50.yaml
./data/FSSD-12      # alternative dataset
./data/NFM9         # NFM9 dataset
```

The corresponding train/val list paths (`DATA.train_list`, `DATA.val_list`) are already configured in the YAML file; switch the commented entries to move between CGFSDS-9, FSSD-12, and NFM9, or between folds.

> **NFM9 Dataset Download Link**: https://pan.baidu.com/s/1T1_pejKy-p1wUIOx9nfAKw?pwd=1234 password: 1234

---

## Getting Started

### Training

```bash
python train.py --config config/fold0_resnet50.yaml
```

All hyperparameters (input size, batch size, learning rate, epochs, backbone, shot, snapshot path, etc.) are read from the YAML config. The default setup uses ResNet-50, 200×200 inputs, batch size 4, and 160 epochs.

### Evaluation

```bash
python val.py --config config/fold0_resnet50.yaml --resume <checkpoint_filename>
```

`--resume` is the checkpoint filename inside the snapshot directory defined in the YAML config (it overrides `TRAIN.resume`). If omitted, the value of `TRAIN.resume` is used; an error is raised when neither is set.

### Model Self-Check

```bash
python -m model.cfmanet
```

Runs a forward pass and loss computation with random `2×224×224` inputs, 1-shot and `classes=2`, without requiring any dataset. It prints the parameter count, trainable parameter count, output shape, and the total / main / patch losses, ending with `CFMANet Test Passed!`.

The model is built with `pretrained=True` and loads ImageNet weights, which requires network access to download `resnet50-19c8e357.pth` (or place it in `~/.cache/torch/hub/checkpoints/` beforehand for offline use). Setting `model.pretrained = False` after construction does not change the already-loaded weights.

---

## Core Modules

`model/cfmanet.py` contains:

| Name | Type | Purpose |
| :--- | :--- | :--- |
| `GBM` | Basic unit | 1×1 → 3×3 depthwise-separable → 1×1, with channel gating and a residual connection |
| `SCAMBlock` / `SCAM` | Statistics-Conditioned Antagonistic Modulation | First-/second-order channel statistics, statistical rectification gating, surface-response suppression, and gated residual fusion; projects the concatenated backbone features to 256 channels |
| `HaarWavelet` | Fixed transform | Parameter-free Haar wavelet decomposition producing LL/LH/HL/HH; not trained |
| `MASR` | Morphology-Adaptive Spectral Routing | GBM → Haar decomposition → directional weighting (ADF) → high-/low-frequency routing → relational aggregation → global context; outputs the support prototype |
| `PMA` | Patchwise Matching Alignment | Patch partitioning + Sinkhorn normalization + confidence-gated fusion |
| `SepAtrousConv` | Basic unit | Atrous depthwise-separable convolution; a multi-scale branch of `ECMD` |
| `ECMD` | Entropic Correspondence Modulation Decoder | Query-support fusion + two 2× upsampling stages + multi-scale atrous convolutions + detail/semantic dual branches → segmentation head |
| `CFMANet` | Main model | Assembles the backbone and the modules above; includes parameter freezing, losses, and the optimizer interface |

---

## Results

Input 200×200, single-GPU inference; values are test-set means (%).

| Dataset | 1-shot mIoU | 1-shot FB-IoU | 5-shot mIoU | 5-shot FB-IoU |
| :--- | :---: | :---: | :---: | :---: |
| CGFSDS-9 | 76.15 | 85.84 | 77.13 | 86.97 |
| NFM9 | 73.94 | 83.78 | 74.87 | 84.61 |
| FSSD-12 | 68.34 | 81.56 | 70.14 | 82.14 |

**Efficiency**

- Parameters: 27.68 M
- FLOPs: 82.16 G
- Inference speed: 28.8 FPS (~34.7 ms/image; RTX 4070, batch size 1, 200×200 input)
- TensorRT FP16: ~15 ms per ROI, with end-to-end latency below 200 ms (RTX 4090)

---

## Programmatic Usage

### Building the Model

```python
from model.cfmanet import CFMANet


class Args:
    vgg = False          # True uses VGG16-BN, False uses ResNet
    layers = 50          # 50 / 101, only effective when vgg=False
    classes = 2          # Segmentation head fixed to 2 channels (defect / background)
    shot = 1
    ignore_label = 255
    weight_decay = 5e-5  # Used only by get_optim


model = CFMANet(Args())
```

The fields actually read in `__init__` are `vgg`, `layers`, `classes`, `shot`, and `ignore_label`; `get_optim` additionally reads `weight_decay`.

### Inference

```python
# query:        [B, 3, H, W]
# support:      [B, shot, 3, H, W]
# support_mask: [B, shot, H, W]

model.eval()
with torch.no_grad():
    logits = model(query, support, support_mask)   # [B, 2, H, W]
    pred = logits.argmax(dim=1)
```

In `eval()` mode, `forward` returns `seg_out` directly — the raw logits without argmax or softmax.

### Training

```python
model.train()
pred, total_loss, main_loss, patch_loss = model(
    x=query, s_x=support, s_y=support_mask, y=query_label
)
```

- In `forward(x, s_x, s_y, y)`, the four positional arguments are query, support, support mask, and query label, in that order.
- The total loss is `main + 0.3 * aux + 0.1 * patch`: `main` is the cross-entropy of the main segmentation head, `aux` comes from the auxiliary segmentation head on `layer4`, and `patch` is the patch-level distance between query features and the prototype.
- In training mode, `pred` is the argmax class index `[B, H, W]`, not the logits.

### Optimizer

```python
optimizer = model.get_optim(args, LR=0.001)   # AdamW; weight_decay is taken from args.weight_decay
```

---

## Data Organization

Inputs are organized as episodes:

```text
query:        [B, 3, H, W]
support:      [B, shot, 3, H, W]
support_mask: [B, shot, H, W]   # Foreground is 1, background is 0
query_label:  [B, H, W]         # Invalid pixels are set to ignore_label
```

`support_mask` should be a floating-point or comparable type (the foreground is taken internally with `== 1`); `query_label` ignores invalid pixels according to `ignore_label`.

---

## Implementation Notes

- Backbone stages `layer0`–`layer4` are frozen at construction (`requires_grad=False`), and `train()` is overridden so the backbone and its BatchNorm layers always stay in eval mode during training.
- The convolutions in `layer3` / `layer4` are changed to atrous (dilated) convolutions (dilation 2 / 4) to enlarge the receptive field.
- On the query side, the `layer3` and `layer2` features are concatenated (1024 + 512 = 1536 channels under ResNet-50) and fed into `SCAM`, which reduces them to 256; the auxiliary branch projects `layer4` (2048 channels) separately.
- The segmentation head has a fixed output of 2 channels; keep `args.classes` at 2.
- The `patch_size` of `PMA` defaults to 5 and can be adjusted via `PMA(dim, patch_size=...)`; `ECMD` uses the default internally.
- The routing weights are only for analysis/visualization and do not affect inference: `model.last_support_routing` (directional and high/low-frequency weights) and `model.masr.last_routing`.

---

## License

This project is released under the **MIT License** (SPDX identifier: `MIT`). You may use, modify, and distribute it freely, including for commercial purposes, as long as the copyright notice is retained. See [LICENSE](LICENSE) for the full terms.

Copyright (c) 2026 CFMANet contributors

---

## Contact

For code-related questions, please open an issue in this repository.
