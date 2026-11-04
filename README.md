# CFMANet

少样本工业缺陷分割（few-shot defect segmentation）代码库。本仓库提供 CFMANet 的模型定义与核心模块实现，单文件 `Final_CFMANet_TII.py` 即可给出完整网络。

---

## 环境

- Python 3.8+
- PyTorch >= 1.10
- numpy
- torchvision（主干构建函数依赖其 `model_urls` 与 ImageNet 权重）

## 运行前提：需要 `model/` 包

`Final_CFMANet_TII.py` 顶部有两条导入来自外部主干代码：

```python
import model.resnet as models
import model.vgg as vgg_models
```

所以请把本文件放在提供 `model/resnet.py` 与 `model/vgg.py` 的训练框架仓库根目录下（或在同目录放置 `model/` 包），否则导入即失败。本仓库只包含模型与模块实现，不含主干代码、数据加载与训练入口。

---

## 文件说明

```text
.
├── Final_CFMANet_TII.py   # CFMANet 模型与核心模块
└── model/                 # 需自备：resnet.py / vgg.py（ResNet / VGG16-BN 构建函数）
```

`Final_CFMANet_TII.py` 包含：

| 名称 | 类型 | 作用 |
| :--- | :--- | :--- |
| `GBM` | 基础单元 | 1×1 → 3×3 深度可分离 → 1×1，带通道门控与残差 |
| `GBM_ResBlock` | 基础单元 | 两层 `GBM` 构成的残差块 |
| `AMGEBlock` / `AMGE` | 嵌入 | 通道一阶/二阶统计、统计整流门、表面响应抑制、门控残差融合 |
| `WavePool` | 变换 | 固定 Haar 小波分解，输出 LL/LH/HL/HH；无参数、不参与训练 |
| `MCA` | 注意力 | 分组方向注意力：长条池化 + 3×3 深度卷积 + 局部响应加权 |
| `GLCM` | 注意力 | 通道注意力，兼容 4D 特征图与 3D 序列特征 |
| `FDPG` | 原型生成 | GBM → Haar 分解 → 方向加权 → 高/低频路由 → 自注意力聚合 → 全局上下文，输出支撑集原型 |
| `PMA` | 匹配 | patch 划分 + Sinkhorn 归一化 + 置信度门控融合 |
| `SepAtrousConv` | 基础单元 | 空洞深度可分离卷积，`CMAD` 的多尺度分支 |
| `CMAD` | 解码器 | query-support 融合 + 两级 2× 上采样 + 多尺度空洞 + 细节/语义双分支 → 分割头 |
| `CFMANet` | 主模型 | 主干与上述模块组装，含参数冻结、损失与优化器接口 |
| `mfanet` | 别名 | `CFMANet` 的兼容别名 |

---

## 快速自检

```bash
python Final_CFMANet_TII.py
```

用 2×224×224 随机输入、1-shot、`classes=2` 跑一次前向与损失计算，不依赖数据集。输出参数量、可训练参数量、输出形状以及 total / main / patch 三项损失，最后打印 `CFMANet Test Passed!`。

模型构造时按 `self.pretrained=True` 加载 ImageNet 权重，需要联网让 `model_zoo` 下载 `resnet50-19c8e357.pth`；离线环境请先把该文件放进 torch hub 缓存目录（默认 `~/.cache/torch/hub/checkpoints/`）。注意 `__main__` 中构造之后写的 `model.pretrained = False` 不会改变已加载的权重。

---

## 结果

输入 200×200，单卡推理，数值为测试集均值（%）：

| Dataset | 1-shot mIoU | 1-shot FB-IoU | 5-shot mIoU | 5-shot FB-IoU |
| :--- | :---: | :---: | :---: | :---: |
| CGFSDS-9 | 76.15 | 85.84 | 77.13 | 86.97 |
| NFM9 | 73.94 | 83.78 | 74.87 | 84.61 |
| FSSD-12 | 68.34 | 81.56 | 70.14 | 82.14 |

**效率**

- 参数量：27.68 M
- FLOPs：82.16 G
- 推理速度：28.8 FPS（约 34.7 ms/图；RTX 4070，batch size 1，输入 200×200）
- TensorRT FP16：单 ROI 约 15 ms，端到端延迟低于 200 ms（RTX 4090）

---

## 使用示例

### 构建模型

```python
import torch
from Final_CFMANet_TII import CFMANet


class Args:
    vgg = False          # True 走 VGG16-BN，False 走 ResNet
    layers = 50          # 50 / 101，仅在 vgg=False 时生效
    classes = 2          # 分割头固定输出 2 通道（缺陷 / 背景）
    shot = 1
    ignore_label = 255
    weight_decay = 5e-5  # 仅 get_optim 使用


model = CFMANet(Args())
```

`__init__` 实际读取的字段是 `vgg`、`layers`、`classes`、`shot`、`ignore_label`，`get_optim` 另外读取 `weight_decay`。

### 推理

```python
# query:        [B, 3, H, W]
# support:      [B, shot, 3, H, W]
# support_mask: [B, shot, H, W]

model.eval()
with torch.no_grad():
    logits = model(query, support, support_mask)   # [B, 2, H, W]
    pred = logits.argmax(dim=1)
```

`eval()` 模式下 `forward` 直接返回 `seg_out`，即未做 argmax / softmax 的原始 logits。

### 训练

```python
model.train()
pred, total_loss, main_loss, patch_loss = model(
    x=query, s_x=support, s_y=support_mask, y=query_label
)
```

- `forward(x, s_x, s_y, y)` 中四个位置参数依次是 query、support、support_mask、query_label。
- 总损失为 `main + 0.3 * aux + 0.1 * patch`：`main` 是主分割头交叉熵，`aux` 来自 `layer4` 上的辅助分割头，`patch` 是 query 特征与原型之间的 patch 级距离。
- 训练模式下 `pred` 是 argmax 后的类别索引 `[B, H, W]`，不是 logits。请将上述接口接入你的训练框架与数据加载器。

### 优化器

```python
optimizer = model.get_optim(args, LR=0.001)   # AdamW，weight_decay 取 args.weight_decay
```

---

## 数据组织

按 episode 形式组织输入：

```text
query:        [B, 3, H, W]
support:      [B, shot, 3, H, W]
support_mask: [B, shot, H, W]   # 前景为 1，背景为 0
query_label:  [B, H, W]         # 无效像素填 ignore_label
```

`support_mask` 需为浮点或可比较类型（内部以 `== 1` 取前景）；`query_label` 按 `ignore_label` 忽略无效像素。

---

## 实现说明

- 主干 `layer0`–`layer4` 在构造时冻结（`requires_grad=False`），且 `train()` 被重写，训练阶段主干与 BN 也始终处于 eval 模式。
- `layer3` / `layer4` 的卷积被改为空洞卷积（dilation 2 / 4），用于扩大感受野。
- query 侧取 `layer3` 与 `layer2` 特征拼接（ResNet-50 下 1024+512=1536 通道）送入 `AMGE` 降到 256；辅助分支单独投影 `layer4`（2048 通道）。
- 分割头输出通道固定为 2，`args.classes` 请保持 2。
- `PMA` 的 `patch_size` 默认 5，可通过 `PMA(dim, patch_size=...)` 调整；`CMAD` 内部使用默认值。
- 路由权重只用于分析/可视化，不影响推理结果：`model.last_support_routing`（方向与高低频权重）、`model.fdpg.last_routing`。

---

## License

本仓库代码开源，具体条款请参见仓库中的 `LICENSE` 文件。如暂未添加 `LICENSE`，请联系作者确认使用许可。

---

## Contact

代码相关问题请在本仓库提交 issue。
