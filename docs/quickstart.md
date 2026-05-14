# 快速开始指南

## 环境准备

### 1. 安装依赖

```bash
cd projects/open_vocabulary_detection
pip install -r requirements.txt
```

### 2. 下载预训练权重

```bash
# YOLO26 权重 (会自动从 ultralytics 下载)
# 可选: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt

# YOLO-World 权重 (开放词汇检测预训练)
wget https://github.com/AILab-CVC/YOLO-World/releases/download/v1.0/yolov8l-world.pt -O models/
```

### 3. 验证安装

```bash
python -c "from ultralytics import YOLO; print('✅ Ultralytics OK')"
python -c "import clip; print('✅ CLIP OK')"
```

---

## 快速演示

### 零样本目标检测 (文本提示)

```bash
python scripts/inference.py \
    --image demo.jpg \
    --prompts "person" "car" "dog" "cat" "bicycle" "book"
```

### 批量处理

```bash
python scripts/inference.py \
    --dir test_images/ \
    --prompts "person" "car" "dog" \
    --output results/
```

### 使用中文提示 (需要 Chinese-CLIP)

```python
from inference import OpenVocabularyDetector

detector = OpenVocabularyDetector(clip_model="ViT-L/14")
result = detector.detect_and_classify(
    "demo.jpg",
    text_prompts=["人", "汽车", "狗", "猫"]
)
```

---

## 数据集准备

### 准备自定义数据集

1. 整理图像和标注:
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        └── img3.txt
```

2. YOLO 格式标注:
```
# 每个目标一行: class_id x_center y_center width height (归一化)
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.2
```

3. 创建数据集配置 `my_data.yaml`:
```yaml
path: ./dataset
train: images/train
val: images/val

nc: 10  # 类别数
names: ['person', 'car', 'dog', ...]  # 类别名
```

---

## 训练

### 标准目标检测微调

```bash
python scripts/train.py \
    --model yolo26m.pt \
    --data my_data.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

### 使用配置文件

```bash
python scripts/train.py \
    --config configs/train_base.yaml
```

### 从断点恢复

```bash
python scripts/train.py --resume runs/open_vocab/exp/weights/last.pt
```

---

## 评测

### 标准评测

```bash
python scripts/evaluate.py \
    --model runs/open_vocab/exp/weights/best.pt \
    --data my_data.yaml \
    --device 0
```

### 零样本评测

```bash
python scripts/evaluate.py \
    --model runs/open_vocab/exp/weights/best.pt \
    --data lvis.yaml \
    --task zero_shot
```

---

## 模型导出

训练完成后，导出为 ONNX/TensorRT 格式:

```bash
# 导出 ONNX
python -c "
from ultralytics import YOLO
model = YOLO('runs/open_vocab/exp/weights/best.pt')
model.export(format='onnx')
"

# 导出 TorchScript
model.export(format='torchscript')
```

---

## 项目结构

```
open_vocabulary_detection/
├── README.md              # 项目说明
├── requirements.txt       # 依赖
├── docs/
│   ├── research_plan.md  # 研究计划
│   └── quickstart.md     # 快速开始
├── configs/
│   └── train_base.yaml   # 训练配置
├── scripts/
│   ├── setup.sh          # 环境搭建
│   ├── demo.py           # 简单演示
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评测脚本
│   ├── inference.py      # 推理脚本
│   └── prepare_data.py   # 数据准备
└── models/               # 模型权重
    └── yolov8l-world.pt
```

---

## 常见问题

### Q: CUDA out of memory
```bash
# 减小 batch size
python scripts/train.py --batch 8

# 或使用更小的模型
python scripts/train.py --model yolo26s.pt
```

### Q: 训练中断如何恢复?
```bash
python scripts/train.py --resume runs/open_vocab/exp/weights/last.pt
```

### Q: 如何使用 Chinese-CLIP?
```python
# 修改 inference.py 中的 CLIP 加载部分
from cn_clip import clip as cn_clip
model, preprocess = cn_clip.load("ViT-L-14")
```

### Q: 如何提高检测精度?
1. 使用更大的模型 (yolo26l.pt, yolo26x.pt)
2. 增加训练轮数 (200+ epochs)
3. 使用更大的图像尺寸 (imgsz=1280)
4. 调整数据增强参数
5. 使用 YOLO-World 预训练权重

---

## 下一步

1. ✅ 跑通 demo
2. 🔧 准备自己的数据集
3. 🏋️ 开始训练
4. 📊 评测模型
5. 🚀 部署应用

有问题随时提问！
