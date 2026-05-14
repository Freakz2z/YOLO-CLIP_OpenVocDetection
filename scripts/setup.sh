#!/bin/bash
# YOLO-World + CLIP 开放词汇检测环境搭建

# 创建虚拟环境
conda create -n open_vocab python=3.10 -y
conda activate open_vocab

# 安装 PyTorch (CUDA 11.8 or 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics YOLO26
pip install ultralytics

# 安装 CLIP (open_clip)
pip install open_clip_toolkit

# 安装 Chinese-CLIP (可选，用于中文)
# pip install cn_clip

# 安装其他依赖
pip install opencv-python pillow matplotlib pandas

# 克隆 YOLO-World
git clone https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World
pip install -e .
cd ..

# 下载预训练权重 (YOLO-World-L)
mkdir -p models
wget -O models/yolov8l-world.pt https://github.com/AILab-CVC/YOLO-World/releases/download/v1.0/yolov8l-world.pt

echo "✅ 环境搭建完成！"
