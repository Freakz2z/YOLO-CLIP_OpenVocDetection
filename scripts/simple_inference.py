#!/usr/bin/env python3
"""
YOLO + CLIP 开放词汇检测 - 简化版

两阶段方案:
1. YOLO 负责目标定位 (提取候选框)
2. CLIP 负责零样本分类 (文本提示)

使用方法:
    python simple_inference.py --image test.jpg --prompts person car dog
"""

import argparse
import torch
from ultralytics import YOLO
import clip
from PIL import Image
import cv2
import numpy as np


def load_models(yolo_model='yolo26m.pt', clip_model='ViT-L/14'):
    """加载 YOLO 和 CLIP 模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📦 加载模型 on {device}...")
    
    yolo = YOLO(yolo_model).to(device)
    clip_model, preprocess = clip.load(clip_model, device=device)
    
    print("✅ 模型加载完成")
    return yolo, clip_model, preprocess, device


def detect(image_path, prompts, yolo, clip_model, preprocess, device, yolo_conf=0.25):
    """
    开放词汇目标检测
    
    Args:
        image_path: 图像路径
        prompts: 检测类别列表，如 ['person', 'car', 'dog']
        yolo: YOLO 模型
        clip_model: CLIP 模型
        preprocess: CLIP 图像预处理
        device: 运行设备
        yolo_conf: YOLO 置信度阈值
    
    Returns:
        detections: 检测结果列表
    """
    # 编码文本提示
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # YOLO 检测
    results = yolo(image_path, conf=yolo_conf, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    print(f"📦 检测到 {len(boxes)} 个目标")
    
    # CLIP 分类
    img = Image.open(image_path).convert('RGB')
    detections = []
    
    for box, yolo_c in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        roi = img.crop((x1, y1, x2, y2))
        roi_input = preprocess(roi).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feat = clip_model.encode_image(roi_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            similarity = (img_feat @ text_features.T * 100).softmax(dim=-1)
            
            pred_idx = similarity.argmax().item()
            pred_conf = similarity[0, pred_idx].item()
        
        detections.append({
            'label': prompts[pred_idx],
            'confidence': pred_conf * yolo_c,
            'box': box.tolist()
        })
    
    return detections


def draw(image_path, detections, output_path=None):
    """绘制检测结果"""
    img = cv2.imread(image_path)
    
    np.random.seed(42)
    colors = {t: tuple(np.random.randint(100, 255, 3).tolist()) 
              for t in set(d['label'] for d in detections)}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        color = colors[det['label']]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{det['label']}: {det['confidence']:.2f}"
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"💾 结果已保存: {output_path}")
    
    return img


def main():
    parser = argparse.ArgumentParser(description="YOLO + CLIP 开放词汇检测")
    parser.add_argument("--image", "-i", required=True, help="图像路径")
    parser.add_argument("--prompts", "-p", nargs="+", 
                        default=['person', 'car', 'dog', 'cat', 'bicycle'],
                        help="检测类别提示")
    parser.add_argument("--output", "-o", default=None, help="输出图像路径")
    parser.add_argument("--yolo-model", default='yolo26m.pt', help="YOLO 模型")
    parser.add_argument("--clip-model", default='ViT-L/14', help="CLIP 模型")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 YOLO + CLIP 开放词汇检测")
    print("=" * 60)
    print(f"📷 图像: {args.image}")
    print(f"🏷️  类别: {args.prompts}")
    
    # 加载模型
    yolo, clip_model, preprocess, device = load_models(args.yolo_model, args.clip_model)
    
    # 检测
    detections = detect(args.image, args.prompts, yolo, clip_model, preprocess, device)
    
    # 打印结果
    print(f"\n📦 检测结果:")
    for i, det in enumerate(detections, 1):
        print(f"   [{i}] {det['label']}: {det['confidence']:.3f}")
    
    # 绘制
    if args.output or True:
        output = args.output or args.image.replace('.', '_result.')
        draw(args.image, detections, output)
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
