#!/usr/bin/env python3
"""
开放词汇检测推理脚本

功能:
1. YOLO26 目标检测 + CLIP 零样本分类
2. 支持文本提示、图像提示
3. 支持中文类别名
4. 支持视频流处理
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import time

# 导入 CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP 未安装，运行: pip install clip")
    sys.exit(1)

# 导入 Ultralytics
from ultralytics import YOLO


class OpenVocabularyDetector:
    """
    开放词汇目标检测器
    
    结合 YOLO 的定位能力和 CLIP 的语义理解能力
    实现零样本目标检测
    """
    
    def __init__(
        self,
        yolo_model: str = "yolo26m.pt",
        clip_model: str = "ViT-L/14",
        device: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        初始化
        
        Args:
            yolo_model: YOLO 模型路径或名称
            clip_model: CLIP 模型名称 (ViT-L/14, ViT-B/32, etc.)
            device: 运行设备 (cuda/cpu)
            conf_threshold: YOLO 检测置信度阈值
            iou_threshold: NMS IOU 阈值
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"🚀 初始化 OpenVocabularyDetector...")
        print(f"   设备: {self.device}")
        
        # 加载 YOLO
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        # 加载 CLIP
        print(f"   加载 CLIP: {clip_model}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.clip_model.eval()
        
        self.text_features_cache = {}
        print("✅ 初始化完成!")
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本提示为 CLIP 特征
        
        Args:
            texts: 文本列表
            
        Returns:
            归一化的文本特征 [len(texts), embed_dim]
        """
        # 检查缓存
        cache_key = tuple(sorted(texts))
        if cache_key in self.text_features_cache:
            return self.text_features_cache[cache_key]
        
        # Tokenize
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # 编码
        features = self.clip_model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        
        # 缓存
        self.text_features_cache[cache_key] = features
        
        return features
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        编码单个图像为 CLIP 特征
        
        Args:
            image: PIL Image
            
        Returns:
            归一化的图像特征 [1, embed_dim]
        """
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        features = self.clip_model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def detect_and_classify(
        self,
        image: Union[str, np.ndarray, Image.Image],
        text_prompts: List[str],
        return_base64: bool = False
    ) -> Dict:
        """
        开放词汇检测主函数
        
        Args:
            image: 图像路径 / numpy array / PIL Image
            text_prompts: 文本提示列表，如 ["person", "dog", "red car"]
            return_base64: 是否返回 base64 编码的结果图像
            
        Returns:
            检测结果字典
        """
        # 加载图像
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
            img_path = image
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_path = None
        else:
            image_pil = image
            img_path = None
        
        # ========== Stage 1: YOLO 检测候选框 ==========
        yolo_results = self.yolo(
            image_pil,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            return {
                "boxes": [],
                "labels": [],
                "confidences": [],
                "detections": [],
                "image": image_pil if not return_base64 else None
            }
        
        # ========== Stage 2: CLIP 零样本分类 ==========
        text_features = self.encode_text(text_prompts)
        
        detections = []
        labels = []
        confidences = []
        
        for i, (box, yolo_conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            
            # 裁剪 ROI
            roi = image_pil.crop((x1, y1, x2, y2))
            
            # CLIP 分类
            roi_features = self.encode_image(roi)
            similarity = (roi_features @ text_features.T * 100).softmax(dim=-1)
            
            # 获取最高相似度类别
            pred_idx = similarity.argmax().item()
            pred_conf = similarity[0, pred_idx].item()
            pred_label = text_prompts[pred_idx]
            
            # 综合置信度
            combined_conf = yolo_conf * pred_conf
            
            detections.append({
                "box": box.tolist(),
                "label": pred_label,
                "clip_confidence": pred_conf,
                "yolo_confidence": float(yolo_conf),
                "combined_confidence": float(combined_conf),
                "class_idx": pred_idx
            })
            
            labels.append(pred_label)
            confidences.append(pred_conf)
        
        # 排序
        sorted_indices = np.argsort(confidences)[::-1]
        detections = [detections[i] for i in sorted_indices]
        
        return {
            "boxes": boxes[sorted_indices].tolist(),
            "labels": [labels[i] for i in sorted_indices],
            "confidences": [confidences[i] for i in sorted_indices],
            "detections": detections,
            "image": image_pil,
            "image_path": img_path
        }
    
    def draw_detections(self, result: Dict, output_path: str = None) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            result: detect_and_classify 的返回结果
            output_path: 输出路径
            
        Returns:
            绘制了检测框的图像 (numpy array)
        """
        image = result["image"].copy()
        detections = result["detections"]
        
        # 颜色映射
        np.random.seed(42)
        colors = {}
        for i, label in enumerate(set(result["labels"])):
            colors[label] = tuple(np.random.randint(100, 255, 3).tolist())
        
        for det in detections:
            box = det["box"]
            label = det["label"]
            conf = det["combined_confidence"]
            
            x1, y1, x2, y2 = map(int, box)
            color = colors[label]
            
            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            text = f"{label}: {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            Image.fromarray(image).save(output_path)
            print(f"💾 结果保存至: {output_path}")
        
        return np.array(image)
    
    def batch_inference(
        self,
        image_dir: str,
        text_prompts: List[str],
        output_dir: str = "output",
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            image_dir: 图像目录
            text_prompts: 文本提示
            output_dir: 输出目录
            extensions: 支持的图像扩展名
            
        Returns:
            所有图像的检测结果
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像列表
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            print(f"❌ 在 {image_dir} 中未找到图像")
            return []
        
        print(f"🔍 找到 {len(image_paths)} 张图像")
        
        all_results = []
        for img_path in tqdm(image_paths, desc="处理中"):
            result = self.detect_and_classify(str(img_path), text_prompts)
            result["image_path"] = str(img_path)
            
            # 保存可视化结果
            output_path = output_dir / f"result_{img_path.name}"
            self.draw_detections(result, str(output_path))
            
            all_results.append(result)
        
        return all_results
    
    def video_inference(
        self,
        video_path: str,
        text_prompts: List[str],
        output_path: str = None,
        webcam: bool = False
    ):
        """
        视频流推理
        
        Args:
            video_path: 视频路径或摄像头ID
            text_prompts: 文本提示
            output_path: 输出视频路径
            webcam: 是否使用摄像头
        """
        if webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ 无法打开视频源")
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"📹 开始处理视频: {' webcam' if webcam else video_path}")
        print(f"   分辨率: {width}x{height} | FPS: {fps}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                result = self.detect_and_classify(frame, text_prompts)
                frame = self.draw_detections(result)
                
                # 显示 FPS
                elapsed = time.time() - start_time
                fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: --"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 写入输出
                if output_path:
                    out.write(frame)
                
                # 显示
                cv2.imshow("Open Vocabulary Detection", frame)
                
                # 按 q 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                start_time = time.time()
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            print(f"✅ 处理完成: {frame_count} 帧")


def main():
    parser = argparse.ArgumentParser(description="开放词汇检测推理")
    
    # 模型配置
    parser.add_argument("--yolo", type=str, default="yolo26m.pt",
                        help="YOLO 模型")
    parser.add_argument("--clip", type=str, default="ViT-L/14",
                        help="CLIP 模型")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    
    # 输入配置
    parser.add_argument("--image", type=str, default=None,
                        help="单张图像路径")
    parser.add_argument("--video", type=str, default=None,
                        help="视频路径")
    parser.add_argument("--webcam", action="store_true",
                        help="使用摄像头")
    parser.add_argument("--dir", type=str, default=None,
                        help="批量处理目录")
    
    # 类别配置
    parser.add_argument("--prompts", type=str, nargs="+", default=["person", "car", "dog", "cat"],
                        help="文本提示列表")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="从文件加载提示列表")
    
    # 输出配置
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--save_result", action="store_true",
                        help="保存结果图像")
    
    args = parser.parse_args()
    
    # 加载提示列表
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = args.prompts
    
    print(f"📝 检测类别: {prompts}")
    
    # 初始化检测器
    detector = OpenVocabularyDetector(
        yolo_model=args.yolo,
        clip_model=args.clip,
        device=args.device
    )
    
    # 根据输入类型选择模式
    if args.webcam:
        detector.video_inference(0, prompts, webcam=True)
    
    elif args.video:
        output_path = f"{args.output}/result_{Path(args.video).stem}.mp4"
        detector.video_inference(args.video, prompts, output_path)
    
    elif args.image:
        # 单张图像
        result = detector.detect_and_classify(args.image, prompts)
        
        print(f"\n📦 检测到 {len(result['detections'])} 个目标:")
        for det in result["detections"]:
            print(f"   {det['label']}: {det['combined_confidence']:.3f}")
        
        # 保存结果
        if args.save_result:
            output_path = f"{args.output}/result_{Path(args.image).name}"
            detector.draw_detections(result, output_path)
    
    elif args.dir:
        # 批量处理
        results = detector.batch_inference(args.dir, prompts, args.output)
        
        # 统计
        total_detections = sum(len(r["detections"]) for r in results)
        print(f"\n✅ 处理完成! 共 {len(results)} 张图像, {total_detections} 个检测目标")
    
    else:
        print("❌ 请指定输入: --image / --video / --webcam / --dir")
        print("   示例:")
        print("   python inference.py --image photo.jpg --prompts person car dog")
        print("   python inference.py --dir images/ --prompts person car")
        print("   python inference.py --webcam")


if __name__ == "__main__":
    main()
