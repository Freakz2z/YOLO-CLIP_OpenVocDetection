#!/usr/bin/env python3
"""
Optimized YOLO + CLIP Open Vocabulary Detection
Batch CLIP: 所有候选框一次过 CLIP forward，避免逐框循环
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import clip
from PIL import Image, ImageDraw, ImageFont
import time
from collections import Counter

# ── 配置 ──────────────────────────────────────
YOLO_WEIGHT = "runs/detect/runs/voc_full2/exp1/weights/best.pt"
CLIP_MODEL = "ViT-L/14@336px"
CONF_THRESH = 0.15
NMS_IOU = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PALETTE = [
    (255, 80, 80), (80, 220, 80), (80, 80, 255), (255, 220, 80),
    (255, 80, 220), (80, 220, 220), (255, 150, 60), (180, 80, 255),
    (80, 200, 255), (220, 120, 255), (255, 190, 60), (190, 255, 80),
]


def load_models():
    print(f"[1/3] YOLO: {YOLO_WEIGHT}")
    yolo = YOLO(YOLO_WEIGHT)
    print(f"[2/3] CLIP: {CLIP_MODEL}")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    clip_model.eval()
    print(f"[3/3] Ready")
    return yolo, clip_model, clip_preprocess


def encode_classes(clip_model, class_names):
    """编码用户指定的文本类别"""
    texts = [f"a photo of a {c}" for c in class_names]
    tokens = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def detect_batch(yolo, clip_model, clip_preprocess, text_features, class_names,
                 image_bgr, conf_threshold=0.2):
    """
    核心优化: Batch CLIP forward
    - YOLO 一次出所有候选框
    - CLIP 一次 batch forward 分所有候选框
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)

    # ── 1. YOLO 候选框 ─────────────────────
    results = yolo(image_bgr, conf=conf_threshold * 0.3, iou=NMS_IOU, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return [], img_rgb

    boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_confs = results[0].boxes.conf.cpu().numpy()
    n = len(boxes)

    # ── 2. Batch 提取所有 crop ──────────────
    crop_batch = []
    valid_idx = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = pil_img.crop((x1, y1, x2, y2))
        crop_tensor = clip_preprocess(crop)
        crop_batch.append(crop_tensor)
        valid_idx.append(i)

    if not crop_batch:
        return [], img_rgb

    # Stack: [N, 3, 224, 224]
    crop_batch = torch.stack(crop_batch).to(DEVICE)

    # ── 3. 单次 CLIP batch forward ─────────
    with torch.no_grad():
        # 一次过完所有 crop
        batch_features = clip_model.encode_image(crop_batch)
        batch_features /= batch_features.norm(dim=-1, keepdim=True)

        # 一次矩阵乘法 [N, num_classes]
        # CLIP 需要 logit_scale (~100) 来让 softmax 分布更 sharp
        LOGIT_SCALE = 100.0
        sim = batch_features @ text_features.T * LOGIT_SCALE
        probs = F.softmax(sim, dim=-1)

        # CLIP 最高类
        clip_confs, top_indices = probs.max(dim=1)  # [N]

    # ── 4. 组装结果 ────────────────────────
    yolo_conf_tensor = torch.tensor(
        [yolo_confs[i] for i in valid_idx], device=DEVICE
    )
    combined = (yolo_conf_tensor * clip_confs).cpu().numpy()

    detections = []
    for j, (orig_idx, (x1, y1, x2, y2)) in enumerate(zip(valid_idx, boxes[valid_idx])):
        conf = combined[j]
        if conf < conf_threshold:
            continue

        cls_idx = top_indices[j].item()
        detections.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "class": class_names[cls_idx],
            "conf": float(conf),
            "clip_conf": float(clip_confs[j].item()),
            "yolo_conf": float(yolo_confs[orig_idx]),
            "color": PALETTE[cls_idx % len(PALETTE)]
        })

    return detections, img_rgb


def draw_and_show(detections, img_rgb, winname="Batch CLIP Detection"):
    draw = ImageDraw.Draw(Image.fromarray(img_rgb))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = font_s = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['class']} {det['conf']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1-22, x1+len(label)*9+6, y1], fill=color)
        draw.text((x1+3, y1-20), label, fill=(255,255,255), font=font_s)

    result = np.array(Image.fromarray(img_rgb))
    return result


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_inference_fast.py <image_path> [classes] [save_path]")
        print("  classes: comma-separated, default='person,dog,cat,car,bicycle,chair,bus,bottle'")
        sys.exit(1)

    image_path = sys.argv[1]

    if len(sys.argv) > 2:
        class_names = [c.strip() for c in sys.argv[2].split(",") if c.strip()]
    else:
        class_names = ["person", "dog", "cat", "car", "bicycle", "chair", "bus", "bottle"]

    save_path = sys.argv[3] if len(sys.argv) > 3 else None

    # 加载模型
    yolo, clip_model, clip_preprocess = load_models()

    # 编码文本
    print(f"[Text] Encoding {len(class_names)} classes: {class_names}")
    text_features = encode_classes(clip_model, class_names)

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        sys.exit(1)

    # 检测
    print(f"[Detect] Processing...")
    t0 = time.time()
    detections, img_rgb = detect_batch(
        yolo, clip_model, clip_preprocess, text_features,
        class_names, img, conf_threshold=CONF_THRESH
    )
    elapsed = time.time() - t0

    # 可视化
    result = draw_and_show(detections, img_rgb)

    # 打印结果
    print(f"\n✓ {len(detections)} detections in {elapsed:.2f}s")
    from collections import Counter
    counter = Counter(d["class"] for d in detections)
    for cls, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {cls:15s} x {cnt}")

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved: {save_path}")

    # 显示
    cv2.imshow("Batch CLIP Detection", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
