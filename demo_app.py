#!/usr/bin/env python3
"""
YOLO + CLIP 开放词汇目标检测 Demo
OpenCV + CustomTkinter GUI

Usage:
    python demo_app.py
"""

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import clip
import os
import threading
import uuid
from collections import Counter

# ── 路径配置 ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHT = os.path.join(PROJECT_ROOT, "runs/detect/runs/voc_full/exp1/weights/best.pt")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "examples/open_vocab_detection_result.jpg")

# If fine-tuned weight doesn't exist, use pretrained yolov8m
if not os.path.exists(YOLO_WEIGHT):
    YOLO_WEIGHT = "yolov8m.pt"  # auto-download from Ultralytics HUB

# ── VOC 20 类 ──────────────────────────────────────
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ── 颜色调色板 ─────────────────────────────────────
PALETTE = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (255, 150, 50), (150, 50, 255),
    (100, 200, 255), (200, 100, 255), (255, 180, 50), (180, 255, 50),
    (50, 180, 255), (200, 255, 150), (255, 150, 200), (150, 255, 200),
    (100, 100, 255), (255, 100, 150), (100, 255, 150), (200, 150, 255),
]


class Detector:
    """检测引擎"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Loading YOLO on {self.device}...")
        self.yolo = YOLO(YOLO_WEIGHT)
        print("[Detector] Loading CLIP ViT-L/14@336px...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model.eval()
        self._text_features = None
        self.classes = None

    def encode_texts(self, class_names):
        """编码文本类别"""
        prompts = [f"a photo of a {c}" for c in class_names]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        self._text_features = features
        self.classes = class_names

    def detect(self, image_bgr, conf_threshold=0.2, nms_iou=0.45):
        """检测并用 CLIP 分类"""
        if self._text_features is None:
            raise ValueError("Call encode_texts() first!")

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # YOLO 检测候选框
        results = self.yolo(image_bgr, conf=conf_threshold * 0.5, iou=nms_iou, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return [], img_rgb

        boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_confs = results[0].boxes.conf.cpu().numpy()

        pil_img = Image.fromarray(img_rgb)
        detections = []

        # 逐框 CLIP 分类
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w-1, x2)), int(min(h-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.clip_model.encode_image(crop_tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
                sim = (feat @ self._text_features.T).squeeze(0) * 100.0
                probs = F.softmax(sim, dim=-1)

            clip_conf, cls_idx = probs.topk(1)
            clip_conf = clip_conf.item()
            cls_idx = cls_idx.item()
            cls_name = self.classes[cls_idx]

            combined = float(yolo_confs[i]) * clip_conf
            if combined < conf_threshold:
                continue

            detections.append({
                "box": [x1, y1, x2, y2],
                "class": cls_name,
                "conf": combined,
                "clip_conf": clip_conf,
                "yolo_conf": float(yolo_confs[i]),
                "color": PALETTE[cls_idx % len(PALETTE)]
            })

        return detections


class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("YOLO + CLIP 开放词汇检测 Demo")
        self.geometry("1100x700")

        self.detector = Detector()
        self.current_image = None
        self.current_image_path = None
        self.detections = []
        self.class_colors = {}

        # ── 布局 ──────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_canvas()

        # 预设类别
        self.entry_classes.insert(0, "person, dog, cat, car, bicycle, chair")

    def _build_sidebar(self):
        """左侧控制面板"""
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure([3, 5, 7, 9, 11, 12], weight=1)

        # 标题
        title = ctk.CTkLabel(sidebar, text="🔍 检测控制", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 5))

        sep = ctk.CTkLabel(sidebar, text="─" * 25, text_color="gray40")
        sep.grid(row=1, column=0, padx=10, pady=2)

        # 类别输入
        lbl = ctk.CTkLabel(sidebar, text="检测类别 (英文,逗号分隔)", font=ctk.CTkFont(weight="bold"))
        lbl.grid(row=2, column=0, padx=20, pady=(10, 2), sticky="w")

        self.entry_classes = ctk.CTkEntry(sidebar, placeholder_text="person, dog, car...")
        self.entry_classes.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        # 快捷按钮
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=4, column=0, padx=20, pady=2, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_frame, text="VOC 20类", command=self._set_voc20,
                      fg_color="transparent", border_width=1).grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(btn_frame, text="常见物体", command=self._set_common,
                      fg_color="transparent", border_width=1).grid(row=0, column=1, padx=2, pady=2)

        # 置信度阈值
        lbl2 = ctk.CTkLabel(sidebar, text="置信度阈值", font=ctk.CTkFont(weight="bold"))
        lbl2.grid(row=5, column=0, padx=20, pady=(10, 2), sticky="w")

        self.slider_conf = ctk.CTkSlider(sidebar, from_=0.05, to=0.9, number_of_steps=17,
                                          command=self._on_slider_change)
        self.slider_conf.set(0.25)
        self.slider_conf.grid(row=6, column=0, padx=20, pady=(2, 0), sticky="ew")
        self.lbl_conf_val = ctk.CTkLabel(sidebar, text="0.25", text_color="gray60")
        self.lbl_conf_val.grid(row=7, column=0, padx=20, pady=(0, 5), sticky="w")

        # 运行按钮
        self.btn_run = ctk.CTkButton(sidebar, text="🚀 开始检测", font=ctk.CTkFont(size=16, weight="bold"),
                                     command=self._run_detection, height=45,
                                     fg_color="#2fa572", hover_color="#258c5e")
        self.btn_run.grid(row=8, column=0, padx=20, pady=10, sticky="ew")

        # 状态
        self.lbl_status = ctk.CTkLabel(sidebar, text="就绪", text_color="gray60", anchor="w")
        self.lbl_status.grid(row=9, column=0, padx=20, pady=5, sticky="ew")

        # 检测结果列表
        lbl3 = ctk.CTkLabel(sidebar, text="检测结果", font=ctk.CTkFont(weight="bold"))
        lbl3.grid(row=10, column=0, padx=20, pady=(10, 2), sticky="w")

        self.result_list = ctk.CTkTextbox(sidebar, font=("Consolas", 12), state="disabled", height=200)
        self.result_list.grid(row=11, column=0, padx=20, pady=5, sticky="ew")

        # 保存按钮
        self.btn_save = ctk.CTkButton(sidebar, text="💾 保存结果", command=self._save_result,
                                      fg_color="transparent", border_width=1)
        self.btn_save.grid(row=12, column=0, padx=20, pady=(5, 20), sticky="ew")

    def _on_slider_change(self, value):
        self.lbl_conf_val.configure(text=f"{value:.2f}")

    def _build_canvas(self):
        """右侧图像画布"""
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # 内层用于拖放
        self.drop_frame = ctk.CTkFrame(self.canvas_frame, fg_color="#1a1a2e", corner_radius=10)
        self.drop_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.drop_frame.grid_rowconfigure(0, weight=1)
        self.drop_frame.grid_columnconfigure(0, weight=1)

        self.lbl_image = ctk.CTkLabel(self.drop_frame, text="拖放图片到此处\n\n或双击选择图片",
                                       font=ctk.CTkFont(size=16), text_color="gray60")
        self.lbl_image.grid(row=0, column=0)

        # 绑定拖放
        self.drop_frame.drop_target_register("DND_Files")
        self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)
        self.lbl_image.bind("<Double-Button-1>", lambda e: self._open_file())

    def _on_drop(self, target):
        files = self.drop_frame.tk.splitlist(target)
        if files:
            path = files[0]
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self._load_image(path)

    def _set_voc20(self):
        self.entry_classes.delete(0, "end")
        self.entry_classes.insert(0, ", ".join(VOC_CLASSES))

    def _set_common(self):
        self.entry_classes.delete(0, "end")
        self.entry_classes.insert(0, "person, dog, cat, car, bicycle, bus, chair, bottle, book, laptop")

    def _open_file(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            self.lbl_status.configure(text=f"无法读取图片: {path}")
            return
        self.current_image = img
        self.current_image_path = path
        self.detections = []
        self._display_image(img)
        self.lbl_status.configure(text=f"已加载: {os.path.basename(path)}")
        self._update_result_list([])

    def _display_image(self, img_bgr):
        """在 canvas 上显示图片"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]

        canvas_w = self.drop_frame.winfo_width()
        canvas_h = self.drop_frame.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 700, 600

        scale = min(canvas_w / iw, canvas_h / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)

        rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(rgb_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.lbl_image.configure(image=img_tk, text="")
        self.lbl_image.image = img_tk

    def _run_detection(self):
        if self.current_image is None:
            self.lbl_status.configure(text="请先加载图片")
            return

        class_text = self.entry_classes.get().strip()
        if not class_text:
            self.lbl_status.configure(text="请输入检测类别")
            return

        classes = [c.strip() for c in class_text.replace("，", ",").split(",") if c.strip()]
        if not classes:
            return

        conf_thresh = float(self.slider_conf.get())
        self.lbl_status.configure(text="检测中...")
        self.btn_run.configure(state="disabled", text="检测中...")

        def detect_thread():
            try:
                self.detector.encode_texts(classes)
                dets = self.detector.detect(self.current_image.copy(), conf_threshold=conf_thresh)
                self.detections = dets
            except Exception as e:
                self.detections = []
                print(f"Detection error: {e}")
            finally:
                self.after(0, self._on_detection_done)

        threading.Thread(target=detect_thread, daemon=True).start()

    def _on_detection_done(self):
        self.btn_run.configure(state="normal", text="🚀 开始检测")

        if not self.detections:
            self.lbl_status.configure(text="未检测到目标，尝试降低阈值")
            return

        # 画框
        img_annotated = self.current_image.copy()
        for det in self.detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            cls = det["class"]
            conf = det["conf"]

            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_annotated, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
            cv2.putText(img_annotated, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        self._display_image(img_annotated)

        # 统计
        counter = Counter(d["class"] for d in self.detections)
        self.lbl_status.configure(text=f"检测完成，共 {len(self.detections)} 个目标")
        self._update_result_list(counter)

    def _update_result_list(self, counter):
        self.result_list.configure(state="normal")
        self.result_list.delete("1.0", "end")
        if not counter:
            self.result_list.insert("end", "暂无结果\n")
        else:
            for cls, cnt in sorted(counter.items(), key=lambda x: -x[1]):
                self.result_list.insert("end", f"  {cls:15s} × {cnt}\n")
        self.result_list.configure(state="disabled")

    def _save_result(self):
        if self.current_image is None:
            self.lbl_status.configure(text="没有可保存的图片")
            return

        import tkinter.filedialog as fd
        path = fd.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"detect_{uuid.uuid4().hex[:6]}.jpg"
        )
        if not path:
            return

        if self.detections:
            img = self.current_image.copy()
            for det in self.detections:
                x1, y1, x2, y2 = det["box"]
                color = det["color"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {det['conf']:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
                cv2.putText(img, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            img = self.current_image

        cv2.imwrite(path, img)
        self.lbl_status.configure(text=f"已保存: {os.path.basename(path)}")


if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
