#!/usr/bin/env python3
"""
YOLO + Chinese-CLIP 开放词汇检测 Demo
支持中文文本输入，真正做中文开放词汇检测

Usage:
    python demo_app_cn.py
"""

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from cn_clip.clip import load_from_name
import os
import threading
import uuid
from collections import Counter

# ── 路径配置 ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHT = os.path.join(PROJECT_ROOT, "runs/detect/runs/voc_full2/exp1/weights/best.pt")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "examples/open_vocab_detection_result.jpg")

if not os.path.exists(YOLO_WEIGHT):
    YOLO_WEIGHT = "yolov8m.pt"

PALETTE = [
    (255, 80, 80), (80, 220, 80), (80, 80, 255), (255, 220, 80),
    (255, 80, 220), (80, 220, 220), (255, 150, 60), (180, 80, 255),
    (80, 200, 255), (220, 120, 255), (255, 190, 60), (190, 255, 80),
]


def encode_text_cn(model, texts):
    """CN-CLIP 文本编码"""
    tokens = [model.tokenizer.tokenize(t) for t in texts]
    ids = [model.tokenizer.convert_tokens_to_ids(t) for t in tokens]
    max_len = max(len(i) for i in ids)
    ids = [i + [0] * (max_len - len(i)) for i in ids]
    return torch.tensor(ids, device="cuda")


class DetectorCN:
    """CN-CLIP 检测引擎"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Detector] Loading YOLO...")
        self.yolo = YOLO(YOLO_WEIGHT)
        print("[Detector] Loading Chinese-CLIP ViT-L-14...")
        self.cn_model, self.cn_preprocess = load_from_name("ViT-L-14", device=self.device)
        print("[Detector] ✓ Chinese-CLIP loaded!")
        self._text_features = None
        self.classes = None

    def encode_texts(self, class_names):
        """编码中文文本类别"""
        tokens = encode_text_cn(self.cn_model, class_names)
        with torch.no_grad():
            features = self.cn_model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        self._text_features = features
        self.classes = class_names

    def detect(self, image_bgr, conf_threshold=0.2, nms_iou=0.45):
        """检测 + CN-CLIP 分类"""
        if self._text_features is None:
            raise ValueError("Call encode_texts() first!")

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        pil_img = Image.fromarray(img_rgb)

        results = self.yolo(image_bgr, conf=conf_threshold * 0.3, iou=nms_iou, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return [], img_rgb

        boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_confs = results[0].boxes.conf.cpu().numpy()

        # Batch crop
        crops, valid_idx = [], []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w-1, x2)), int(min(h-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(self.cn_preprocess(pil_img.crop((x1, y1, x2, y2))))
            valid_idx.append(i)

        if not crops:
            return [], img_rgb

        crop_batch = torch.stack(crops).to(self.device)

        with torch.no_grad():
            batch_features = self.cn_model.encode_image(crop_batch)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            sim = batch_features @ self._text_features.T
            probs = F.softmax(sim * self.cn_model.logit_scale.exp(), dim=-1)
            clip_confs, top_indices = probs.max(dim=1)

        yolo_conf_t = torch.tensor([yolo_confs[i] for i in valid_idx], device=self.device)
        combined = (yolo_conf_t * clip_confs).cpu().numpy()

        detections = []
        for j, orig_idx in enumerate(valid_idx):
            conf = float(combined[j])
            if conf < conf_threshold:
                continue
            cls_idx = top_indices[j].item()
            x1, y1, x2, y2 = boxes[orig_idx].astype(int)
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "class": self.classes[cls_idx],
                "conf": conf,
                "color": PALETTE[cls_idx % len(PALETTE)]
            })

        return detections, img_rgb


class DemoAppCN(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")  # 换个主题色

        self.title("YOLO + Chinese-CLIP 开放词汇检测")
        self.geometry("1100x700")

        self.detector = DetectorCN()
        self.current_image = None
        self.detections = []

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_canvas()

        self.entry_classes.insert(0, "狗, 人, 猫, 车, 瓶子")

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure([3, 5, 7, 9, 11], weight=1)

        title = ctk.CTkLabel(sidebar, text="🔍 中文检测控制",
                              font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 5))

        lbl = ctk.CTkLabel(sidebar, text="检测类别 (中文,逗号分隔)",
                            font=ctk.CTkFont(weight="bold"))
        lbl.grid(row=2, column=0, padx=20, pady=(10, 2), sticky="w")

        self.entry_classes = ctk.CTkEntry(sidebar, placeholder_text="狗, 人, 猫, 车...")
        self.entry_classes.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        # 快捷
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=4, column=0, padx=20, pady=2, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btn_frame, text="常见中文", command=lambda: self.entry_classes.delete(0, "end") or self.entry_classes.insert(0, "狗, 人, 猫, 车, 瓶子, 椅子"),
                      fg_color="transparent", border_width=1).grid(row=0, column=0, padx=2, pady=2)
        ctk.CTkButton(btn_frame, text="VOC中文", command=lambda: self.entry_classes.delete(0, "end") or self.entry_classes.insert(0, "飞机, 自行车, 鸟, 船, 狗, 猫, 人, 车"),
                      fg_color="transparent", border_width=1).grid(row=0, column=1, padx=2, pady=2)

        lbl2 = ctk.CTkLabel(sidebar, text="置信度阈值", font=ctk.CTkFont(weight="bold"))
        lbl2.grid(row=5, column=0, padx=20, pady=(10, 2), sticky="w")

        self.slider_conf = ctk.CTkSlider(sidebar, from_=0.05, to=0.9,
                                          number_of_steps=17, command=self._on_slider)
        self.slider_conf.set(0.1)
        self.slider_conf.grid(row=6, column=0, padx=20, pady=(2, 0), sticky="ew")
        self.lbl_conf = ctk.CTkLabel(sidebar, text="0.10", text_color="gray60")
        self.lbl_conf.grid(row=7, column=0, padx=20, pady=(0, 5), sticky="w")

        self.btn_run = ctk.CTkButton(sidebar, text="🚀 开始检测",
                                      font=ctk.CTkFont(size=16, weight="bold"),
                                      command=self._run_detection, height=45,
                                      fg_color="#2fa572", hover_color="#258c5e")
        self.btn_run.grid(row=8, column=0, padx=20, pady=10, sticky="ew")

        self.lbl_status = ctk.CTkLabel(sidebar, text="就绪", text_color="gray60", anchor="w")
        self.lbl_status.grid(row=9, column=0, padx=20, pady=5, sticky="ew")

        lbl3 = ctk.CTkLabel(sidebar, text="检测结果", font=ctk.CTkFont(weight="bold"))
        lbl3.grid(row=10, column=0, padx=20, pady=(10, 2), sticky="w")

        self.result_list = ctk.CTkTextbox(sidebar, font=("Consolas", 12), state="disabled", height=200)
        self.result_list.grid(row=11, column=0, padx=20, pady=5, sticky="ew")

        self.btn_save = ctk.CTkButton(sidebar, text="💾 保存结果",
                                      command=self._save, fg_color="transparent", border_width=1)
        self.btn_save.grid(row=12, column=0, padx=20, pady=(5, 20), sticky="ew")

    def _on_slider(self, val):
        self.lbl_conf.configure(text=f"{val:.2f}")

    def _build_canvas(self):
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.drop_frame = ctk.CTkFrame(self.canvas_frame, fg_color="#1a1a2e", corner_radius=10)
        self.drop_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.drop_frame.grid_rowconfigure(0, weight=1)
        self.drop_frame.grid_columnconfigure(0, weight=1)

        self.lbl_image = ctk.CTkLabel(self.drop_frame,
                                       text="拖放图片到此处\n或双击选择图片",
                                       font=ctk.CTkFont(size=16), text_color="gray60")
        self.lbl_image.grid(row=0, column=0)

        self.drop_frame.drop_target_register("DND_Files")
        self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)
        self.lbl_image.bind("<Double-Button-1>", lambda e: self._open_file())

    def _on_drop(self, target):
        files = self.drop_frame.tk.splitlist(target)
        if files:
            path = files[0]
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self._load_image(path)

    def _open_file(self):
        import tkinter.filedialog as fd
        path = fd.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            self.lbl_status.configure(text=f"无法读取: {path}")
            return
        self.current_image = img
        self.detections = []
        self._display_image(img)
        self.lbl_status.configure(text=f"已加载: {os.path.basename(path)}")
        self._update_results([])

    def _display_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        cw, ch = self.drop_frame.winfo_width(), self.drop_frame.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 700, 600
        scale = min(cw / iw, ch / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
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
        conf_thresh = float(self.slider_conf.get())
        self.lbl_status.configure(text="检测中...")
        self.btn_run.configure(state="disabled", text="检测中...")

        def thread():
            try:
                self.detector.encode_texts(classes)
                dets = self.detector.detect(self.current_image.copy(), conf_threshold=conf_thresh)
                self.detections = dets
            except Exception as e:
                print(f"Error: {e}")
                self.detections = []
            finally:
                self.after(0, self._on_done)

        threading.Thread(target=thread, daemon=True).start()

    def _on_done(self):
        self.btn_run.configure(state="normal", text="🚀 开始检测")
        if not self.detections:
            self.lbl_status.configure(text="未检测到目标，尝试降低阈值")
            return
        img = self.current_image.copy()
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except:
            font = font_s = ImageFont.load_default()
        for det in self.detections:
            x1, y1, x2, y2 = det["box"]
            c = det["color"]
            label = f"{det['class']} {det['conf']:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
            draw.rectangle([x1, y1-22, x1+len(label)*9+6, y1], fill=c)
            draw.text((x1+3, y1-20), label, fill=(255,255,255), font=font_s)
        self._display_image(img)
        counter = Counter(d["class"] for d in self.detections)
        self.lbl_status.configure(text=f"检测完成，共 {len(self.detections)} 个目标")
        self._update_results(counter)

    def _update_results(self, counter):
        self.result_list.configure(state="normal")
        self.result_list.delete("1.0", "end")
        if not counter:
            self.result_list.insert("end", "暂无结果\n")
        else:
            for cls, cnt in sorted(counter.items(), key=lambda x: -x[1]):
                self.result_list.insert("end", f"  {cls:15s} × {cnt}\n")
        self.result_list.configure(state="disabled")

    def _save(self):
        if self.current_image is None:
            return
        import tkinter.filedialog as fd
        path = fd.asksaveasfilename(defaultextension=".jpg",
                                      filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
                                      initialfile=f"cn_detect_{uuid.uuid4().hex[:6]}.jpg")
        if not path:
            return
        img = self.current_image.copy()
        if self.detections:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            except:
                font = font_s = ImageFont.load_default()
            for det in self.detections:
                x1, y1, x2, y2 = det["box"]
                c = det["color"]
                label = f"{det['class']} {det['conf']:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
                draw.rectangle([x1, y1-22, x1+len(label)*9+6, y1], fill=c)
                draw.text((x1+3, y1-20), label, fill=(255,255,255), font=font_s)
        cv2.imwrite(path, img)
        self.lbl_status.configure(text=f"已保存: {os.path.basename(path)}")


if __name__ == "__main__":
    app = DemoAppCN()
    app.mainloop()
