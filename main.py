import streamlit as st
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, nms
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import numpy as np
import subprocess
import os
import sys
import tempfile

# Thêm đường dẫn cho EfficientDet
sys.path.append(os.path.join(os.path.dirname(__file__), 'Yet-Another-EfficientDet-Pytorch'))

from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from utils.utils import preprocess, invert_affine, postprocess

# ==================== CÀI ĐẶT EFFICIENTDET ====================
compound_coef = 1
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.2
use_cuda = torch.cuda.is_available()

obj_list = ["comedones", "nodules", "papules", "pustules"]  # 4 lớp

@st.cache_resource
def load_efficientdet():
    try:
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        model.load_state_dict(torch.load('efficientdet-d1_19_9900.pth', map_location='cpu'))
        model.eval()
        if use_cuda:
            model = model.cuda()
        st.success("EfficientDet loaded!")
        return model
    except Exception as e:
        st.error(f"EfficientDet Load error: {e}")
        return None

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Hàm inference cho EfficientDet
def predict_efficientdet(pil_image, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name
        pil_image.save(tmp_path)

    try:
        ori_imgs, framed_imgs, framed_metas = preprocess(tmp_path, max_size=512)
        os.unlink(tmp_path)  # Xóa file tạm

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            _, regression, classification, anchors = model(x)
            out = postprocess(
                x, anchors, regression, classification,
                regressBoxes, clipBoxes,
                threshold, iou_threshold
            )
            out = invert_affine(framed_metas, out)
        return out
    except:
        os.unlink(tmp_path)
        return []

# ==================== LOAD CÁC MODEL KHÁC ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_faster_rcnn():
    def build_model(num_classes):
        densenet = torchvision.models.densenet121(weights="DEFAULT")
        backbone = densenet.features
        backbone.out_channels = 1024
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0)))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler, box_score_thresh=0.01, box_nms_thresh=0.3)
        return model

    model = build_model(5)  # 5 classes: background + 4 mụn
    try:
        model.load_state_dict(torch.load("fasterrcnn.pth", map_location=device))
        st.success("Faster R-CNN loaded!")
    except Exception as e:
        st.error(f"RCNN Load error: {e}")
        return None
    model.to(device).eval()
    return model

@st.cache_resource
def load_yolov8():
    try:
        model = YOLO(r'yolo.pt')
        st.success("YOLOv8 loaded!")
        return model
    except Exception as e:
        st.error(f"YOLO Load error: {e}")
        return None

# Load models
model_rcnn = load_faster_rcnn()
model_yolo = load_yolov8()
model_effdet = load_efficientdet()

# Class names
rcnn_class_names = ["background", "comedones", "nodules", "papules", "pustules"]
yolo_class_names = ["comedones", "nodules", "papules", "pustules"]
effdet_class_names = obj_list

# ==================== HÀM VẼ CHUNG ====================
def draw_predictions(image_pil, boxes, scores, labels, score_thresh, iou_thresh, class_names):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    if isinstance(boxes, list):
        boxes = torch.tensor(boxes) if len(boxes) > 0 else torch.empty((0, 4))
        scores = torch.tensor(scores) if len(scores) > 0 else torch.empty((0,))
        labels = torch.tensor(labels) if len(labels) > 0 else torch.empty((0,), dtype=torch.long)
    else:
        boxes = torch.from_numpy(boxes) if isinstance(boxes, np.ndarray) else boxes
        scores = torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) > 0:
        keep_nms = nms(boxes, scores, iou_thresh)
        boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_idx = int(label.item())
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"Class {cls_idx}"

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        text = f"{cls_name}: {score:.2f}"
        draw.text((x1, y1 - 22), text, fill="lime", font=font)

    return image_pil

# ==================== STREAMLIT APP ====================
st.title("Phát hiện mụn – So sánh 3 Model")

# Chọn model
model_choice = st.selectbox("Chọn model:", ["Faster R-CNN", "YOLOv8", "EfficientDet"])

# Slider chung
col1, col2 = st.columns(2)
with col1:
    score_thresh = st.slider("Ngưỡng tin cậy", 0.3, 0.95, 0.65, 0.05)
with col2:
    iou_thresh = st.slider("NMS IoU", 0.1, 0.7, 0.3, 0.05)

uploaded_file = st.file_uploader("Tải ảnh:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("Dự đoán"):
        with st.spinner(f"Đang chạy {model_choice}..."):
            result_img = image.copy()

            if model_choice == "Faster R-CNN" and model_rcnn is not None:
                img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_rcnn(img_tensor)[0]
                result_img = draw_predictions(
                    result_img, pred['boxes'].cpu(), pred['scores'].cpu(), pred['labels'].cpu(),
                    score_thresh, iou_thresh, rcnn_class_names
                )

            elif model_choice == "YOLOv8" and model_yolo is not None:
                results = model_yolo.predict(
                    source=image,
                    conf=score_thresh,
                    iou=0.45,
                    verbose=False,
                    imgsz=640
                )[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu()
                    scores = results.boxes.conf.cpu()
                    labels = results.boxes.cls.cpu().int()
                    result_img = draw_predictions(
                        result_img, boxes, scores, labels,
                        score_thresh, iou_thresh, yolo_class_names
                    )
                else:
                    st.warning("YOLOv8: Không phát hiện được mụn nào.")

            elif model_choice == "EfficientDet" and model_effdet is not None:
                preds = predict_efficientdet(image, model_effdet)
                if preds and len(preds[0]['rois']) > 0:
                    boxes = preds[0]['rois']
                    scores = preds[0]['scores']
                    labels = preds[0]['class_ids']
                    result_img = draw_predictions(
                        result_img, boxes, scores, labels,
                        score_thresh, iou_thresh, effdet_class_names
                    )
                else:
                    st.warning("EfficientDet: Không phát hiện được mụn nào.")

            # Hiển thị kết quả
            st.image(result_img, caption=f"Kết quả - {model_choice}", use_container_width=True)

            # Tải ảnh
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(
                "Tải kết quả",
                data=buf.getvalue(),
                file_name=f"result_{model_choice.lower().replace(' ', '_')}.png",
                mime="image/png"
            )