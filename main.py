import pandas as pd
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
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import openpyxl

# ==================== THÊM ĐƯỜNG DẪN EFFICIENTDET ====================
sys.path.append(os.path.join(os.path.dirname(__file__), 'Yet-Another-EfficientDet-Pytorch'))
from efficientdet.utils import BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
from utils.utils import preprocess, invert_affine, postprocess

# ==================== KHỞI TẠO SESSION STATE ====================
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.result_img = None
    st.session_state.boxes = None
    st.session_state.scores = None
    st.session_state.labels = None
    st.session_state.class_names = None
    st.session_state.stats_df = None
    st.session_state.model_choice = None
    st.session_state.uploaded_file = None

# ==================== CÀI ĐẶT EFFICIENTDET ====================
compound_coef = 1
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.2
use_cuda = torch.cuda.is_available()
CLASS_NAMES = ["comedones", "nodules", "papules", "pustules"]

@st.cache_resource
def load_efficientdet():
    try:
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(CLASS_NAMES),
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

def predict_efficientdet(pil_image, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name
        pil_image.save(tmp_path)

    try:
        ori_imgs, framed_imgs, framed_metas = preprocess(tmp_path, max_size=512)
        os.unlink(tmp_path)

        x = torch.stack([torch.from_numpy(fi).cuda() if use_cuda else torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            _, regression, classification, anchors = model(x)
            out = postprocess(x, anchors, regression, classification,
                              regressBoxes, clipBoxes, threshold, iou_threshold)
            out = invert_affine(framed_metas, out)
        return out
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.error(f"EfficientDet inference error: {e}")
        return []

# ==================== LOAD CÁC MODEL KHÁC ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_faster_rcnn():
    def build_model(num_classes):
        densenet = torchvision.models.densenet121(weights="DEFAULT")
        backbone = densenet.features
        backbone.out_channels = 1024
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                          aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_score_thresh=0.01, box_nms_thresh=0.3)
        return model

    model = build_model(5)  # background + 4 classes
    try:
        model.load_state_dict(torch.load("fasterrcnn_best.pth", map_location=device))
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

model_rcnn = load_faster_rcnn()
model_yolo = load_yolov8()
model_effdet = load_efficientdet()

rcnn_class_names = ["background"] + CLASS_NAMES
yolo_class_names = CLASS_NAMES.copy()
effdet_class_names = CLASS_NAMES.copy()

# ==================== HÀM THỐNG KÊ ====================
def count_predictions(labels, class_names):
    if len(labels) == 0:
        return pd.DataFrame({"Acne type": class_names, "Count": [0] * len(class_names)})
    labels = labels.to(torch.long)
    unique, counts = torch.unique(labels, return_counts=True)
    count_dict = {int(idx.item()): int(cnt.item()) for idx, cnt in zip(unique, counts)}
    df = pd.DataFrame({
        "Acne type": class_names,
        "Count": [count_dict.get(i, 0) for i in range(len(class_names))]
    })
    df = df[df["Count"] > 0].reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame({"Acne type": class_names, "Count": [0] * len(class_names)})
    return df

# ==================== HÀM VẼ ====================
def draw_predictions(image_pil, boxes, scores, labels, score_thresh, iou_thresh, class_names):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # chuẩn hoá
    if isinstance(boxes, (list, np.ndarray)):
        boxes = torch.tensor(boxes) if len(boxes) > 0 else torch.empty((0, 4))
        scores = torch.tensor(scores) if len(scores) > 0 else torch.empty((0,))
        labels = torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.empty((0,), dtype=torch.long)
    else:
        boxes = boxes.clone().detach()
        scores = scores.clone().detach()
        labels = labels.clone().detach().to(torch.long)

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) > 0:
        keep_nms = nms(boxes, scores, iou_thresh)
        boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_idx = int(label.item())
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"Class {cls_idx}"
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1 - 22), f"{cls_name}: {score:.2f}", fill="lime", font=font)
    return image_pil

# ==================== STREAMLIT UI ====================
st.title("Acne Detection - 3 Models Comparison")

model_choice = st.selectbox("Select model:", ["Faster R-CNN", "YOLOv8", "EfficientDet"])

col1, col2 = st.columns(2)
with col1:
    score_thresh = st.slider("Confidence threshold", 0.3, 0.95, 0.65, 0.05)
with col2:
    iou_thresh = st.slider("NMS IoU", 0.1, 0.7, 0.3, 0.05)

uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lưu file vào session để không mất khi rerun
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner(f"Running {model_choice}..."):
            # Reset
            st.session_state.prediction_done = False
            result_img = image.copy()
            boxes, scores, labels = [], [], []
            class_names = []

            # ---------- XÁC ĐỊNH CLASS_NAMES ----------
            if model_choice == "Faster R-CNN":
                class_names = rcnn_class_names
            elif model_choice == "YOLOv8":
                class_names = yolo_class_names
            elif model_choice == "EfficientDet":
                class_names = effdet_class_names

            # ---------- FASTER R-CNN ----------
            if model_choice == "Faster R-CNN" and model_rcnn is not None:
                img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model_rcnn(img_tensor)[0]
                keep = pred['scores'].cpu() >= score_thresh
                boxes = pred['boxes'].cpu()[keep]
                scores = pred['scores'].cpu()[keep]
                labels = pred['labels'].cpu()[keep] - 1   # chuyển về 0‑3

                if len(boxes) > 0:
                    keep_nms = nms(boxes, scores, iou_thresh)
                    boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

                result_img = draw_predictions(result_img, boxes, scores, labels + 1,  # +1 để vẽ đúng tên
                                              score_thresh, iou_thresh, class_names)
                stats_df = count_predictions(labels, CLASS_NAMES)

            # ---------- YOLOv8 ----------
            elif model_choice == "YOLOv8" and model_yolo is not None:
                results = model_yolo.predict(source=image, conf=score_thresh,
                                            iou=iou_thresh, verbose=False, imgsz=640)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu()
                    scores = results.boxes.conf.cpu()
                    labels = results.boxes.cls.cpu().int()
                    keep = scores >= score_thresh
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                    result_img = draw_predictions(result_img, boxes, scores, labels,
                                                  score_thresh, iou_thresh, class_names)
                    stats_df = count_predictions(labels, CLASS_NAMES)
                else:
                    st.warning("YOLOv8: No acne detected.")
                    stats_df = pd.DataFrame({"Acne type": CLASS_NAMES, "Count": [0] * 4})

            # ---------- EFFICIENTDET ----------
            elif model_choice == "EfficientDet" and model_effdet is not None:
                preds = predict_efficientdet(image, model_effdet)
                if preds and len(preds[0]['rois']) > 0:
                    boxes = torch.tensor(preds[0]['rois'])
                    scores = torch.tensor(preds[0]['scores'])
                    labels = torch.tensor(preds[0]['class_ids'])
                    keep = scores >= score_thresh
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                    if len(boxes) > 0:
                        keep_nms = nms(boxes, scores, iou_thresh)
                        boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]
                    result_img = draw_predictions(result_img, boxes, scores, labels,
                                                  score_thresh, iou_thresh, class_names)
                    stats_df = count_predictions(labels, CLASS_NAMES)
                else:
                    st.warning("EfficientDet: No acne detected.")
                    stats_df = pd.DataFrame({"Acne type": CLASS_NAMES, "Count": [0] * 4})

            # ---------- LƯU KẾT QUẢ ----------
            st.session_state.result_img = result_img
            st.session_state.boxes = boxes
            st.session_state.scores = scores
            st.session_state.labels = labels
            st.session_state.class_names = class_names
            st.session_state.stats_df = stats_df
            st.session_state.model_choice = model_choice
            st.session_state.prediction_done = True

        st.success("Predict hoàn tất!")

    # ==================== HIỂN THỊ KẾT QUẢ (SAU KHI ĐÃ PREDICT) ====================
    if st.session_state.prediction_done:
        st.image(st.session_state.result_img,
                 caption=f"Result - {st.session_state.model_choice}",
                 use_container_width=True)

        # ---- Tải ảnh ----
        buf = io.BytesIO()
        st.session_state.result_img.save(buf, format="PNG")
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name=f"result_{st.session_state.model_choice.lower().replace(' ', '_')}.png",
            mime="image/png",
            key="download_image"
        )

        # ---- Thống kê ----
        stats_df = st.session_state.stats_df
        total = stats_df["Count"].sum()
        st.subheader(f"Acne Detection Summary ({st.session_state.model_choice})")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Table of acne count**")
            st.dataframe(stats_df, use_container_width=True)
        with col2:
            st.write("**Graph**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(stats_df["Acne type"], stats_df["Count"], color='skyblue', edgecolor='black')
            ax.set_ylabel("Count")
            ax.set_title("Acne type")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        st.success(f"**Total: {total} acnes** detected.")

        # ---- Tải Excel ----
        if len(st.session_state.boxes) > 0:
            detail_data = []
            for box, score, label in zip(st.session_state.boxes,
                                         st.session_state.scores,
                                         st.session_state.labels):
                x1, y1, x2, y2 = map(int, box.tolist())
                cls_idx = int(label.item())
                if st.session_state.model_choice == "Faster R-CNN":
                    cls_idx += 1                     # vì đã trừ 1 ở trên
                cls_name = (st.session_state.class_names[cls_idx]
                            if 0 <= cls_idx < len(st.session_state.class_names)
                            else "Unknown")
                detail_data.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": cls_name,
                    "confidence": round(float(score.item()), 4),
                })
            detail_df = pd.DataFrame(detail_data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                detail_df.to_excel(writer, sheet_name='Detections', index=False)
                stats_df.to_excel(writer, sheet_name='Summary', index=False)
            output.seek(0)

            st.download_button(
                label="Download Excel file (details + summary)",
                data=output.getvalue(),
                file_name=f"acne_detection_{st.session_state.model_choice.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
        else:
            st.info("Không có phát hiện nào để xuất Excel.")