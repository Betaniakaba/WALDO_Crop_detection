import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import random
import torch
import torchvision
from ultralytics import YOLO

# --- Helper Functions ---

def split_image(image, tile_size=(960, 960), overlap=0.1):
    height, width, _ = image.shape
    tile_height, tile_width = tile_size

    stride_h = int(tile_height * (1 - overlap))
    stride_w = int(tile_width * (1 - overlap))

    tiles = []
    for y in range(0, max(1, height - tile_height + stride_h), stride_h):
        for x in range(0, max(1, width - tile_width + stride_w), stride_w):
            # Clamp to image boundaries
            y1 = min(y, height - tile_height) if height > tile_height else 0
            x1 = min(x, width - tile_width) if width > tile_width else 0
            y2 = min(y1 + tile_height, height)
            x2 = min(x1 + tile_width, width)

            tile = image[y1:y2, x1:x2, :]

            # If tile is smaller than tile_size (at edges), pad it
            if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_height - tile.shape[0], 0, tile_width - tile.shape[1],
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

            tiles.append(((x1, y1), tile))

            if x1 + tile_width >= width: break
        if y1 + tile_height >= height: break

    return tiles

# --- Model Download Utility ---

def download_file(url, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with st.spinner(f"Downloading {os.path.basename(local_path)}..."):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"Failed to download model from {url}")

def get_model_path(option):
    models = {
        "WALDO v3.0 (Large - Best)": {
            "url": "https://huggingface.co/StephanST/WALDO30/resolve/main/WALDO30_yolov8l_640x640.pt",
            "local": "models/WALDO30_yolov8l_640x640.pt"
        },
        "WALDO v3.0 (Medium)": {
            "url": "https://huggingface.co/StephanST/WALDO30/resolve/main/WALDO30_yolov8m_640x640.pt",
            "local": "models/WALDO30_yolov8m_640x640.pt"
        }
    }

    model_info = models.get(option)
    download_file(model_info["url"], model_info["local"])
    return model_info["local"]

# --- Detection Logic ---

def run_inference(image, model_path, conf_threshold, tile_size_val, use_tiling, iou_threshold=0.45):
    model = YOLO(model_path)
    names = model.names
    colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names.values()}

    all_boxes = [] # Will store [x1, y1, x2, y2, conf, cls_id]

    if not use_tiling:
        results = model(image, conf=conf_threshold)[0]
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            all_boxes.append([*b, float(box.conf[0]), int(box.cls[0])])
    else:
        tile_size = (tile_size_val, tile_size_val)
        tiles = split_image(image, tile_size=tile_size, overlap=0.2)

        for (offset_x, offset_y), tile in tiles:
            results = model(tile, conf=conf_threshold)[0]
            for box in results.boxes:
                b = box.xyxy[0].cpu().numpy()
                # Global coordinates
                global_box = [
                    b[0] + offset_x,
                    b[1] + offset_y,
                    b[2] + offset_x,
                    b[3] + offset_y,
                    float(box.conf[0]),
                    int(box.cls[0])
                ]
                all_boxes.append(global_box)

    if not all_boxes:
        return image, {}

    # Convert to tensor for NMS
    boxes_tensor = torch.tensor([b[:4] for b in all_boxes])
    scores_tensor = torch.tensor([b[4] for b in all_boxes])
    classes_tensor = torch.tensor([b[5] for b in all_boxes])

    # Batched NMS (keeps boxes separate for different classes)
    max_wh = 4096
    offsets = classes_tensor.to(boxes_tensor) * max_wh
    boxes_for_nms = boxes_tensor + offsets[:, None]
    keep_indices = torchvision.ops.nms(boxes_for_nms, scores_tensor, iou_threshold)

    final_boxes = [all_boxes[i] for i in keep_indices]

    # Draw results
    res_img = image.copy()
    category_count = {}

    for b in final_boxes:
        x1, y1, x2, y2, conf, cls_id = b
        name = names[cls_id]
        color = colors[name]

        cv2.rectangle(res_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(res_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

        category_count[name] = category_count.get(name, 0) + 1

    return res_img, category_count

# --- Streamlit UI ---

st.set_page_config(page_title="WALDO Crop Detector", page_icon="🚜", layout="wide")

st.title("🚜 WALDO Crop Detector")
st.markdown("""
This app uses the **WALDO v3.0** model with **Sliced Inference** to detect small objects in high-resolution overhead imagery.
""")

st.sidebar.header("Settings")
model_option = st.sidebar.selectbox("Model Version", ("WALDO v3.0 (Large - Best)", "WALDO v3.0 (Medium)"))
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
tile_size_val = st.sidebar.selectbox("Tile Size", (640, 960, 1024), index=1)
use_tiling = st.sidebar.checkbox("Use Sliced Inference", value=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))
    col1, col2 = st.columns(2)
    with col1: st.image(image, caption='Original', use_column_width=True)

    if st.button('Run Detection'):
        model_path = get_model_path(model_option)
        with st.spinner('Processing...'):
            final_img, counts = run_inference(img_array, model_path, conf_threshold, tile_size_val, use_tiling)
        with col2: st.image(final_img, caption='Detections', use_column_width=True)

        if counts:
            st.subheader("Summary")
            summary_text = ", ".join([f"**{n}:** {c}" for n, c in counts.items()])
            st.markdown(summary_text)
        else:
            st.info("No objects detected.")
