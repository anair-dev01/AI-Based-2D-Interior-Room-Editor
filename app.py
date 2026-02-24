#interior_env\Scripts\activate
#cd AI_Interior_Editor
#streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os
import random

st.set_page_config(page_title="AI Interior Editor", layout="wide")
st.title("AI Interior Editor")

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    return YOLO("yolov8n-seg.pt")

model = load_model()

# ---------------- FURNITURE HELPERS ---------------- #

def get_random_furniture(category):
    folder_path = f"assets/{category.lower()}"
    if not os.path.exists(folder_path):
        return None
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    if not files:
        return None
    selected = random.choice(files)
    return os.path.join(folder_path, selected)

def overlay_furniture(base_img, furniture_path, bbox):

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    furniture = cv2.imread(furniture_path, cv2.IMREAD_UNCHANGED)

    if furniture is None:
        return base_img, bbox

    # FIX COLOR (BGR → RGB)
    if furniture.shape[2] == 4:
        furniture = cv2.cvtColor(furniture, cv2.COLOR_BGRA2RGBA)
    else:
        furniture = cv2.cvtColor(furniture, cv2.COLOR_BGR2RGB)

    # Resize slightly smaller
    scale = 0.95
    width = int(width * scale)
    height = int(height * scale)
    furniture = cv2.resize(furniture, (width, height))

    # Center inside bbox
    x1 = x1 + (x2 - x1 - width) // 2
    y1 = y1 + (y2 - y1 - height) // 2
    x2 = x1 + width
    y2 = y1 + height

    if furniture.shape[2] == 4:
        alpha = furniture[:, :, 3] / 255.0
        furniture_rgb = furniture[:, :, :3]
    else:
        alpha = np.ones((height, width))
        furniture_rgb = furniture

    furniture_rgb = cv2.GaussianBlur(furniture_rgb, (3, 3), 0)

    roi = base_img[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (
            alpha * furniture_rgb[:, :, c] +
            (1 - alpha) * roi[:, :, c]
        )

    base_img[y1:y2, x1:x2] = roi

    return base_img, (x1, y1, x2, y2)

def add_shadow(base_img, bbox):

    x1, y1, x2, y2 = bbox
    shadow_height = int((y2 - y1) * 0.2)

    shadow_layer = np.zeros_like(base_img)

    center_x = (x1 + x2) // 2
    bottom_y = y2

    cv2.ellipse(
        shadow_layer,
        (center_x, bottom_y),
        ((x2 - x1)//2, shadow_height),
        0, 0, 180,
        (0, 0, 0),
        -1
    )

    shadow_layer = cv2.GaussianBlur(shadow_layer, (51, 51), 0)
    base_img = cv2.addWeighted(base_img, 1, shadow_layer, 0.3, 0)

    return base_img

# ---------------- SESSION STATE INIT ---------------- #

for key in [
    "image_np",
    "results",
    "masks",
    "object_list",
    "remove_result",
    "replace_result"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload a Room Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Reset if new image
    if (
        st.session_state.image_np is None
        or not np.array_equal(st.session_state.image_np, image_np)
    ):
        st.session_state.image_np = image_np
        st.session_state.results = None
        st.session_state.masks = None
        st.session_state.object_list = None
        st.session_state.remove_result = None
        st.session_state.replace_result = None

    # -------- DISPLAY ORIGINAL IMAGE (ALWAYS SAFE) -------- #

    st.image(
        st.session_state.image_np,
        caption="Original Image (Unchanged)",
        use_container_width=True
    )

    # ---------------- DETECTION ---------------- #

    if st.button("Detect Furniture"):
        with st.spinner("Detecting objects..."):
            results = model(st.session_state.image_np)
            result = results[0]

            st.session_state.results = result

            if result.masks is not None and len(result.masks.data) > 0:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                class_ids = boxes.cls.cpu().numpy().astype(int)
                class_names = result.names

                object_list = []
                for i, class_id in enumerate(class_ids):
                    label = class_names[class_id]
                    object_list.append(f"{i} - {label}")

                st.session_state.masks = masks
                st.session_state.object_list = object_list
            else:
                st.session_state.masks = None
                st.session_state.object_list = None

# ---------------- SHOW DETECTION ---------------- #

if st.session_state.results is not None:

    annotated_image = st.session_state.results.plot()
    st.image(annotated_image, caption="Detected Objects", use_container_width=True)

    if st.session_state.masks is not None:

        # ---------------- REMOVE ---------------- #

        st.subheader("Remove Object")

        selected_object = st.selectbox(
            "Select object to remove",
            st.session_state.object_list
        )

        index = int(selected_object.split(" - ")[0])
        mask = st.session_state.masks[index]

        binary_mask = (mask * 255).astype("uint8")
        binary_mask = cv2.resize(
            binary_mask,
            (
                st.session_state.image_np.shape[1],
                st.session_state.image_np.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        binary_mask = (binary_mask > 10).astype("uint8") * 255

        if st.button("Remove Selected Object"):

            inpainted = cv2.inpaint(
                st.session_state.image_np.copy(),
                binary_mask,
                7,
                cv2.INPAINT_TELEA
            )

            inpainted = cv2.inpaint(
                inpainted,
                binary_mask,
                3,
                cv2.INPAINT_NS
            )

            st.session_state.remove_result = inpainted
            st.session_state.replace_result = None

        if st.session_state.remove_result is not None:
            st.image(st.session_state.remove_result,
                     caption="After Object Removal",
                     use_container_width=True)

        # ---------------- REPLACE ---------------- #

        st.subheader("Replace Object")

        replacement_option = st.selectbox(
            "Choose replacement object:",
            ["Chair", "Table", "Lamp", "Plant"]
        )

        if st.button("Replace Selected Object"):

            removed = cv2.inpaint(
                st.session_state.image_np.copy(),
                binary_mask,
                7,
                cv2.INPAINT_TELEA
            )

            y_indices, x_indices = np.where(binary_mask > 0)

            if len(y_indices) > 0:

                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()

                bbox = (x_min, y_min, x_max, y_max)

                furniture_path = get_random_furniture(replacement_option)

                if furniture_path is not None:

                    replaced_img, new_bbox = overlay_furniture(
                        removed,
                        furniture_path,
                        bbox
                    )

                    replaced_img = add_shadow(replaced_img, new_bbox)

                    st.session_state.replace_result = replaced_img
                    st.session_state.remove_result = None
                else:
                    st.warning("No PNG found for this category.")

        if st.session_state.replace_result is not None:
            st.image(st.session_state.replace_result,
                     caption="Object Replaced",
                     use_container_width=True)

    else:
        st.warning("No segmentation masks found.")

elif uploaded_file is None:
    st.info("Please upload an image to begin.")








