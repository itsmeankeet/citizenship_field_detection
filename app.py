
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image
import tempfile

st.set_page_config(layout="wide", page_title="Citizenship OCR with Nepali Fields")

st.title("🪪 Citizenship OCR Using YOLO + Alignment + Nepali/English Field Detection")

# Set Tesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load both YOLO models
@st.cache_resource
def load_models():
    doc_model = YOLO("./my_model/citizenship_detection/my_model.pt")
    field_model = YOLO("./my_model/field_detection_model/my_model.pt")
    return doc_model, field_model, field_model.names

document_model, field_model, class_names = load_models()

def align_image(image, target, MAX_NUMBER_OF_FEATURES=4000):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_NUMBER_OF_FEATURES)
    kp1, des1 = orb.detectAndCompute(image_gray, None)
    kp2, des2 = orb.detectAndCompute(target_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(des1, des2))
    matches.sort(key=lambda x: x.distance)

    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, _ = target.shape
    aligned_image = cv2.warpPerspective(image, h, (width, height))

    return aligned_image

# OCR with dynamic language selection
def extract_text_from_image(image, lang='eng'):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bordered = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    text = pytesseract.image_to_string(bordered, lang=lang)
    return text.strip()

def get_cropped_field_dict(aligned_image, model, class_names, conf_thresh=0.5):
    results = model(aligned_image)
    boxes_with_conf = []

    for result in results:
        for box in result.boxes:
            if box.conf > conf_thresh:
                boxes_with_conf.append((box, float(box.conf[0])))

    boxes_with_conf.sort(key=lambda x: x[1], reverse=True)
    cropped_dict = {}

    for box, _ in boxes_with_conf:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = aligned_image[y1:y2, x1:x2]
        class_id = int(box.cls[0])
        label = class_names[class_id]

        if label not in cropped_dict:
            cropped_dict[label] = []
        cropped_dict[label].append(cropped)

    return {k: v[0] for k, v in cropped_dict.items()}

# Load template image
template_path = "d:/yolo/align_image_using_template/images/template_images/temp_2.jpg"
target = cv2.imread(template_path)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Upload UI
doc_file = st.file_uploader("Upload Scanned Citizenship Image", type=["jpg", "jpeg", "png"])

if doc_file and st.button("🔍 Detect & Extract Text"):
    # Template image already loaded above
    document = np.array(Image.open(doc_file).convert("RGB"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, cv2.cvtColor(document, cv2.COLOR_RGB2BGR))
        results = document_model(temp_file.name)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_doc = document[y1:y2, x1:x2]
            st.image(cropped_doc, caption="📄 Detected Document Region", use_column_width=True)

            aligned = align_image(cropped_doc, target)
            st.image(aligned, caption="🧭 Aligned Image", use_column_width=True)

            # Use field-level model
            field_dict = get_cropped_field_dict(aligned, field_model, class_names)

            st.subheader("📋 Extracted Fields")
            for label, crop in field_dict.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(crop, caption=f"🖼️ {label}", width=200)
                with col2:
                    lang = 'nep' if any(kw in label.lower() for kw in ['bamsha', 'issued_date', 'जारी']) else 'eng'
                    text = extract_text_from_image(crop, lang=lang)
                    st.text_area(f"✍️ {label}", text, height=100)
