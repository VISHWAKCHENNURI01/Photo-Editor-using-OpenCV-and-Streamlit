import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo Editor using OpenCV", layout="wide")
st.title("📸 Photo Editor using OpenCV and Streamlit")
st.write("Upload an image and apply different image processing techniques.")

# ----------------------------
# Helper Functions
# ----------------------------

def ensure_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def apply_blur(image, ksize):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def warm_filter(image, intensity):
    img = image.copy().astype(np.float32)
    img[:, :, 2] = np.clip(img[:, :, 2] + intensity, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] - intensity / 2, 0, 255)
    return img.astype(np.uint8)

def sharpen(image, strength):
    kernel = np.array([[0, -1, 0], [-1, 5 + strength, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def portrait_blur(image):
    image = ensure_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(gray, 1.2, 5)
    blurred = cv2.GaussianBlur(image, (31, 31), 0)
    result = blurred.copy()
    for (x, y, w, h) in faces:
        result[y:y+h, x:x+w] = image[y:y+h, x:x+w]
    return result

def edge_detection(image, t1, t2):
    image = ensure_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def sketch_effect(image):
    image = ensure_rgb(image)
    gray, sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def cartoon_effect(image):
    image = ensure_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def negative(image):
    return 255 - image

def sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    return np.clip(cv2.transform(image, kernel), 0, 255).astype(np.uint8)

def emboss(image):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.convertScaleAbs(cv2.filter2D(image, -1, kernel))

def hdr(image):
    return cv2.detailEnhance(ensure_rgb(image), sigma_s=12, sigma_r=0.15)

def denoise(image):
    return cv2.fastNlMeansDenoisingColored(ensure_rgb(image), None, 10, 10, 7, 21)

def rotate(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def download_image(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.sidebar.subheader("Resize")
width = st.sidebar.slider("Width", 100, 1500, 800)
height = st.sidebar.slider("Height", 100, 1500, 600)

st.sidebar.subheader("Brightness & Contrast")
brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)

st.sidebar.subheader("Basic Filters")
grayscale = st.sidebar.checkbox("Grayscale")
blur = st.sidebar.slider("Blur", 0, 25, 0)
warm = st.sidebar.slider("Warm Filter", 0, 100, 0)
sharpen_strength = st.sidebar.slider("Sharpen", 0, 5, 0)
portrait = st.sidebar.checkbox("Portrait Background Blur")

st.sidebar.subheader("Extra Effects")
edge = st.sidebar.checkbox("Edge Detection")
if edge:
    t1 = st.sidebar.slider("Threshold 1", 50, 255, 100)
    t2 = st.sidebar.slider("Threshold 2", 50, 255, 200)
else:
    t1, t2 = 100, 200

sketch = st.sidebar.checkbox("Pencil Sketch")
cartoon = st.sidebar.checkbox("Cartoon")
negative_img = st.sidebar.checkbox("Negative")
sepia_img = st.sidebar.checkbox("Sepia")
emboss_img = st.sidebar.checkbox("Emboss")
hdr_img = st.sidebar.checkbox("HDR")
denoise_img = st.sidebar.checkbox("Denoise")

st.sidebar.subheader("Transform")
angle = st.sidebar.slider("Rotate", 0, 360, 0)
flip = st.sidebar.selectbox("Flip", ("None", "Horizontal", "Vertical"))

# ----------------------------
# Main Area
# ----------------------------

if uploaded_file is not None:
    raw = Image.open(uploaded_file).convert("RGB")
    image = np.array(raw)
    edited = image.copy()

    edited = cv2.resize(edited, (width, height))
    edited = adjust_brightness_contrast(edited, brightness, contrast)

    if grayscale:
        edited = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
        edited = cv2.cvtColor(edited, cv2.COLOR_GRAY2RGB)
    if blur > 0:
        edited = apply_blur(edited, blur)
    if warm > 0:
        edited = warm_filter(edited, warm)
    if sharpen_strength > 0:
        edited = sharpen(edited, sharpen_strength)
    if portrait:
        edited = portrait_blur(edited)
    if edge:
        edited = edge_detection(edited, t1, t2)
    if sketch:
        edited = sketch_effect(edited)
    if cartoon:
        edited = cartoon_effect(edited)
    if negative_img:
        edited = negative(edited)
    if sepia_img:
        edited = sepia(edited)
    if emboss_img:
        edited = emboss(edited)
    if hdr_img:
        edited = hdr(edited)
    if denoise_img:
        edited = denoise(edited)
    if angle != 0:
        edited = rotate(edited, angle)
    if flip == "Horizontal":
        edited = cv2.flip(edited, 1)
    if flip == "Vertical":
        edited = cv2.flip(edited, 0)

    st.subheader("Image Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(edited, caption="Edited Image", use_column_width=True)

    st.download_button(
        label="📥 Download Edited Image",
        data=download_image(edited),
        file_name="edited_image.png",
        mime="image/png"
    )

else:
    st.info("Please upload an image.")
