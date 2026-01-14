# app.py
import streamlit as st
import numpy as np
import cv2
import joblib
import os
from io import BytesIO
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage import morphology
import matplotlib.pyplot as plt

st.set_page_config(page_title="DR Grading (Classical)", layout="wide")

# ----------------------------
# Helper: load pipeline
# ----------------------------
MODEL_DIR = "models"  # change if needed
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
FEAT_SCALER_PATH = os.path.join(MODEL_DIR, "feat_scaler.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")

@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")
    models['scaler'] = joblib.load(SCALER_PATH)
    models['pca'] = joblib.load(PCA_PATH)
    models['feat_scaler'] = joblib.load(FEAT_SCALER_PATH)
    models['svm'] = joblib.load(SVM_PATH)
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load models from `{MODEL_DIR}`: {e}")
    st.stop()

scaler = models['scaler']
pca = models['pca']
feat_scaler = models['feat_scaler']
svm = models['svm']

GRADE_NAMES = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']

# ----------------------------
# Preprocessing & feature extraction (must match training)
# ----------------------------
def preprocess_img(img_bgr):
    """Preprocess image: resize -> illumination correction -> equalize Y -> CLAHE (L channel)."""
    img = cv2.resize(img_bgr, (512, 512))
    img_float = img.astype(np.float32)
    background = cv2.GaussianBlur(img_float, (75, 75), 0)
    img_norm = cv2.divide(img_float, background + 1e-6, scale=255)
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_norm, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    img_eq = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
    lab = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return enhanced

def extract_features_from_image(img_bgr):
    """Extract same feature vector used in training."""
    feats = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # GLCM (single distance/angle used in training snippet)
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        for p in props:
            feats.append(float(graycoprops(glcm, p)[0, 0]))
    except Exception:
        feats.extend([0.0]*5)

    # LBP histogram (10 bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(hist.tolist())

    # Wavelet features (haar)
    try:
        coeffs = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs
        for c in [cA, cH, cV, cD]:
            feats.append(float(np.mean(c)))
            feats.append(float(np.std(c)))
    except Exception:
        feats.extend([0.0]*8)

    return np.array(feats).reshape(1, -1)

# ----------------------------
# Lesion visualization helpers (simple heuristics)
# ----------------------------
def segment_vessels(img_bgr):
    green = img_bgr[:, :, 1]
    green = cv2.createCLAHE(2.0, (8, 8)).apply(green)
    bg = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    vessel = cv2.subtract(green, bg)
    _, mask = cv2.threshold(vessel, 15, 255, cv2.THRESH_BINARY)
    # optional skeletonize
    mask = morphology.skeletonize(mask > 0)
    mask = (mask.astype(np.uint8) * 255)
    return mask

def detect_microaneurysms(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7,7), np.uint8))
    _, mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
    # filter by area
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for c in cnts:
        a = cv2.contourArea(c)
        if 5 < a < 120:
            cv2.drawContours(filtered, [c], -1, 255, -1)
    return filtered

def detect_hemorrhages(img_bgr):
    red = img_bgr[:, :, 2]
    bg = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    diff = cv2.subtract(red, bg)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    # remove small noises
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) > 40:
            cv2.drawContours(filtered, [c], -1, 255, -1)
    return filtered

def detect_exudates(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 40, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) > 40:
            cv2.drawContours(filtered, [c], -1, 255, -1)
    return filtered

def detect_macula(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (45,45), 0)
    h, w = gray.shape
    cx, cy = w//2, h//2
    crop = blur[cy-120:cy+120, cx-120:cx+120]
    if crop.size == 0:
        return (cx, cy)
    min_loc = np.unravel_index(np.argmin(crop), crop.shape)
    mac_y = cy - 120 + min_loc[0]
    mac_x = cx - 120 + min_loc[1]
    return (mac_x, mac_y)

# ----------------------------
# Prediction helpers
# ----------------------------
def model_predict_proba(feats_raw):
    """Take raw feature vector -> scaled -> pca -> feat_scale -> predict. Return (pred, confidence_dict)."""
    # scale features (original scaler used during training)
    feats_s = scaler.transform(feats_raw)           # scaler from training pipeline
    feats_p = pca.transform(feats_s)
    feats_f = feat_scaler.transform(feats_p)
    # predict
    if hasattr(svm, "predict_proba"):
        probs = svm.predict_proba(feats_f)[0]
    else:
        # softmax on decision_function
        if hasattr(svm, "decision_function"):
            df = svm.decision_function(feats_f)
            # if binary, make into vector
            if df.ndim == 1:
                # two-class: approximate
                probs = np.vstack([1/(1+np.exp(-df)), 1/(1+np.exp(df))]).T[0]
                # fallback: spread
                probs = np.array([1-probs, probs])
            else:
                # multiclass: softmax
                e = np.exp(df - np.max(df))
                probs = e / e.sum(axis=1, keepdims=True)
                probs = probs[0]
        else:
            # fallback: no confidence
            probs = np.zeros(len(GRADE_NAMES))
            probs[0] = 1.0
    pred = int(np.argmax(probs))
    conf_dict = {GRADE_NAMES[i]: float(probs[i]) for i in range(len(GRADE_NAMES))}
    return pred, conf_dict

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Diabetic Retinopathy (Classical) — Demo")
st.write("Upload a fundus image to see preprocessing, lesion visualization and grade prediction (SVM+PCA).")

col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload fundus image (.jpg/.png)", type=["jpg","jpeg","png"])
    if st.button("Use example image from models folder"):
        # try to use a sample from models folder or a placeholder sample
        # find first jpg in current dir
        sample = None
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    sample = os.path.join(root, f)
                    break
            if sample:
                break
        if sample:
            uploaded = open(sample, "rb")

    st.markdown("---")
    st.write("Model info:")
    st.write(f"- SVM model: {os.path.basename(SVM_PATH)}")
    st.write(f"- PCA components: {getattr(pca,'n_components_', getattr(pca,'n_components',None))}")
    st.write(f"- Feature vector length (raw): {scaler.mean_.shape[0] if hasattr(scaler,'mean_') else 'unknown'}")

with col2:
    st.write("Predicted Grade")
    placeholder = st.empty()

if uploaded is not None:
    # read image bytes
    if hasattr(uploaded, "read"):
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img_bgr = cv2.imread(uploaded)  # path-like object

    if img_bgr is None:
        st.error("Could not read image. Make sure it's a valid image file.")
    else:
        # preprocess and extract features
        pre = preprocess_img(img_bgr)
        feats = extract_features_from_image(pre)  # raw features shape (1, n)
        pred, conf = model_predict_proba(feats)

        # visualization masks
        vessel_mask = segment_vessels(pre)
        ma_mask = detect_microaneurysms(pre)
        hem_mask = detect_hemorrhages(pre)
        ex_mask = detect_exudates(pre)
        mac_x, mac_y = detect_macula(pre)

        # combined overlay (RGB)
        overlay = pre.copy()
        # color masks
        overlay[ma_mask > 0] = [0, 255, 0]      # green microaneurysms
        overlay[hem_mask > 0] = [255, 0, 0]     # red hemorrhages
        overlay[ex_mask > 0] = [0, 0, 255]      # blue exudates
        # draw macula
        cv2.circle(overlay, (mac_x, mac_y), 10, (255, 255, 0), 2)

        # show images side by side
        colA, colB = st.columns(2)
        with colA:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
            st.image(cv2.cvtColor(pre, cv2.COLOR_BGR2RGB), caption="Preprocessed", use_column_width=True)
        with colB:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Lesion overlay (G:MA, R:HE, B:EX)", use_column_width=True)
            st.image(vessel_mask, caption="Vessel skeleton", use_column_width=True)

        # show prediction and confidences
        st.subheader(f"Predicted grade: {pred} — {GRADE_NAMES[pred]}")
        st.write("Confidence (approx.):")
        for gname, pval in conf.items():
            st.write(f"- {gname}: {pval:.3f}")

        # Save result button: combine original and overlay side-by-side
        if st.button("Download result image"):
            # create a side-by-side result
            h1, w1 = img_bgr.shape[:2]
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # resize to same height
            overlay_resized = cv2.resize(overlay_rgb, (w1, h1))
            combined = np.hstack([orig_rgb, overlay_resized])
            _, buf = cv2.imencode(".png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            st.download_button("Download PNG", buf.tobytes(), file_name="dr_result.png", mime="image/png")

st.markdown("---")
st.write("Notes: This demo uses a classical feature-based pipeline (GLCM, LBP, Wavelets, lesion heuristics) with PCA + SVM. Confidence is approximated if `predict_proba` is not available.")
