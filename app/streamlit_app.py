"""Tomato Leaf Doctor – tomato leaf disease detection and treatment advice (MSc project).
Run locally:  streamlit run app/streamlit_app.py
"""
import os, json
import numpy as np
import torch, timm
import torch.nn.functional as F
import matplotlib
import streamlit as st
from PIL import Image
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Tomato Leaf Doctor", layout="wide")

# ---------- 1. Load model, settings and advice (cached: loaded once, not on every click) ----------
@st.cache_resource
def load_everything():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    kb = json.load(open(os.path.join(HERE, "disease_info.json"), encoding="utf-8"))
    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(cfg["classes"]))
    model.load_state_dict(torch.load(os.path.join(HERE, "model.pt"), map_location="cpu"))
    model.eval()
    return cfg, kb, model

CFG, KB, model = load_everything()
CLASSES, THRESHOLD = CFG["classes"], CFG["confidence_threshold"]
preprocess = T.Compose([T.Resize((CFG["img_size"], CFG["img_size"])), T.ToTensor(),
                        T.Normalize(CFG["mean"], CFG["std"])])
JET = matplotlib.colormaps["jet"]

def pretty(name):
    return KB["diseases"][name]["display_name"]

# ---------- 2. Leaf check: enough plant colour (brown / yellow / green)? ----------
def plant_fraction(img):
    hsv = np.asarray(img.resize((128, 128)).convert("HSV")).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]          # PIL hue scale 0-255
    return ((h >= 15) & (h <= 120) & (s > 40) & (v > 40)).mean()

# ---------- 3. Prediction + Grad-CAM heatmap ----------
def predict_with_cam(img):
    x = preprocess(img).unsqueeze(0)
    feats = model.forward_features(x)          # last feature maps
    feats.retain_grad()
    logits = model.forward_head(feats)
    probs = torch.softmax(logits, dim=1)[0]
    cls = int(probs.argmax())
    model.zero_grad()
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probs.detach().numpy(), cls, cam.detach().numpy()

def overlay(img, cam):
    base = np.asarray(img.resize((cam.shape[1], cam.shape[0])), dtype=np.float32) / 255
    heat = JET(cam)[..., :3]
    return (np.clip(0.55 * base + 0.45 * heat, 0, 1) * 255).astype(np.uint8)

# ---------- 4. Advice shown to the farmer ----------
def bullets(items):
    return "\n".join(f"- {x}" for x in items) if items else "- Not needed"

def show_advice(name, conf):
    d = KB["diseases"][name]
    if name == "healthy":
        st.success(f"Healthy leaf ({conf:.0%} confidence). No disease detected.")
        st.markdown("**Keep your crop healthy:**\n" + bullets(d["prevention"]))
        return
    level = {"High": st.error, "Medium": st.warning}.get(d["urgency"], st.info)
    level(f"{d['display_name']} detected ({conf:.0%} confidence). Urgency: {d['urgency']}")
    st.markdown(f"**Type:** {d['type']} | **Cause:** {d['pathogen']}")
    st.markdown("#### Symptoms\n" + bullets(d["symptoms"]))
    st.markdown(f"**How it spreads:** {d['spread']}")
    c1, c2 = st.columns(2)
    c1.markdown("#### What to do now (organic / cultural)\n" + bullets(d["organic_control"]))
    c2.markdown("#### Chemical control (active ingredients)\n" + bullets(d["chemical_control"]))
    st.markdown("#### Prevention for the next crop\n" + bullets(d["prevention"]))
    st.caption(KB["disclaimer"])
    st.caption("Sources: " + "; ".join(KB["sources"]))

# ---------- 5. Page layout ----------
st.title("Tomato Leaf Doctor")
st.write("Upload or take a photo of a tomato leaf to detect disease and get treatment advice. "
         "Detects 9 diseases and healthy leaves.")

source = st.radio("Image source", ["Upload a photo", "Use camera", "Try an example"], horizontal=True)
img = None
if source == "Upload a photo":
    f = st.file_uploader("Tomato leaf photo", type=["jpg", "jpeg", "png"])
    img = Image.open(f) if f else None
elif source == "Use camera":
    f = st.camera_input("Take a photo of one leaf")
    img = Image.open(f) if f else None
else:
    ex_dir = os.path.join(HERE, "examples")
    files = sorted(os.listdir(ex_dir)) if os.path.isdir(ex_dir) else []
    if files:
        choice = st.selectbox("Example image", files)
        img = Image.open(os.path.join(ex_dir, choice))

if img is not None:
    img = img.convert("RGB")
    left, right = st.columns(2)
    left.image(img, caption="Your photo", width="stretch")
    if plant_fraction(img) < 0.05:
        st.error("This doesn't look like a leaf. Please upload a close-up photo of a single tomato leaf.")
    else:
        with st.spinner("Analysing the leaf..."):
            probs, cls, cam = predict_with_cam(img)
        right.image(overlay(img, cam), caption="Where the model looked (Grad-CAM)", width="stretch")
        st.subheader("Top predictions")
        for i in np.argsort(probs)[::-1][:3]:
            st.progress(float(probs[i]), text=f"{pretty(CLASSES[i])}: {probs[i]:.0%}")
        st.divider()
        if probs[cls] < THRESHOLD:
            st.warning(f"Not sure ({probs[cls]:.0%} confidence). The model is not confident enough to give "
                       "treatment advice. Try a clearer photo: one leaf, good daylight, plain background. "
                       "If symptoms continue, consult your local agriculture officer or Krishi Vigyan Kendra.")
        else:
            show_advice(CLASSES[cls], float(probs[cls]))

st.divider()
st.caption(f"Model: {CFG['model_name']} fine-tuned on PlantVillage tomato images (test accuracy 99.7%). "
           "Trained on lab photos; accuracy on field photos may be lower. MSc project – advisory use only.")
