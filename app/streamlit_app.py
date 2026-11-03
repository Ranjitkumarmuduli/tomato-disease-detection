"""Tomato Leaf Doctor v2 – TomatoLeafNet (our own CNN) + multilingual RAG advisor (Gemma SLM / Gemini LLM)."""
import json, os
import numpy as np
import streamlit as st
import torch, torch.nn.functional as F
from PIL import Image

from tomatoleafnet import TomatoLeafNet
from advisor import (LANGUAGES, RETAKE, SOURCE, SYSTEM, KnowledgeBase, LanguageModel,
                     build_prompt, resolve_language, scrub_doses)

st.set_page_config(page_title="Tomato Leaf Doctor", page_icon="🍅", layout="centered")
torch.set_num_threads(2)                       # Streamlit Cloud has very few CPU cores


def log(msg):                                  # appears in "Manage app" logs
    print(f"[TLD] {msg}", flush=True)


log("script started")
st.title("🍅 Tomato Leaf Doctor")
# find app_model_v2 next to this file, one folder up, or in the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = next((p for p in [os.path.join(_HERE, "app_model_v2"), os.path.join(_HERE, "..", "app_model_v2"),
                              os.path.join(os.getcwd(), "app_model_v2")]
                  if os.path.exists(os.path.join(p, "config.json"))), None)
if MODEL_DIR is None:
    st.error("Model files not found. Put model.pt, config.json and disease_info.json in a folder named "
             "app_model_v2 next to streamlit_app.py.")
    st.stop()


# ---------------- loading (cached: runs once per server) ----------------
@st.cache_resource
def load_vision():
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    model = TomatoLeafNet(num_classes=len(cfg["classes"]), multiscale=cfg.get("multiscale", True), se=cfg.get("se", True))
    sd = torch.load(f"{MODEL_DIR}/model.pt", map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval(), cfg


@st.cache_resource
def load_kb():
    kb = json.load(open(f"{MODEL_DIR}/disease_info.json", encoding="utf-8"))
    return kb, KnowledgeBase(kb)


def _secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:                      # no secrets file at all
        return ""


@st.cache_resource(show_spinner=False)
def load_llm():
    """Connect to Google only when the advisor is first used (keeps page start fast)."""
    key = _secret_key()
    if not key:
        return None
    try:
        llm = LanguageModel(key)
        log(f"LLM ready: SLM={llm.slm[:2]} LLM={llm.llm[:2]}")
        return llm
    except Exception as e:
        log(f"LLM connect failed: {e}")
        return None


with st.spinner("Loading TomatoLeafNet…"):
    model, cfg = load_vision()
    KB, KBASE = load_kb()
log("vision model + knowledge base loaded")
HAS_KEY = bool(_secret_key())
THRESHOLD = cfg.get("confidence_threshold", 0.6)


def kb_key(class_name):
    """'Tomato___Late_blight' -> 'Late_blight' (key used in disease_info.json)."""
    k = class_name.replace("Tomato___", "")
    if k in KB["diseases"]:
        return k
    norm = lambda s: s.lower().replace("_", "").replace(" ", "")
    return next((d for d in KB["diseases"] if norm(d) == norm(k)), k)


# ---------------- image helpers ----------------
def preprocess(img):
    x = np.asarray(img.convert("RGB").resize((cfg["img_size"], cfg["img_size"])), dtype=np.float32) / 255.0
    x = (x - np.array(cfg["mean"])) / np.array(cfg["std"])
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)


def plant_fraction(img):
    """Share of pixels that look like plant tissue (green / yellow / brown). Used to reject non-leaf photos."""
    hsv = np.asarray(img.convert("RGB").resize((128, 128)).convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return float(((h >= 10) & (h <= 160) & (s > 0.18) & (v > 0.12)).mean())


def predict_with_cam(x):
    """Prediction + Grad-CAM heat-map (where the model looked)."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    probs = torch.softmax(logits, 1)[0].detach().numpy()
    cls = int(probs.argmax())
    model.zero_grad(); logits[0, cls].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probs, cam


def overlay(img, cam):
    base = np.asarray(img.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255
    c = cam[..., None]
    heat = np.concatenate([np.clip(1.5 - abs(4 * c - 3), 0, 1), np.clip(1.5 - abs(4 * c - 2), 0, 1),
                           np.clip(1.5 - abs(4 * c - 1), 0, 1)], axis=2)          # jet colours
    return Image.fromarray((np.clip(0.55 * base + 0.45 * heat, 0, 1) * 255).astype(np.uint8))


def fixed_message(lang, mode):
    if lang in RETAKE:
        return RETAKE[lang]
    LLM = load_llm() if HAS_KEY else None
    if LLM:
        try:
            return LLM.translate(RETAKE["English"], lang, mode)
        except Exception:
            pass
    return RETAKE["English"]


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    lang_choice = st.selectbox("Answer language", LANGUAGES, index=0,
                               help="Auto = reply in the language you type your question in.")
    mode = st.radio("Advisor model", ["Auto", "SLM (Gemma)", "LLM (Gemini)"],
                    help="SLM = small open model (Gemma). LLM = large cloud model (Gemini). Auto uses the SLM first.")
    show_cam = st.toggle("Show where the model looked (Grad-CAM)", value=True)
    st.divider()
    st.caption(f"Vision model: **TomatoLeafNet** (our own CNN, "
               f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M params)")
    st.caption("Advisor: RAG over TNAU / ICAR-IIHR / UC IPM facts" + ("" if HAS_KEY else " — ⚠️ no API key, advisor offline"))
    st.caption("M.Sc. Data Science project, VIT Vellore – Ranjit Kumar Muduli")

# ---------------- main page ----------------
st.write("Take or upload a photo of **one tomato leaf**. The app finds the disease and our advisor tells you what to do "
         "— ask in **your own language**.")

tab_up, tab_cam = st.tabs(["📁 Upload photo", "📷 Use camera"])
with tab_up:
    up = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with tab_cam:
    cam_photo = st.camera_input("Leaf photo", label_visibility="collapsed")
file = up or cam_photo

if file is None:
    st.info("Tip: one leaf, daylight, plain background, leaf fills most of the photo.")
    st.stop()

img = Image.open(file).convert("RGB")
image_id = f"{getattr(file, 'name', 'camera')}-{getattr(file, 'size', 0)}"
if st.session_state.get("image_id") != image_id:          # new photo -> new chat
    st.session_state.image_id, st.session_state.chat, st.session_state.chat_meta = image_id, [], []

lang_for_msgs = resolve_language(lang_choice)

# 1. is it a leaf?
if plant_fraction(img) < 0.05:
    st.image(img, width="stretch")
    st.error("This does not look like a plant leaf. " + fixed_message(lang_for_msgs, mode))
    st.stop()

# 2. diagnose
log("predicting")
probs, cam = predict_with_cam(preprocess(img))
order = probs.argsort()[::-1]
cls_name, conf = cfg["classes"][order[0]], float(probs[order[0]])
disease = kb_key(cls_name)
info = KB["diseases"].get(disease, {})

c1, c2 = st.columns(2)
c1.image(img, caption="Your photo", width="stretch")
if show_cam:
    c2.image(overlay(img, cam), caption="Where the model looked", width="stretch")

if conf < THRESHOLD:
    st.warning(f"Not sure ({conf:.0%}). " + fixed_message(lang_for_msgs, mode))
    st.stop()

healthy = "healthy" in disease.lower()
(st.success if healthy else st.error)(f"**{info.get('display_name', disease)}** — confidence {conf:.0%}")
with st.expander("Other possibilities"):
    for i in order[1:3]:
        st.write(f"{KB['diseases'].get(kb_key(cfg['classes'][i]), {}).get('display_name', cfg['classes'][i])}: {probs[i]:.1%}")

# 3. disease card (always available, even offline)
with st.expander("📋 Disease facts (knowledge base)", expanded=not HAS_KEY):
    if not healthy:
        st.write(f"**Type:** {info.get('type', '-')} · **Cause:** {info.get('pathogen', '-')} · **Urgency:** {info.get('urgency', '-')}")
    for field, label in [("symptoms", "Symptoms"), ("organic_control", "Organic / cultural control"),
                         ("chemical_control", "Chemical control (active ingredients)"), ("prevention", "Prevention")]:
        v = info.get(field)
        if v:
            st.markdown(f"**{label}**\n" + "\n".join(f"- {x}" for x in (v if isinstance(v, list) else [v])))
    st.caption(KB.get("disclaimer", SOURCE))

# 4. chat advisor
st.subheader("💬 Ask the advisor")
if not HAS_KEY:
    st.info("The AI advisor needs an API key (add GEMINI_API_KEY in the app secrets). The facts above are still valid.")
    st.stop()


def answer(question):
    lang = resolve_language(lang_choice, question)
    lang_line = lang if not lang_choice.startswith("Auto") or lang != "English" else "same as the FARMER QUESTION"
    prompt = build_prompt(KBASE, disease, conf, question, lang_line, st.session_state.chat)
    LLM = load_llm()
    if LLM is None:
        return "The advisor could not connect right now. Please use the disease facts above.", "unavailable"
    try:
        text, used, sec = LLM.generate(SYSTEM, prompt, mode)
        return scrub_doses(text) + f"\n\n*{SOURCE}*", f"{used} · {sec:.1f}s"
    except Exception:
        return ("The advisor is busy right now (free limit reached). Please use the disease facts above and try again later.",
                "unavailable")


for role, text, meta in st.session_state.chat_meta:          # show the conversation so far
    with st.chat_message(role, avatar="🧑‍🌾" if role == "user" else "🍅"):
        st.markdown(text)
        if meta:
            st.caption(meta)

quick = st.button("What should I do now?") if not st.session_state.chat_meta else False
question = st.chat_input("Type your question in any language…")
question = question or ("What should I do now?" if quick else None)

if question:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🍅"):
        with st.spinner("Thinking…"):
            text, meta = answer(question)
        st.markdown(text)
        st.caption(meta)
    st.session_state.chat += [("farmer", question), ("advisor", text)]
    st.session_state.chat_meta += [("user", question, ""), ("assistant", text, meta)]
