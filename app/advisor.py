"""RAG farm advisor: retrieves facts for the predicted disease and asks a language model (Gemma SLM / Gemini LLM)."""
import re, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- languages ----------------
LANGUAGES = ["Auto (same as my question)", "English", "Hindi", "Marathi", "Bengali", "Assamese", "Odia", "Tamil",
             "Telugu", "Kannada", "Malayalam", "Gujarati", "Punjabi", "Urdu", "Nepali"]

SCRIPTS = {"Hindi": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Punjabi": (0x0A00, 0x0A7F),
           "Gujarati": (0x0A80, 0x0AFF), "Odia": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
           "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
           "Urdu": (0x0600, 0x06FF)}


def detect_language(text):
    """Guess the language from the script of the letters. Latin letters -> None (could be English or Hinglish)."""
    letters = [ch for ch in text if ch.isalpha()]
    for lang, (lo, hi) in SCRIPTS.items():
        if letters and sum(lo <= ord(ch) <= hi for ch in letters) / len(letters) > 0.5:
            return lang
    return None


def resolve_language(choice, question=""):
    if not choice.startswith("Auto"):
        return choice
    return detect_language(question) or "English"


# fixed messages (no model call) – other languages are translated by the model on demand
RETAKE = {
 "English": "The photo is not clear enough for a reliable diagnosis. Please take a new photo of one leaf in daylight, on a plain background. If the problem continues, contact your local Krishi Vigyan Kendra.",
 "Hindi":   "फोटो साफ़ नहीं है, इसलिए सही पहचान नहीं हो पाई। कृपया दिन की रोशनी में, सादे पृष्ठभूमि पर एक पत्ते की नई फोटो लें। समस्या बनी रहे तो अपने नज़दीकी कृषि विज्ञान केंद्र से संपर्क करें।",
 "Odia":    "ଫଟୋ ସ୍ପଷ୍ଟ ନୁହେଁ, ତେଣୁ ଠିକ୍ ଚିହ୍ନଟ ହୋଇପାରିଲା ନାହିଁ। ଦୟାକରି ଦିନ ଆଲୁଅରେ, ସାଦା ପୃଷ୍ଠଭୂମିରେ ଗୋଟିଏ ପତ୍ରର ନୂଆ ଫଟୋ ନିଅନ୍ତୁ। ସମସ୍ୟା ରହିଲେ ନିକଟସ୍ଥ କୃଷି ବିଜ୍ଞାନ କେନ୍ଦ୍ର ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ।",
 "Tamil":   "புகைப்படம் தெளிவாக இல்லை, அதனால் சரியாகக் கண்டறிய முடியவில்லை. பகல் வெளிச்சத்தில், வெற்றுப் பின்னணியில் ஒரு இலையின் புதிய புகைப்படம் எடுக்கவும். பிரச்சனை தொடர்ந்தால் அருகிலுள்ள கிருஷி விஞ்ஞான கேந்திராவைத் தொடர்பு கொள்ளவும்.",
 "Telugu":  "ఫోటో స్పష్టంగా లేదు, కాబట్టి సరిగ్గా గుర్తించలేకపోయాము. దయచేసి పగటి వెలుతురులో, సాదా నేపథ్యంపై ఒక ఆకు కొత్త ఫోటో తీయండి. సమస్య కొనసాగితే మీ దగ్గరలోని కృషి విజ్ఞాన కేంద్రాన్ని సంప్రదించండి.",
 "Bengali": "ছবিটি পরিষ্কার নয়, তাই সঠিকভাবে রোগ চেনা যায়নি। দয়া করে দিনের আলোয়, সাদামাটা পটভূমিতে একটি পাতার নতুন ছবি তুলুন। সমস্যা থাকলে নিকটবর্তী কৃষি বিজ্ঞান কেন্দ্রে যোগাযোগ করুন।"}

SOURCE = "Source: TNAU / ICAR-IIHR / UC IPM. Follow the product label and confirm with your local agriculture officer."

SYSTEM = """You are the farm advisor of the Tomato Leaf Doctor app. Rules:
1. Use ONLY the facts in CONTEXT. If the farmer asks about anything not written in CONTEXT (a product, remedy or method),
   say it is not in your information and suggest the local Krishi Vigyan Kendra. Do not guess.
2. Never give doses, amounts, mixing rates or spray schedules. Name active ingredients only and say: follow the product label.
3. Use simple words a farmer understands. Short sentences. Bullet points for steps (at most 8).
4. Reply ONLY in the language named in LANGUAGE (if it says "same as the question", use the farmer's language and script,
   e.g. Hinglish -> Hinglish). Keep chemical names in English in brackets so the farmer can show them at the shop.
5. For virus diseases say clearly that there is no cure; advise removing plants and controlling the insect that spreads it."""

FIELDS = {"symptoms": "Symptoms", "spread": "How it spreads", "organic_control": "Organic and cultural control",
          "chemical_control": "Chemical control (active ingredients only)", "prevention": "Prevention"}


class KnowledgeBase:
    """Splits disease_info.json into chunks and retrieves the ones for the predicted disease (TF-IDF ranking)."""
    def __init__(self, kb):
        self.kb, self.chunks = kb, []
        for key, d in kb["diseases"].items():
            name = d["display_name"]
            self.chunks.append(dict(disease=key, field="overview",
                                    text=f"{name}: type {d['type']}; cause {d['pathogen']}; urgency {d['urgency']}."))
            for f, label in FIELDS.items():
                v = d.get(f, "")
                text = "; ".join(v) if isinstance(v, list) else v
                if text and text != "-":
                    self.chunks.append(dict(disease=key, field=f, text=f"{name} – {label}: {text}"))
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, disease, question, k=5):
        idx = [i for i, c in enumerate(self.chunks) if c["disease"] == disease]
        sims = cosine_similarity(self.vec.transform([question]), self.mat[idx])[0]
        order = [idx[j] for j in np.argsort(sims)[::-1]]
        overview = [i for i in idx if self.chunks[i]["field"] == "overview"]
        return [self.chunks[i] for i in overview + [i for i in order if i not in overview][:k]]


def build_prompt(kbase, disease, confidence, question, language, history=()):
    name = kbase.kb["diseases"][disease]["display_name"]
    context = "\n".join(f"- {c['text']}" for c in kbase.retrieve(disease, question))
    past = "\n".join(f"{r.upper()}: {t[:400]}" for r, t in list(history)[-4:])
    return (f"DIAGNOSIS: {name} (model confidence {confidence:.0%})\nLANGUAGE: {language}\n"
            f"CONTEXT:\n{context}\n"
            + (f"\nPREVIOUS CONVERSATION:\n{past}\n" if past else "")
            + f"\nFARMER QUESTION: {question}\n\n"
            + (f"Answer in {language} only, using only the CONTEXT." if not language.startswith("same")
               else "Answer in the same language and script as the FARMER QUESTION, using only the CONTEXT."))


class LanguageModel:
    """Talks to Google's API. Tries the small Gemma model (SLM) or Gemini (LLM) and falls back automatically
    when a model is busy or its free daily limit is used up."""
    GEMMA_SIZES = ["4b", "e4b", "12b", "e2b", "1b"]
    GEMINI = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.7-flash", "gemini-3.8-flash"]

    def __init__(self, api_key):
        from google import genai
        self.genai = genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))  # 30 s max
        names = [m.name.replace("models/", "") for m in self.client.models.list()]
        gemmas = [n for n in names if "gemma" in n.lower() and n.endswith("-it")]
        self.slm = [g for s in self.GEMMA_SIZES for g in gemmas if f"-{s}-" in g]
        self.llm = [g for g in self.GEMINI if g in names]

    def candidates(self, mode):
        if mode == "SLM (Gemma)":
            return self.slm + self.llm          # SLM first, LLM only as backup
        if mode == "LLM (Gemini)":
            return self.llm + self.slm
        return self.slm + self.llm              # Auto: small model first (cheaper, larger free quota)

    def generate(self, system, prompt, mode="Auto"):
        from google.genai import types
        last_err = "no model available"
        for model in dict.fromkeys(self.candidates(mode)):
            for attempt in range(2):
                try:
                    t0 = time.time()
                    if "gemma" in model:            # Gemma on this API: rules go inside the prompt
                        r = self.client.models.generate_content(
                            model=model, contents=system + "\n\n" + prompt,
                            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048))
                    else:
                        r = self.client.models.generate_content(
                            model=model, contents=prompt,
                            config=types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                                               max_output_tokens=4096))
                    text = (r.text or "").strip()
                    if text:
                        return text, model, time.time() - t0
                    break
                except Exception as e:                # busy / limit / not available -> next try or next model
                    last_err = str(e)[:120]
                    if "503" in last_err and attempt == 0:
                        time.sleep(2); continue
                    break
        raise RuntimeError(last_err)

    def translate(self, text, language, mode="Auto"):
        out, _, _ = self.generate("You are a translator. Translate faithfully, simple words.",
                                  f"Translate into {language}. Return only the translation.\n\n{text}", mode)
        return out


def scrub_doses(text):
    """Safety net: remove any dose the model still wrote (e.g. '2 g per litre')."""
    return re.sub(r"\b\d+(\.\d+)?\s*(ml|g|gm|grams?|kg|litres?|liters?|ppm)\b(\s*(per|/)\s*\w+)?",
                  "[see product label]", text, flags=re.I)
