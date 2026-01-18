import streamlit as st
import numpy as np
from PIL import Image
from newspaper import Article
from fpdf import FPDF
import re
import warnings
import joblib
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, CLIPProcessor, CLIPModel

def safe_import_easyocr():
    try:
        import easyocr
        return easyocr
    except ImportError:
        return None


warnings.filterwarnings("ignore")

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="TruthShield | Forensic News Analysis",
    page_icon="📰",
    layout="wide"
)

# -------------------------------------------------
# FRONTEND STYLES (UNCHANGED)
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Roboto:wght@300;400;700&display=swap');

.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=2070");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 70px;
    font-weight: 900;
    color: white;
    text-align: center;
    padding-top: 40px;
    border-bottom: 2px solid white;
}

.hero-subtitle {
    font-family: 'Roboto', sans-serif;
    color: #ccc;
    text-align: center;
    font-size: 18px;
    margin-bottom: 40px;
}

.verdict-box {
    text-align: center;
    background: rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 15px;
    margin-top: 20px;
}

.verdict-text {
    font-size: 42px;
    font-weight: 900;
}

.fake-text { color: #ff4b4b; }
.real-text { color: #2ecc71; }

.trigger {
    background: rgba(255,75,75,0.3);
    border-bottom: 2px solid red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
TRIGGERS = [
    "shocking", "urgent", "conspiracy", "exposed",
    "secret", "unbelievable", "miracle", "beware"
]

HIGH_RISK_PHRASES = [
    "died", "die", "death", "suicide", "killed",
    "passed away", "assassinated", "murdered"
]

PUBLIC_FIGURES = [
    "donald trump", "joe biden", "narendra modi",
    "elon musk", "barack obama", "virat kohli"
]

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def high_risk_claim(text):
    t = text.lower()
    return any(p in t for p in HIGH_RISK_PHRASES) and any(
        person in t for person in PUBLIC_FIGURES
    )

def sanitize_for_pdf(text):
    return text.encode("latin-1", errors="ignore").decode("latin-1")

# -------------------------------------------------
# OCR
# -------------------------------------------------
@st.cache_resource
def get_reader():
    easyocr = safe_import_easyocr()
    if easyocr is None:
        return None
    return easyocr.Reader(['en'], gpu=False)


# -------------------------------------------------
# ML MODEL (TF-IDF + LOGISTIC)
# -------------------------------------------------
@st.cache_resource
def load_ml_model():
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    model = LogisticRegression()

    X = [
        "Government announces new education policy",
        "Scientists discover new planet",
        "Shocking secret government exposed",
        "Miracle cure doctors don't want you to know",
        "Official budget released today",
        "Urgent conspiracy revealed"
    ]
    y = [0, 0, 1, 1, 0, 1]

    Xv = vectorizer.fit_transform(X)
    model.fit(Xv, y)

    return model, vectorizer

def ml_predict(text):
    model, vectorizer = load_ml_model()
    vec = vectorizer.transform([text])
    return model.predict_proba(vec)[0][1]

# -------------------------------------------------
# TEXT AI (BERT)
# -------------------------------------------------
@st.cache_resource
def load_bert_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )

def bert_predict(text):
    bert = load_bert_model()
    result = bert(text[:512])[0]
    scores = {r["label"]: r["score"] for r in result}
    return scores.get("NEGATIVE", 0.0)

# -------------------------------------------------
# IMAGE AI (CLIP)
# -------------------------------------------------
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def clip_image_spam_predict(image):
    model, processor = load_clip_model()

    labels = [
        "a legitimate news article image",
        "a real newspaper photo",
        "a spam advertisement poster",
        "a fake news image",
        "a clickbait social media post"
    ]

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    spam_score = probs[2] + probs[3] + probs[4]
    real_score = probs[0] + probs[1]

    return spam_score > real_score, spam_score.item()

# -------------------------------------------------
# ANALYSIS ENGINE
# -------------------------------------------------
def analyze_text_locally(text):
    found = [t for t in TRIGGERS if t in text.lower()]

    if high_risk_claim(text):
        return "FAKE", 0.95, found + ["high-risk factual claim"]

    caps_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
    rule_score = (len(found) * 0.05) + (caps_ratio * 0.2)

    ml_score = ml_predict(text)
    bert_score = bert_predict(text)

    final_score = (bert_score * 0.5) + (ml_score * 0.3) + (rule_score * 0.2)

    if final_score > 0.5:
        return "FAKE", min(0.98, 0.6 + final_score), found
    else:
        return "REAL", min(0.98, 0.7 + (0.5 - final_score)), found

# -------------------------------------------------
# PDF
# -------------------------------------------------
def create_pdf_report(text, verdict, confidence, triggers):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 12, "TRUTHSHIELD FORENSIC REPORT", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Verdict: {verdict}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2%}", ln=True)
    pdf.cell(0, 10, f"Triggers: {', '.join(triggers) if triggers else 'None'}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, sanitize_for_pdf(text))

    return pdf.output(dest="S")

# -------------------------------------------------
# UI
# -------------------------------------------------
st.markdown('<div class="hero-title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Advanced AI-Powered Forensic Analysis Engine</div>', unsafe_allow_html=True)

def perform_analysis(content):
    verdict, conf, found = analyze_text_locally(content)
    style = "fake-text" if verdict == "FAKE" else "real-text"

    st.markdown(f"""
    <div class="verdict-box">
        <p>The News is</p>
        <p class="verdict-text {style}">{verdict}</p>
        <p>Confidence: {conf:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    pdf = create_pdf_report(content, verdict, conf, found)
    st.download_button("📥 Download Report", pdf, "TruthShield_Report.pdf")

# -------------------------------------------------
# TABS
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    tab1, tab2, tab3 = st.tabs(["[ 📄 ] TEXT", "[ 🖼️ ] IMAGE", "[ 🌐 ] URL"])

    with tab1:
        text = st.text_area("Text", height=150)
        if st.button("Verify Text", key="text"):
            perform_analysis(text)

  with tab2:
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Extract & Verify", key="img"):
            is_spam, score = clip_image_spam_predict(img)

            if is_spam:
                st.markdown(f"""
                <div class="verdict-box">
                    <p>The Image News is</p>
                    <p class="verdict-text fake-text">FAKE / SPAM</p>
                    <p>AI Image Spam Score: {score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                reader = get_reader()

                if reader is None:
                    st.warning(
                        "OCR is not available on cloud. "
                        "Image AI analysis was completed successfully."
                    )
                else:
                    result = reader.readtext(
                        np.array(img),
                        detail=0,
                        paragraph=True
                    )

                    extracted = " ".join(result)

                    if extracted.strip():
                        perform_analysis(extracted)
                    else:
                        st.error("No readable text found.")


    with tab3:
        url = st.text_input("Article URL")
        if st.button("Check URL", key="url"):
            try:
                article = Article(url, browser_user_agent="Mozilla/5.0")
                article.download()
                article.parse()
                perform_analysis(article.text)
            except:
                st.error("Failed to fetch article.")


