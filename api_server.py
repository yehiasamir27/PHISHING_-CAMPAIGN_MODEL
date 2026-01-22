import os
import json
import re
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- CONFIG ----------
MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

# Put your key here OR set env var PHISH_API_KEY
API_KEY = os.getenv("PHISH_API_KEY", "MY_SUPER_SECRET_KEY_123456")

# ---------- MODEL ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

URL_REGEX = re.compile(r"(https?://|www\.)", re.IGNORECASE)

def has_url(text: str) -> bool:
    return bool(URL_REGEX.search(text))

def to_pct(x: float) -> str:
    return f"{x * 100:.2f}%"

def normalize_pair(a: float, b: float):
    s = a + b
    if s == 0:
        return 0.5, 0.5
    return a / s, b / s

def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def email_only_predict(subject: str, sender: str, body: str):
    cleaned_body = clean_text(body)
    url_detected = has_url(cleaned_body)

    inputs = tokenizer(cleaned_body, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()

    # Email-only: use classes 0 and 1 and renormalize
    p_legit_email, p_phish_email = normalize_pair(probs[0], probs[1])

    predicted_class = "phishing_email" if p_phish_email >= p_legit_email else "legitimate_email"
    confidence = max(p_legit_email, p_phish_email)
    verdict = "PHISHING" if predicted_class == "phishing_email" else "LEGITIMATE"

    return {
        "subject": subject or "",
        "sender": sender or "",
        "cleaned_body": cleaned_body,
        "verdict": verdict,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),          # numeric (best for n8n IF)
        "accuracy": to_pct(confidence),              # pretty string
        "all_probabilities": {
            "legitimate_email": to_pct(p_legit_email),
            "phishing_email": to_pct(p_phish_email),
        },
        "url_detected": url_detected
    }

# ---------- FASTAPI ----------
app = FastAPI(title="Phishing Email Classifier API")

class EmailReq(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: EmailReq, x_api_key: str = Header(default="")):
    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return email_only_predict(payload.subject, payload.sender, payload.body)
