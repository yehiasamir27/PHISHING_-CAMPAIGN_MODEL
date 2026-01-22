import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# 0 legitimate_email, 1 phishing_email, 2 legitimate_url, 3 phishing_url
IDX2CLASS = {
    0: "legitimate_email",
    1: "phishing_email",
    2: "legitimate_url",
    3: "phishing_url",
}

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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_loose_object(s: str) -> dict:
    s = s.strip().lstrip("\ufeff")

    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    lower = s.lower()

    def find_key(key):
        idx = lower.find(key + ":")
        return idx if idx != -1 else None

    i_sub = find_key("subject")
    i_sen = find_key("sender")
    i_body = find_key("body")

    if i_sub is None and i_sen is None and i_body is None:
        return {"subject": "", "sender": "", "body": s}

    positions = [(i_sub, "subject"), (i_sen, "sender"), (i_body, "body")]
    positions = [(i, k) for i, k in positions if i is not None]
    positions.sort()

    result = {"subject": "", "sender": "", "body": ""}

    for idx, (pos, key) in enumerate(positions):
        start = pos + len(key) + 1
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(s)
        value = s[start:end].strip().strip(",").strip()
        result[key] = value

    return result

def parse_input(arg: str) -> dict:
    if arg is None:
        return {"subject": "", "sender": "", "body": ""}

    arg = arg.strip().lstrip("\ufeff")

    if "{" in arg and "}" in arg:
        start = arg.find("{")
        end = arg.rfind("}")
        candidate = arg[start:end+1].strip()
    else:
        candidate = arg

    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            obj = json.loads(candidate)
            return {
                "subject": str(obj.get("subject", "")),
                "sender": str(obj.get("sender", "")),
                "body": str(obj.get("body", "")),
            }
        except json.JSONDecodeError:
            return parse_loose_object(candidate)

    return {"subject": "", "sender": "", "body": candidate}

def predict_email(subject: str, sender: str, body: str):
    """
    ✅ EMAIL-ONLY classification always.
    ✅ If URL is present, we still output everything + url_detected=true
    (URL analysis is handled in n8n, but we keep this flag for routing.)
    """
    cleaned_body = clean_text(body)
    url_detected = has_url(cleaned_body)

    inputs = tokenizer(cleaned_body, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()

    # Use only email classes (0,1) and renormalize
    p_legit_email, p_phish_email = normalize_pair(probs[0], probs[1])

    predicted_class = "phishing_email" if p_phish_email >= p_legit_email else "legitimate_email"
    confidence = max(p_legit_email, p_phish_email)
    verdict = "PHISHING" if predicted_class == "phishing_email" else "LEGITIMATE"

    shown_probs = {
        "legitimate_email": to_pct(p_legit_email),
        "phishing_email": to_pct(p_phish_email),
    }

    return {
        "subject": subject,
        "sender": sender,
        "cleaned_body": cleaned_body,
        "verdict": verdict,
        "predicted_class": predicted_class,
        "accuracy": to_pct(confidence),
        "all_probabilities": shown_probs,
        "url_detected": url_detected
    }

if __name__ == "__main__":
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input(
        'Paste JSON: {"subject":"...","sender":"...","body":"..."}\n'
    )

    data = parse_input(raw)
    out = predict_email(data["subject"], data["sender"], data["body"])
    print(json.dumps(out, indent=2))
