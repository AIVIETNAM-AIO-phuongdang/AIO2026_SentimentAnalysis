"""
main.py — Vietnamese Sentiment FastAPI Server
=============================================
Cách chạy:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from scipy.sparse import hstack
import joblib

# ── Load model ────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.getenv("MODEL_PATH",
             os.path.join(BASE, "models", "sentiment_model.joblib"))

try:
    _obj       = joblib.load(MODEL_PATH)
    MODEL      = _obj["model"]
    TFIDF_WORD = _obj["tfidf_word"]
    TFIDF_CHAR = _obj["tfidf_char"]
    USE_CHAR   = _obj.get("use_char", True)
    MODEL_NAME = _obj.get("model_name", "unknown")
    ACCURACY   = round(_obj.get("accuracy", 0), 4)
    F1_MACRO   = round(_obj.get("f1_macro", 0), 4)
    print(f"✅ Model loaded: {MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"❌ Cannot load model: {e}\n"
                       f"   → Run: python notebook/train.py")

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="🛍️ Vietnamese Sentiment API",
    description="Phân tích cảm xúc đánh giá sản phẩm tiếng Việt",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

EMOJI = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😞"}

# ── Schemas ───────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      example="chất lượng tuyệt vời giao hàng rất nhanh")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100,
                             example=["hàng đẹp lắm", "giao sai màu rồi"])

class PredictResult(BaseModel):
    text: str
    sentiment: str
    label: str
    confidence: Optional[float] = None

class BatchResult(BaseModel):
    count: int
    results: List[PredictResult]
    latency_ms: float

# ── Inference helper ──────────────────────────────────────────────
def _run(texts: List[str]):
    Xw = TFIDF_WORD.transform(texts)
    X  = hstack([Xw, TFIDF_CHAR.transform(texts)]) if USE_CHAR else Xw
    labels = MODEL.predict(X)
    confs  = (MODEL.predict_proba(X).max(axis=1).round(4).tolist()
              if hasattr(MODEL, "predict_proba") else [None]*len(texts))
    return labels, confs

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {"status": "✅ running", "model": MODEL_NAME,
            "accuracy": ACCURACY, "f1_macro": F1_MACRO,
            "docs": "/docs"}

@app.get("/health", tags=["Info"])
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResult, tags=["Inference"])
def predict(body: SingleRequest):
    """Phân tích cảm xúc **1 câu**."""
    try:
        labels, confs = _run([body.text])
        lbl = str(labels[0])
        return PredictResult(text=body.text,
                             sentiment=f"{EMOJI[lbl]} {lbl}",
                             label=lbl,
                             confidence=confs[0])
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult, tags=["Inference"])
def predict_batch(body: BatchRequest):
    """Phân tích cảm xúc **nhiều câu** (tối đa 100)."""
    try:
        t0 = time.perf_counter()
        labels, confs = _run(body.texts)
        ms = round((time.perf_counter()-t0)*1000, 2)
        results = [
            PredictResult(text=t,
                          sentiment=f"{EMOJI[str(l)]} {l}",
                          label=str(l),
                          confidence=c)
            for t, l, c in zip(body.texts, labels, confs)
        ]
        return BatchResult(count=len(results), results=results, latency_ms=ms)
    except Exception as e:
        raise HTTPException(500, str(e))
