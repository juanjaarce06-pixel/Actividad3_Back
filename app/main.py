# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, json, numpy as np
import onnxruntime as ort
import requests

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "squeezenet1.1.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_labels.json")
MODEL_URL = "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

os.makedirs(MODEL_DIR, exist_ok=True)

def _download_if_missing(url: str, path: str, is_json: bool = False):
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

_download_if_missing(MODEL_URL, MODEL_PATH)
_download_if_missing(LABELS_URL, LABELS_PATH)

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)  # 1000 labels, e.g. "tench", "goldfish", ...

# ONNXRuntime session
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name

def preprocess(img: Image.Image) -> np.ndarray:
    # SqueezeNet expects 224x224 RGB, normalized to ImageNet stats
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

app = FastAPI(title="Actividad3 Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok", "loaded": True, "classes": len(LABELS)}

@app.get("/labels")
def labels():
    return {"count": len(LABELS), "labels": LABELS}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    x = preprocess(img)
    logits = sess.run(None, {INPUT_NAME: x})[0]
    probs = softmax(logits)
    probs = probs[0]  # shape (1000,)

    topk_idx = np.argsort(probs)[-5:][::-1]
    topk = [
        {"index": int(i), "label": LABELS[i], "prob": float(probs[i])}
        for i in topk_idx
    ]
    best = topk[0]
    return {"topk": topk, "best": best, "count_classes": len(LABELS)}
