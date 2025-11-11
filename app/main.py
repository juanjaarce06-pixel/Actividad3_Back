import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, json, numpy as np
import onnxruntime as ort
import requests

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50-v1-7.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_labels.json")

# ✅ URLS válidas (SqueezeNet estaba dando 404)
MODEL_URL = os.getenv("MODEL_URL", "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx")
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url: str, path: str):
    if os.path.exists(path):
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
def ensure_assets():
    download_file(MODEL_URL, MODEL_PATH)
    download_file(LABELS_URL, LABELS_PATH)

# Cache global (se inicializa on-demand)
_SESS = None
_LABELS = None

def get_session_and_labels():
    global _SESS, _LABELS
    if _SESS is None or _LABELS is None:
        ensure_assets()
        _SESS = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        with open(LABELS_PATH, "r") as f:
            _LABELS = json.load(f)  # 1000 labels
    return _SESS, _LABELS

def preprocess(img: Image.Image) -> np.ndarray:
    # ResNet50: 224x224, normalización ImageNet
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, 0)        # NCHW
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def topk(probs: np.ndarray, labels: list[str], k: int = 5):
    idxs = np.argsort(-probs)[:k]
    return [
        {"index": int(i), "label": labels[int(i)], "prob": float(probs[int(i)])}
        for i in idxs
    ]
app = FastAPI(title="Actividad3 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    try:
        sess, labels = get_session_and_labels()
        return {"status": "ok", "loaded": True, "classes": len(labels)}
    except Exception as e:
        return {"status": "error", "loaded": False, "detail": str(e)}

@app.get("/labels")
def labels():
    _, labels = get_session_and_labels()
    return labels

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    sess, labels = get_session_and_labels()

    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    x = preprocess(img)  # (1,3,224,224)

    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})[0].squeeze()  # (1000,)

    probs = softmax(out)
    tk = topk(probs, labels, k=5)
    best = max(tk, key=lambda d: d["prob"])

    return {"topk": tk, "best": best, "count_classes": len(labels)}
