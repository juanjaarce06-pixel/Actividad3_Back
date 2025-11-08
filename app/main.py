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

def _download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

_download_if_missing(MODEL_URL, MODEL_PATH)
_download_if_missing(LABELS_URL, LABELS_PATH)

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)  # 1000 labels

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std  # HWC
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr

def topk_from_logits(logits: np.ndarray, k: int = 5):
    # logits shape: (1, 1000, 1, 1) ó (1, 1000)
    vec = logits.reshape(1, -1)[0]
    exps = np.exp(vec - np.max(vec))
    probs = exps / exps.sum()
    idxs = probs.argsort()[-k:][::-1]
    out = []
    for i in idxs:
        out.append({
            "index": int(i),
            "label": LABELS[i] if i < len(LABELS) else f"class_{i}",
            "prob": float(probs[i])
        })
    best = out[0]
    return out, best

app = FastAPI(title="Actividad3 Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    return {"count": len(LABELS), "labels": LABELS}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    x = preprocess(img)
    logits = sess.run(None, {input_name: x})[0]
    topk, best = topk_from_logits(logits, k=5)
    return {"topk": topk, "best": best, "count_classes": len(LABELS)}
