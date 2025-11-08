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

def _download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)

# Descargas livianas en primer arranque
_download_if_missing(MODEL_URL, MODEL_PATH)
_download_if_missing(LABELS_URL, LABELS_PATH)

# Carga modelo y etiquetas
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

app = FastAPI(title="Actividad3 Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "OK"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    return {"classes": LABELS}

def preprocess(img: Image.Image) -> np.ndarray:
    # SqueezeNet: 224x224, RGB, [0,1], normalización ImageNet
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))              # CHW
    arr = np.expand_dims(arr, 0)                    # NCHW
    return arr

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Requiere python-multipart instalado (ya lo ponemos en requirements)
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    x = preprocess(img)
    inp_name = sess.get_inputs()[0].name
    logits = sess.run(None, {inp_name: x})[0]       # [1,1000]
    probs = softmax(logits)                         # [1,1000]
    probs = probs[0]

    topk = 5
    idxs = np.argsort(-probs)[:topk].tolist()
    results = [
        {"label": LABELS[i], "index": i, "prob": float(round(probs[i], 6))}
        for i in idxs
    ]
    return {
        "topk": results,
        "best": results[0],
        "count_classes": len(LABELS)
    }
