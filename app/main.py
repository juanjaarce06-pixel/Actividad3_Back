from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import random

app = FastAPI(title="Actividad 3 - Backend", version="1.0.0")

FRONT_ORIGIN = "https://actividad3-frontend-537174375411.us-central1.run.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_ORIGIN, "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": FRONT_ORIGIN,
        "Vary": "Origin",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Max-Age": "86400",
    }

@app.get("/health")
def health():
    return {"status": "ok", "loaded": True, "classes": ["cat","dog","bird"]}

@app.options("/predict")
async def options_predict():
    return JSONResponse({"ok": True}, headers=_cors_headers())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    b = await file.read()
    await file.close()
    # dummy model
    classes = ["cat","dog","bird"]
    idx = (len(b) or 1) % 3
    pred = classes[idx]
    probs = [0.8, 0.15, 0.05]
    prob = probs[idx]
    return JSONResponse({"prediction": pred, "probability": prob}, headers=_cors_headers())
