from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.inference.orchestrator import Ensemble
from app.schemas import PredictResponse
from app.services.telemetry import log_prediction, log_error, get_metrics

app = FastAPI(title="Ensemble Image API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ensemble = Ensemble()

@app.get("/")
def index(): return {"message":"OK"}

@app.get("/health")
def health(): return ensemble.health()

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Tipo no soportado")
    try:
        b = await file.read()
        out = ensemble.run(b)
        log_prediction({"request_id": out["request_id"], "summary": out["summary"]})
        return out
    except Exception as e:
        log_error(str(e))
        raise HTTPException(500, f"Inferencia falló: {e}")

@app.get("/metrics")
def metrics(): return get_metrics()
