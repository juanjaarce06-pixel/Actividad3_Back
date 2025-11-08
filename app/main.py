# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Actividad 3 - Backend", version="1.0.0")

# ---- CORS ----
# Si quieres restringir, sustituye "*" por:
# ["https://actividad3-frontend-537174375411.us-central1.run.app"]
ALLOW_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Utilidad para poner siempre los headers CORS
def _cors_headers(resp: Response):
    resp.headers["Access-Control-Allow-Origin"] = (
        ALLOW_ORIGINS[0] if ALLOW_ORIGINS != ["*"] else "*"
    )
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

@app.get("/health")
def health():
    return {"status": "ok", "loaded": True, "classes": ["cat", "dog", "bird"]}

# ---------- Handler dual para /predict (OPTIONS + POST) ----------
@app.api_route("/predict", methods=["OPTIONS", "POST"])
async def predict(request: Request, file: UploadFile | None = File(None)):
    # Preflight
    if request.method == "OPTIONS":
        return _cors_headers(Response(status_code=204))

    # POST real
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    name = (file.filename or "").lower()
    if "cat" in name:
        pred, prob = "cat", 0.98
    elif "dog" in name:
        pred, prob = "dog", 0.97
    elif "bird" in name:
        pred, prob = "bird", 0.95
    else:
        pred, prob = "unknown", 0.51

    resp = JSONResponse({"prediction": pred, "probability": prob})
    return _cors_headers(resp)

# Opcional: OPTIONS comodín por si algo más hace falta
@app.options("/{path:path}", include_in_schema=False)
def options_any(path: str):
    return _cors_headers(Response(status_code=204))
