# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Actividad 3 - Backend", version="1.0.0")

# === CORS ===
# Para destrabar, usa "*" (luego puedes restringir a tu front).
# Si quieres restringir ya, usa:
# allow_origins=["https://actividad3-frontend-537174375411.us-central1.run.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                 # <-- o la URL exacta del front
    allow_credentials=False,
    allow_methods=["*"],                 # IMPORTANTE: "*" incluye OPTIONS
    allow_headers=["*"],                 # al menos "Content-Type" y "Authorization"
    expose_headers=["*"],
    max_age=86400,
)

# === Handlers explícitos para preflight ===
# 1) Sólo para /predict
@app.options("/predict", include_in_schema=False)
def options_predict():
    # Aunque CORSMiddleware normalmente inyecta los headers,
    # los agregamos explícitos por si el middleware no intercepta antes:
    resp = Response(status_code=204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

# 2) Comodín para cualquier ruta (por si el front pega a otra ruta en el futuro)
@app.options("/{path:path}", include_in_schema=False)
def options_any(path: str):
    resp = Response(status_code=204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

@app.get("/health")
def health():
    return {"status": "ok", "loaded": True, "classes": ["cat", "dog", "bird"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
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
    return {"prediction": pred, "probability": prob}
