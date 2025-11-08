from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.status import HTTP_204_NO_CONTENT

app = FastAPI(title="Actividad 3 - Backend", version="1.0.0")

ALLOWED_ORIGINS = ["*"]
ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
ALLOWED_HEADERS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # si quieres, luego restringes al FRONT_URL
    allow_credentials=False,
    allow_methods=["*"],          # importante para OPTIONS
    allow_headers=["*"],          # importante para OPTIONS con content-type
    expose_headers=[],
    max_age=86400,                # caching del preflight (opcional)
)

@app.get("/health")
def health():
    return {"status": "ok", "loaded": True, "classes": ["cat", "dog", "bird"]}

@app.options("/predict")
def predict_options():
    return Response(status_code=HTTP_204_NO_CONTENT)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    name = (file.filename or "").lower()
    pred = {"cat": 0.33, "dog": 0.33, "bird": 0.33}

    if "cat" in name:
        pred = {"cat": 0.88, "dog": 0.07, "bird": 0.05}
    elif "dog" in name:
        pred = {"cat": 0.06, "dog": 0.90, "bird": 0.04}
    elif "bird" in name:
        pred = {"cat": 0.09, "dog": 0.08, "bird": 0.83}

    return {"prediction": pred, "probability": max(pred.values())}

@app.exception_handler(404)
def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
