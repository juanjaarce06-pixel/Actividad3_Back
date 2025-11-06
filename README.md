# Actividad3 - Backend (FastAPI)
API de inferencia con orquestador de modelos. Listo para Cloud Run.

## Local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
