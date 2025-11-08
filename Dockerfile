# ===== build stage (opcional, si compilas algo nativo) =====
FROM python:3.11-slim AS base

# Evita buffering y fuerza logs inmediatos
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Si tienes requirements.txt, primero solo requirements para cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto del proyecto
COPY . /app

# *Clave*: escuchar en 0.0.0.0 y en el puerto que Cloud Run inyecta (PORT)
# Ajusta "main:app" si tu archivo/instancia difiere (por ejemplo "app:app" o "src.api:app")
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
