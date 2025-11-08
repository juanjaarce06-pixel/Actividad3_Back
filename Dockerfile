# Dockerfile (backend)
FROM python:3.11-slim

# Evita bytecode y buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# Instala dependencias del sistema si hiciera falta (opcional)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copia tus requirements y los instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la app (asegúrate que tu main está en app/main.py)
COPY app/ /app/app

# Expone el puerto del contenedor
EXPOSE 8080
ENV PORT=8080
# <--- ESTA es la línea CMD que viste --->
# Arranca FastAPI con Uvicorn (NO se ejecuta en la terminal; va dentro del Dockerfile)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
