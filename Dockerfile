FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# Pillow/reportlab runtime libs. libgomp1 only needed if OCR (paddlepaddle)
# is enabled again in the future — kept for convenience.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Core app deps only. OCR is intentionally NOT installed here to keep the
# image small and the Render free-tier build within time/memory limits.
# To enable OCR: uncomment requirements-ocr.txt below (~2GB, may time out).
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements-ocr.txt

COPY requirements-ocr.txt .
COPY app1.py .
COPY .streamlit .streamlit/

# products.db lives next to the script by default; set DB_FILE=/data/products.db
# (and mount a Render disk at /data) to persist the catalog across deploys.
RUN mkdir -p /data

EXPOSE 8501

CMD ["streamlit", "run", "app1.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
