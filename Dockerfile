FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app1.py .
COPY .streamlit .streamlit/

RUN mkdir -p /data

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app1.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
