FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /lib/apt/lists/*

# Add the rest of your build instructions below (WORKDIR, COPY, RUN pip install, CMD, etc.)