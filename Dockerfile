FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Install PyTorch with CUDA (override CPU-only from requirements)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy app code
COPY api/ ./api/
COPY training/model_metric.py ./training/model_metric.py

# Photo storage
RUN mkdir -p /photos

EXPOSE 8000

# Single worker - GPU not thread-safe
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
