FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git csm_repo && \
    cd csm_repo && pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/app/csm_repo:$PYTHONPATH
ENV MODEL_REPO=BiggestLab/csm-1b
ENV DEFAULT_SPEAKER_ID=0
ENV MAX_LENGTH_MS=10000

# Copy handler
COPY src/ ./src/

CMD ["python", "-u", "/app/src/handler.py"]
