# Public build recipe. Code, models and scoring files live in a private repo and are
# downloaded at container start with the Space secret HF_TOKEN, so this image holds none of them.
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080 EMBEDDING_LAYER=-1 OMP_NUM_THREADS=2 \
    CHECKPOINT_PATH=/opt/models/checkpoint.pt APP_DIR=/tmp/app \
    HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/.cache
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0
COPY requirements-deploy.txt /opt/requirements-deploy.txt
RUN pip install --no-cache-dir -r /opt/requirements-deploy.txt
# Similarity model for the Evaluate page (already public on Hugging Face)
RUN mkdir -p /opt/models && curl -fL --retry 3 -o /opt/models/checkpoint.pt \
    "https://huggingface.co/tianyi0216/hmong-xlsr-checkpoint/resolve/main/checkpoint_slim.pt"
COPY start.sh /opt/start.sh
EXPOSE 8080
CMD ["bash", "/opt/start.sh"]
