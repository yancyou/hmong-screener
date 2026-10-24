# Single-container deploy: FastAPI serves the built web app (/) + the
# similarity API (/api). CPU-only. Pins match the locally verified env.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    CHECKPOINT_PATH=/app/checkpoint.pt \
    EMBEDDING_LAYER=-1 \
    OMP_NUM_THREADS=4

# System libs: libsndfile for soundfile, build tools + git for fairseq editable
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU wheel first (big, cached layer)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0

# Python deps (pinned)
COPY requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

# Vendored fairseq (editable install with custom tasks)
COPY fairseq ./fairseq
RUN pip install --no-build-isolation -e ./fairseq

# Application code + assets
COPY api_similarity_server.py freeze_layers_hook.py template_manifest_all_sources_with_xlsx.csv ./
COPY GUI/dist ./GUI/dist
# Reference word library (enables the /api/words dropdown); ~85 MB
COPY all_data ./all_data
# Slim inference checkpoint (~1.3 GB), downloaded at build time from the public
# model repo so no large file is needed in the build context. To use a different
# checkpoint:  docker build --build-arg CHECKPOINT_REPO=<user>/<repo> .
ARG CHECKPOINT_REPO=tianyi0216/hmong-xlsr-checkpoint
ARG CHECKPOINT_FILE=checkpoint_slim.pt
RUN curl -fL --retry 3 -o /app/checkpoint.pt \
      "https://huggingface.co/${CHECKPOINT_REPO}/resolve/main/${CHECKPOINT_FILE}" \
    && ls -lh /app/checkpoint.pt

# Pre-create the fallback letter dictionary so nothing has to be written at
# runtime (the container may run without write access to /app).
RUN mkdir -p /app/_fallback_data \
    && python -c "import pathlib; t=list('abcdefghijklmnopqrstuvwxyz')+[chr(39),'|']; \
pathlib.Path('/app/_fallback_data/dict.ltr.txt').write_text('\n'.join(f'{c} 1' for c in t)+'\n')"

EXPOSE 8080
CMD ["python", "api_similarity_server.py"]
