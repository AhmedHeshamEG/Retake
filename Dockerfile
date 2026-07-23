FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-alignment.txt requirements-enhancement.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -r requirements-alignment.txt \
    && python -m pip install --no-cache-dir -r requirements-enhancement.txt

COPY retake.py alignment_worker.py enhancement_worker.py index.html ./
RUN mkdir -p /app/models/llm /app/projects/.incoming

EXPOSE 8710

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8710/status', timeout=3)"

CMD ["python", "retake.py"]
