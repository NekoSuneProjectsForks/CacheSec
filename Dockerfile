FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/data \
    HOST=0.0.0.0 \
    PORT=5000 \
    DATABASE_PATH=/data/cachesec.db \
    RECORDINGS_DIR=/data/recordings \
    SNAPSHOTS_DIR=/data/snapshots \
    UPLOAD_FOLDER=/data/uploads/faces \
    LOG_FILE=/data/logs/cachesec.log

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libfreenect-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0-dev \
        libsm6 \
        libxext6 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p \
        /data/logs \
        /data/recordings \
        /data/snapshots \
        /data/uploads/faces \
        /data/.cachesec/models

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

EXPOSE 5000

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/', timeout=5)"

ENTRYPOINT ["tini", "--"]

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
