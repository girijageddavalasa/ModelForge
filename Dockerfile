FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system modelforge && useradd --system --gid modelforge --home-dir /app modelforge
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x /app/docker-entrypoint.sh && mkdir -p /app/instance /app/storage && chown -R modelforge:modelforge /app

USER modelforge
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health/ready', timeout=3)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]