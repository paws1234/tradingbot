# TradingBot image — Dockerized FastAPI + asyncio trading engine (OANDA + DeepSeek).
#
# Build (from TradingBot/):  docker build -t tradingbot .
# Run:   docker run --env-file .env -p 8000:8000 tradingbot
# Probe: curl localhost:8000/health

FROM python:3.12-slim

# Unbuffered stdout for Docker logs; no __pycache__ writes (read-only /app).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root runtime user (plan: python:3.12-slim, non-root).
RUN useradd --create-home appuser

WORKDIR /app

# Install pinned dependencies first so the layer stays cached across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (tests/ and local tooling are excluded by .dockerignore).
COPY app ./app

# Default port; the uvicorn CMD and healthcheck honor a Render-injected PORT.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os; from urllib.request import urlopen; r = urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=5); r.read(); r.close()"

USER appuser

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
