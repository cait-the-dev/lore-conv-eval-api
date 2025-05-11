FROM python:3.10-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget git && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry export -f requirements.txt --without-hashes -o deps.txt && \
    pip install --no-cache-dir -r deps.txt && \
    rm deps.txt

RUN pip install --no-cache-dir torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu


FROM python:3.10-slim AS runtime
LABEL org.opencontainers.image.source="https://github.com/cait-the-dev/lore-conv-eval-api"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

WORKDIR /app
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]