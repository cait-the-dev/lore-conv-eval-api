APP = lore-conv-eval-api
PY = poetry run
PORT = 8000
IMAGE = ghcr.io/cait-the-dev/$(APP):dev

.PHONY: help install dev test lint format docker docker-run clean

help:
	@echo "Available targets: install dev test lint format docker docker-run clean"

install:
	poetry install --with dev

dev:
	$(PY) uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	$(PY) pytest -q

lint:
	$(PY) ruff check .

format:
	$(PY) black . && $(PY) isort .

docker:
	docker build -t $(IMAGE) -f infrastructure/docker/Dockerfile .

docker-run:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE)

compose:
	docker compose -f infrastructure/docker/docker-compose.yml up --build

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache