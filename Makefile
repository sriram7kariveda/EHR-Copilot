.PHONY: install dev lint typecheck test test-unit test-integration test-eval run-api run-ui docker-up docker-down synthea download-models

install:
	uv sync

dev:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/unit/ tests/integration/ -v --tb=short

test-unit:
	uv run pytest tests/unit/ -v --tb=short

test-integration:
	uv run pytest tests/integration/ -v --tb=short -m integration

test-eval:
	uv run python tests/evaluation/eval_runner.py

run-api:
	uv run uvicorn ehr_copilot.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

run-ui:
	uv run streamlit run src/ehr_copilot/ui/streamlit_app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

synthea:
	bash scripts/generate_synthea.sh

download-models:
	bash scripts/download_models.sh
