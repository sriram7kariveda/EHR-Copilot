FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync --no-dev

COPY config/ config/
COPY src/ src/

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "ehr_copilot.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
