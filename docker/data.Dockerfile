# Data Service container — pgvector-backed when VECTOR_STORE=pgvector.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e ".[postgres]"

ENV DATA_PORT=50054
EXPOSE 50054

CMD ["python", "-m", "services.data.main"]
