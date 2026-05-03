# Observability Service container.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV OBSERVABILITY_PORT=50059
EXPOSE 50059

CMD ["python", "-m", "services.observability.main"]
