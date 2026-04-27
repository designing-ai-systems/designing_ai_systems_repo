# Model Service container.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV MODELS_PORT=50053
EXPOSE 50053

CMD ["python", "-m", "services.models.main"]
