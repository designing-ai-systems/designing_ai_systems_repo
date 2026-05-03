# Experimentation Service container.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV EXPERIMENTS_PORT=50060
EXPOSE 50060

CMD ["python", "-m", "services.experiments.main"]
