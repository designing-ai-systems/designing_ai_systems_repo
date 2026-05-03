# Guardrails Service container.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV GUARDRAILS_PORT=50055
EXPOSE 50055

CMD ["python", "-m", "services.guardrails.main"]
