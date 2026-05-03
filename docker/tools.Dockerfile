# Tool Service container.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV TOOLS_PORT=50056
EXPOSE 50056

CMD ["python", "-m", "services.tools.main"]
