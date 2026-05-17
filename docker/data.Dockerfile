# Data Service container — pgvector-backed when VECTOR_STORE=pgvector.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN uv sync --frozen --no-dev --extra postgres

ENV PATH="/app/.venv/bin:$PATH"

ENV DATA_PORT=50054
EXPOSE 50054

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50054)); s.close()" || exit 1

CMD ["python", "-m", "services.data.main"]
