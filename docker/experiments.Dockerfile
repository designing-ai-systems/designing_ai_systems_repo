# Experimentation Service container.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

# `--extra postgres` pulls in psycopg2 so the service can use the
# Postgres-backed store when EXPERIMENTS_POSTGRES_DSN is set.
RUN uv sync --frozen --no-dev --extra postgres

ENV PATH="/app/.venv/bin:$PATH"

ENV EXPERIMENTS_PORT=50060
EXPOSE 50060

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50060)); s.close()" || exit 1

CMD ["python", "-m", "services.experiments.main"]
