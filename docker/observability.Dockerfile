# Observability Service container.

FROM python:3.12-slim

# uv binary copied from the official Astral image — small and deterministic.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

# uv sync installs the project + locked deps into /app/.venv reproducibly.
# `--extra postgres` pulls in psycopg2 so the service can use the
# Postgres-backed store when OBSERVABILITY_POSTGRES_DSN is set.
RUN uv sync --frozen --no-dev --extra postgres

# Put the venv's python on PATH so the CMD resolves to it.
ENV PATH="/app/.venv/bin:$PATH"

ENV OBSERVABILITY_PORT=50059
EXPOSE 50059

# TCP-level readiness: passes once the gRPC server is accepting connections.
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50059)); s.close()" || exit 1

CMD ["python", "-m", "services.observability.main"]
