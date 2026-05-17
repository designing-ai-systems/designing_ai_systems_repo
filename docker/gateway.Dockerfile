# API Gateway container.
#
# Exposes 8080 (external HTTP -> workflows) and 50051 (internal gRPC ->
# platform services). The only platform service whose ports are mapped to
# the host in docker-compose, since external clients live on the host.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080
EXPOSE 50051

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50051)); s.close()" || exit 1

CMD ["python", "-m", "services.gateway.main"]
