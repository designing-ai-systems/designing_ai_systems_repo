# Sessions Service container.
#
# One image per platform service (chapter 8 plan, "Local development &
# production-deployment architecture"). The same artifact a platform team
# would push to a registry and reference from a Kubernetes Deployment.
#
# Build context: repo root. ``docker compose build`` handles this.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

# Install the SDK + the postgres extra so PostgreSQL-backed storage works
# when SESSION_STORAGE=postgres in the compose env.
RUN uv sync --frozen --no-dev --extra postgres

ENV PATH="/app/.venv/bin:$PATH"

ENV SESSIONS_PORT=50052
EXPOSE 50052

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50052)); s.close()" || exit 1

CMD ["python", "-m", "services.sessions.main"]
