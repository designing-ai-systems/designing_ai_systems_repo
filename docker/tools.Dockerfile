# Tool Service container.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENV TOOLS_PORT=50056
EXPOSE 50056

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50056)); s.close()" || exit 1

CMD ["python", "-m", "services.tools.main"]
