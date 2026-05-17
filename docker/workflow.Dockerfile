# Workflow Service container.
#
# Pure bookkeeping service — registry, deployment records, async jobs,
# route push to the gateway. The Workflow Service does NOT shell out to
# docker; that responsibility lives in the `genai-platform deploy` CLI,
# which runs on the developer's host where Docker already lives. So this
# image stays small and unprivileged: no docker CLI, no /var/run/docker.sock
# mount in compose. See chapters/book_discrepancies_chapter8.md for the
# rationale (the chapter prescribes the Workflow Service calling the
# Kubernetes API; for our local Docker demo we put the docker action in
# the CLI to avoid giving this service privileged access to the host).

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENV WORKFLOW_PORT=50058
EXPOSE 50058

HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost', 50058)); s.close()" || exit 1

CMD ["python", "-m", "services.workflow.main"]
