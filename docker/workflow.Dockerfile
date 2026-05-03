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

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV WORKFLOW_PORT=50058
EXPOSE 50058

CMD ["python", "-m", "services.workflow.main"]
