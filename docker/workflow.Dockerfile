# Workflow Service container.
#
# Slightly heavier than the other platform services because the Workflow
# Service shells out to ``docker run`` to launch new workflow containers
# (chapter 8 ``DeployWorkflow``). For that to work inside compose, the
# image needs the docker CLI and the host's docker socket needs to be
# mounted in (``docker-compose.yml`` does the mount).
#
# When `WORKFLOW_DOCKER_NETWORK` is set in compose, the deployer attaches
# new workflow containers to the same compose network so the gateway can
# reach them by name without any host port mapping.

FROM python:3.12-slim

# docker.io brings the Docker CLI; we just need the client to talk to the
# host's docker daemon over the mounted socket. (No daemon runs in here.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

ENV WORKFLOW_PORT=50058
EXPOSE 50058

CMD ["python", "-m", "services.workflow.main"]
