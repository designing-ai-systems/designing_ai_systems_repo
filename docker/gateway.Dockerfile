# API Gateway container.
#
# Exposes 8080 (external HTTP -> workflows) and 50051 (internal gRPC ->
# platform services). The only platform service whose ports are mapped to
# the host in docker-compose, since external clients live on the host.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

RUN pip install --no-cache-dir -e .

EXPOSE 8080
EXPOSE 50051

CMD ["python", "-m", "services.gateway.main"]
