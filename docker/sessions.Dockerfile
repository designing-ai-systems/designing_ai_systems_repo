# Sessions Service container.
#
# One image per platform service (chapter 8 plan, "Local development &
# production-deployment architecture"). The same artifact a platform team
# would push to a registry and reference from a Kubernetes Deployment.
#
# Build context: repo root. ``docker compose build`` handles this.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/

# Install the SDK + the postgres extra so PostgreSQL-backed storage works
# when SESSION_STORAGE=postgres in the compose env.
RUN pip install --no-cache-dir -e ".[postgres]"

ENV SESSIONS_PORT=50052
EXPOSE 50052

CMD ["python", "-m", "services.sessions.main"]
