# Observability Streamlit dashboard container.
#
# Reads through the platform SDK against the gateway, so it requires no
# direct database connection. Configure GENAI_GATEWAY_URL to point at
# the gateway (default: gateway:50051 inside compose).

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY genai_platform/ ./genai_platform/
COPY proto/ ./proto/
COPY services/ ./services/
COPY dashboards/ ./dashboards/

# Streamlit + plotly + pandas live in the `dashboards` optional extra.
RUN uv sync --frozen --no-dev --extra dashboards

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "dashboards/observability/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
