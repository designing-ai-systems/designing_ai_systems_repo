"""Chapter-8 deploy CLI.

``genai-platform deploy <file.py>`` scans the file for ``@workflow``
functions, generates a Dockerfile + Kubernetes manifests for each, builds
a Docker image, and registers/deploys the workflow with the platform's
Workflow Service. See ``genai_platform/cli/deploy.py`` for the
orchestration.
"""

from .deploy import main

__all__ = ["main"]
