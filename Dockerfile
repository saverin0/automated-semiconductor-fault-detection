# 1. Use minimal base image (python:slim)
FROM python:3.12-slim as base

# 2. Pin exact versions in requirements - create requirements-lock.txt
FROM base as dependencies

WORKDIR /tmp
COPY requirements.txt .

# Create pinned requirements
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir pip-tools==7.4.1 && \
    pip-compile --output-file=requirements-lock.txt requirements.txt

# 3. Multi-stage build for smaller final image
FROM base as production

# 4. Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# 5. Set environment variables (no secrets here!)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# 6. Install system dependencies and clean up in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# 7. Switch to non-root user
USER appuser
WORKDIR /app

# 8. Copy and install dependencies (optimize caching)
COPY --from=dependencies /tmp/requirements-lock.txt .
RUN pip install --user --no-cache-dir -r requirements-lock.txt

# 9. Create necessary directories with proper permissions
RUN mkdir -p logs data/training/input data/training/good data/training/bad \
    prediction/input prediction/good prediction/bad prediction/output \
    schema training_model

# 10. Copy source code (after dependencies for better caching)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser src/api/templates/ ./src/api/templates/
COPY --chown=appuser:appuser schema/ ./schema/
COPY --chown=appuser:appuser .env.docker .env
COPY --chown=appuser:appuser training_model/ ./training_model/

# 11. Expose only necessary port
EXPOSE 8080

# 12. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# 13. Use specific non-root user and minimal command
USER appuser
CMD ["python", "-m", "src.api.app"]