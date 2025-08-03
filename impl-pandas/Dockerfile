# Multi-stage build for HPI-FHFA application
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements
COPY setup.py setup.cfg README.md ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgeos-c1v5 \
    libproj19 \
    libgdal28 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 hpiuser

# Set work directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install application
RUN pip install --no-cache-dir --no-index --find-links /wheels hpi-fhfa && \
    rm -rf /wheels

# Copy configuration files
COPY config/ ./config/
COPY scripts/entrypoint.sh ./

# Create necessary directories
RUN mkdir -p data logs cache results && \
    chown -R hpiuser:hpiuser /app

# Switch to non-root user
USER hpiuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HPI_CONFIG_PATH=/app/config/production.yaml
ENV HPI_DATA_PATH=/app/data
ENV HPI_LOG_PATH=/app/logs
ENV HPI_CACHE_PATH=/app/cache

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["serve"]