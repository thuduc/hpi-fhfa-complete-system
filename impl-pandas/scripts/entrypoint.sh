#!/bin/bash
set -e

# Default command
COMMAND=${1:-serve}

# Function to wait for services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    
    echo "Waiting for $service to be ready..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Wait for dependent services if needed
if [ -n "$HPI_POSTGRES_URL" ]; then
    # Extract host and port from URL
    POSTGRES_HOST=$(echo $HPI_POSTGRES_URL | sed -E 's/.*@([^:]+):.*/\1/')
    POSTGRES_PORT=$(echo $HPI_POSTGRES_URL | sed -E 's/.*:([0-9]+)\/.*/\1/')
    wait_for_service $POSTGRES_HOST $POSTGRES_PORT "PostgreSQL"
fi

if [ -n "$HPI_REDIS_URL" ]; then
    # Extract host and port from URL
    REDIS_HOST=$(echo $HPI_REDIS_URL | sed -E 's/.*\/\/([^:]+):.*/\1/')
    REDIS_PORT=$(echo $HPI_REDIS_URL | sed -E 's/.*:([0-9]+)\/.*/\1/')
    wait_for_service $REDIS_HOST $REDIS_PORT "Redis"
fi

# Run migrations if database is configured
if [ -n "$HPI_POSTGRES_URL" ] && [ "$COMMAND" = "serve" ]; then
    echo "Running database migrations..."
    hpi-fhfa db upgrade
fi

# Execute command
case "$COMMAND" in
    serve)
        echo "Starting HPI-FHFA API server..."
        exec gunicorn \
            --bind 0.0.0.0:8000 \
            --workers ${HPI_WORKERS:-4} \
            --timeout ${HPI_TIMEOUT:-300} \
            --access-logfile - \
            --error-logfile - \
            --log-level ${HPI_LOG_LEVEL:-info} \
            "hpi_fhfa.api.server:create_app()"
        ;;
    worker)
        echo "Starting HPI-FHFA batch worker..."
        exec hpi-fhfa worker \
            --concurrency ${HPI_WORKER_CONCURRENCY:-4}
        ;;
    scheduler)
        echo "Starting HPI-FHFA scheduler..."
        exec hpi-fhfa scheduler
        ;;
    shell)
        echo "Starting interactive shell..."
        exec python
        ;;
    *)
        echo "Executing custom command: $@"
        exec "$@"
        ;;
esac