# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the parent GraphRAG source code
COPY graphrag /app/graphrag

# Install GraphRAG package in development mode
RUN cd /app && pip install -e ./graphrag

# Copy FastAPI application files
COPY app /app/api

# Install FastAPI application dependencies
RUN pip install -r /app/api/requirements.txt

# Create directories for data
RUN mkdir -p /app/data/input /app/data/output

# Set environment variables
ENV PYTHONPATH=/app:/app/graphrag:/app/api
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
