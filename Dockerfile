# DRDO MALE UAV Aero Piston Engine Digital Twin - Container Image
# Multi-stage production image for tactical shelter server deployment

FROM python:3.11-slim

LABEL maintainer="DRDO Propulsion Directorate" \
      project="SIH26054-DigitalTwin" \
      version="2.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for socket and math libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    can-utils \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source tree
COPY . .

# Expose HTTP/WebSocket port (8000) and HIL UDP CAN port (5555)
EXPOSE 8000 5555/udp

# Healthcheck endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Launch the GCS server daemon
CMD ["python", "run_gcs.py", "--host", "0.0.0.0", "--port", "8000"]
