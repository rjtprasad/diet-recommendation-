FROM python:3.12-slim

# Install curl for healthcheck (optional)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p /app/backend

# Copy requirements and install
COPY FastAPI_Backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code + model/data files
COPY FastAPI_Backend/ /app/backend/
COPY Data/ /app/Data/

WORKDIR /app/backend

# Expose any port (Railway uses dynamic PORT)
EXPOSE 8080

# Dynamic PORT from Railway environment
ENV PORT 8080

# Optional: Remove old healthcheck (will fail on dynamic port)
# CMD automatically checks if server is up

# Start uvicorn using dynamic PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]