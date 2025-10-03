# Use Python slim image
FROM python:3.10.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src folder from repo root
COPY ./src ./src

# Set environment variable so Python can find src
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Start the app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
