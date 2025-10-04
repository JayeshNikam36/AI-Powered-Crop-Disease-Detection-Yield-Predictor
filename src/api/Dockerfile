# Use Python slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only backend requirements first (for caching)
COPY src/api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire src folder (so api and models are included)
COPY src ./src

# Ensure Python can import from /app
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 8000

# Run uvicorn with correct import path
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
