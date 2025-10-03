# Use official slim Python image
FROM python:3.10.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire src folder
COPY src ./src

# Set PYTHONPATH so Python can find src
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
