# src/api/Dockerfile

FROM python:3.10.11-slim

WORKDIR /app

# system deps for OpenCV & PIL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements (from src/api)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy backend source code
COPY . .

# expose uvicorn port
EXPOSE 8000

# start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
