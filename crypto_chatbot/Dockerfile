# Use official Python image as the base
FROM python:3.12.7-slim

# Set environment variables
ENV TRANSFORMERS_CACHE="/models"
ENV PIP_REQUIRE_HASHES=0

WORKDIR /app

# 1. Install system dependencies (for better performance and compatibility)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies first (to leverage Docker cache)
COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Download the model during build to avoid runtime downloads
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3')"

# 4. Copy application code to the container
COPY /src .

EXPOSE 5010

# 5. Define the entry point (modify this based on your app)
CMD ["python", "chatbot_resource.py"]
