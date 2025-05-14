# Dockerfile

# Use a slim Python base image with a specific version (e.g., Python 3.10)
FROM python:3.10-slim

# Optional: Avoids interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements and scripts
COPY requirements.txt .
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install additional system dependencies (if needed)
# RUN apt-get update && apt-get install -y libgl1

# Default command (can be overridden)
CMD ["python"]
