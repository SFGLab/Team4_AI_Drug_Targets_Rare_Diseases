# Use a base image with Python and basic tools
FROM python:3.10-slim

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Install Python packages
RUN pip install --upgrade pip
RUN pip install \
    torch \
    transformers \
    pandas \
    scikit-learn \
    tqdm