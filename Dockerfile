FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies
# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Colombo apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata && \
    rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /workspace/app

# Install Python dependencies
RUN pip install --no-cache-dir \
    diffusers[torch] \
    transformers \
    accelerate \
    fastapi \
    uvicorn[standard] \
    imageio[ffmpeg] \
    pydantic \
    sentencepiece \
    protobuf

# Note: Model (~10-20GB) will be downloaded automatically on first container start
# This keeps Docker build fast and image small

# Copy server script
COPY ltx_video_server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "ltx_video_server.py"]
