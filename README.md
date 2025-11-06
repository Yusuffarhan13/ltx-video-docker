# LTX Video Docker API

Production-ready Docker image for LTX Video generation with FastAPI server. Optimized for cloud deployment with pre-downloaded models.

## Docker Image

```bash
docker pull yusuffarhan/ltx-video:latest
```

## Features

- 🚀 Pre-downloaded LTX Video model (~10-20GB) - instant container starts
- 🔥 GPU-accelerated inference with CUDA 12.4
- 🌐 FastAPI REST API with automatic docs
- 📦 Single Docker image - no external dependencies
- ☁️ Optimized for Vast.ai and cloud deployment

## Quick Start

### Pull and Run

```bash
docker pull yusuffarhan/ltx-video:latest
docker run --gpus all -p 8000:8000 yusuffarhan/ltx-video:latest
```

### Generate a Video

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano in a jazz club",
    "num_frames": 121,
    "height": 480,
    "width": 704,
    "num_inference_steps": 50
  }' \
  --output video.mp4
```

## API Documentation

Once running, visit:
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Endpoints

#### POST `/generate`

Generate a video from a text prompt.

**Request Body:**
```json
{
  "prompt": "string",
  "num_frames": 121,
  "height": 480,
  "width": 704,
  "num_inference_steps": 50,
  "guidance_scale": 3.0
}
```

**Response:** Video file (MP4)

#### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true
}
```

## Vast.ai Deployment

### Using the Pre-built Image

1. Go to [Vast.ai](https://vast.ai)
2. Click "Create" → "New Instance"
3. Configure:
   - **Image**: `yusuffarhan/ltx-video:latest`
   - **GPU**: Any GPU with 24GB+ VRAM (e.g., RTX 4090, A6000)
   - **Disk Space**: 40GB minimum
   - **On-start script**: Leave empty (model pre-downloaded)
4. Expose port `8000`
5. Launch instance

Your API will be available at: `http://<instance-ip>:8000`

### Environment Variables

You can customize the server with environment variables:

```bash
docker run --gpus all -p 8000:8000 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  yusuffarhan/ltx-video:latest
```

## Building from Source

### Clone Repository

```bash
git clone https://github.com/Yusuffarhan13/ltx-video-docker.git
cd ltx-video-docker
```

### Build Docker Image

```bash
docker build -t yusuffarhan/ltx-video:latest .
```

**Note:** Build downloads ~10-20GB model weights and takes 30-60 minutes.

### Push to Docker Hub

```bash
docker push yusuffarhan/ltx-video:latest
```

## Local Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Server

```bash
python ltx_video_server.py
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM
- **RAM**: 32GB+ recommended
- **Disk**: 40GB+ for model weights and temp files
- **CUDA**: 12.1+ (included in Docker image)

## Performance Tips

1. **Batch Processing**: Process multiple videos sequentially in the same container to reuse loaded model
2. **Resolution**: Lower resolutions (480x704) generate faster than 720x1280
3. **Frames**: Fewer frames (121) are faster than maximum (257)
4. **Inference Steps**: 30-40 steps often sufficient, 50 for highest quality

## Troubleshooting

### Out of Memory

Reduce resolution or number of frames:
```json
{
  "height": 480,
  "width": 704,
  "num_frames": 121
}
```

### Slow Generation

- Ensure GPU is being used (check `/health` endpoint)
- Reduce `num_inference_steps` to 30-40
- Use smaller resolution

### Container Won't Start

- Check GPU availability: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Ensure enough disk space (40GB+)
- Check Docker has GPU access configured

## Cost Optimization

Using Vast.ai on-demand instances:
- **RTX 4090**: ~$0.30/hour (recommended)
- **A6000**: ~$0.40/hour
- **RTX 3090**: ~$0.20/hour (minimum for 24GB VRAM)

Average video generation: 2-3 minutes = $0.01-0.02 per video

## License

This is a wrapper around the LTX Video model from Lightricks. See [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) for model license.

## Credits

- **Model**: [Lightricks LTX Video](https://huggingface.co/Lightricks/LTX-Video)
- **Framework**: [Diffusers](https://github.com/huggingface/diffusers)
- **Server**: [FastAPI](https://fastapi.tiangolo.com/)
