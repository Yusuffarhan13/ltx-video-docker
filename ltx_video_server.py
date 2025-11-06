"""
LTX Video Generation Server

FastAPI server that runs inside Docker container on Vast.ai.
Provides endpoints for video generation using LTX Video model.
"""

import os
import uuid
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LTX Video Generation Server",
    description="On-demand video generation using LTX Video model",
    version="1.0.0"
)

# Global state
jobs: Dict[str, Dict] = {}
model = None
OUTPUT_DIR = Path("/tmp/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class GenerationRequest(BaseModel):
    """Video generation request"""
    prompt: str
    duration: int = 8  # seconds
    resolution: str = "768x512"
    fps: int = 25
    guidance_scale: float = 3.0
    num_inference_steps: int = 50


class GenerationResponse(BaseModel):
    """Video generation response"""
    job_id: str
    status: str
    message: str


def load_ltx_model():
    """Load LTX Video model into memory"""
    global model

    try:
        logger.info("Loading LTX Video model...")

        # Check if GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! GPU required for video generation.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Import LTX Video pipeline
        from diffusers import LTXPipeline

        # Load model with optimizations
        model = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        # Enable memory optimizations
        model.enable_model_cpu_offload()
        model.enable_vae_slicing()

        logger.info("LTX Video model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


async def generate_video_task(job_id: str, request: GenerationRequest):
    """Background task to generate video"""
    global jobs, model

    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 10

        logger.info(f"Starting video generation for job {job_id}")
        logger.info(f"Prompt: {request.prompt[:100]}...")

        # Calculate number of frames
        num_frames = request.duration * request.fps

        # Parse resolution
        width, height = map(int, request.resolution.split('x'))

        jobs[job_id]['progress'] = 20

        # Generate video
        logger.info(f"Generating {num_frames} frames at {width}x{height}")

        output = model(
            prompt=request.prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        jobs[job_id]['progress'] = 80

        # Export video
        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        logger.info(f"Exporting video to {output_path}")

        # Save frames as video using imageio
        import imageio
        frames = output.frames[0]  # Get first video

        writer = imageio.get_writer(
            output_path,
            fps=request.fps,
            codec='libx264',
            pixelformat='yuv420p'
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

        jobs[job_id]['progress'] = 100
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['output_path'] = str(output_path)
        jobs[job_id]['completed_at'] = datetime.now().isoformat()

        file_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"Video generation complete: {output_path} ({file_size:.2f} MB)")

    except Exception as e:
        logger.error(f"Error generating video for job {job_id}: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("=" * 60)
    logger.info("LTX Video Server Starting...")
    logger.info("=" * 60)

    # Load model
    success = load_ltx_model()

    if not success:
        logger.error("Failed to load model! Server will not be able to generate videos.")
    else:
        logger.info("Server ready to accept requests!")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "ltx-video",
        "ready": model is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        "jobs_active": len([j for j in jobs.values() if j['status'] == 'processing']),
        "jobs_total": len(jobs)
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe - checks if model is loaded and GPU available"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="GPU not available")

    return {
        "ready": True,
        "gpu": torch.cuda.get_device_name(0),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Start video generation job"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create job
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'prompt': request.prompt,
        'duration': request.duration,
        'resolution': request.resolution,
        'progress': 0,
        'created_at': datetime.now().isoformat(),
        'output_path': None,
        'error': None
    }

    logger.info(f"Job {job_id} created and queued")

    # Start generation in background
    background_tasks.add_task(generate_video_task, job_id, request)

    return GenerationResponse(
        job_id=job_id,
        status='queued',
        message=f'Video generation started. Poll /status/{job_id} for progress.'
    )


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'created_at': job['created_at'],
        'completed_at': job.get('completed_at'),
        'error': job.get('error')
    }


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download completed video"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    output_path = job['output_path']

    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete job and cleanup files"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Delete output file if exists
    if job.get('output_path'):
        try:
            os.remove(job['output_path'])
            logger.info(f"Deleted output file for job {job_id}")
        except:
            pass

    # Remove from jobs dict
    del jobs[job_id]

    return {"message": f"Job {job_id} deleted"}


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(jobs),
        "jobs": [
            {
                'job_id': j['job_id'],
                'status': j['status'],
                'progress': j['progress'],
                'created_at': j['created_at']
            }
            for j in jobs.values()
        ]
    }


@app.post("/cleanup")
async def cleanup_completed_jobs():
    """Cleanup completed jobs older than 1 hour"""
    from datetime import datetime, timedelta

    cleaned = 0
    now = datetime.now()

    for job_id in list(jobs.keys()):
        job = jobs[job_id]

        if job['status'] in ['completed', 'failed']:
            created_at = datetime.fromisoformat(job['created_at'])
            age = now - created_at

            if age > timedelta(hours=1):
                # Delete file
                if job.get('output_path'):
                    try:
                        os.remove(job['output_path'])
                    except:
                        pass

                # Remove job
                del jobs[job_id]
                cleaned += 1

    logger.info(f"Cleaned up {cleaned} old jobs")

    return {"cleaned": cleaned, "remaining": len(jobs)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
