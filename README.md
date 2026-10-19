# DeepSwap

GPU-accelerated face replacement for images and videos, with optional GFPGAN
or CodeFormer enhancement.

## Requirements

- Windows
- Python 3.10 or newer
- NVIDIA GPU with compatible CUDA drivers (recommended)

## Getting started

1. Run `start.bat`.
2. The launcher creates the local virtual environment and installs
   `requirements.txt`.
3. Required model files are downloaded into `models/`.
4. Open <http://127.0.0.1:8000>.

The model files, uploaded media, generated videos, and local virtual
environment are intentionally excluded from Git. Run
`backend\download_models.py` directly if the model files need to be downloaded
or repaired.

## API

- `POST /api/swap-video` accepts `source_image`, `target_video`, `enhance`,
  `restore_type`, and `fidelity`.
- `GET /api/status/{job_id}` returns background job progress.
- `GET /download/{job_id}` downloads a completed video.

Use only media for which you have the necessary rights and consent.
