import os
import sys

# Ensure backend directory is in sys.path regardless of execution directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Configure GPU environment BEFORE importing anything that touches CUDA
from core.gpu_config import configure_environment
configure_environment()

import cv2
import uuid
import threading
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from core.swapper import Swapper
from core.restorer import Restorer
from core.codeformer import CodeFormer
from core.video_processor import process_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Try initializing swapper lazily
swapper_instance = None
restorer_instance = None
codeformer_instance = None

jobs: Dict[str, dict] = {} # Store job progress

def get_swapper():
    global swapper_instance
    if swapper_instance is None:
        try:
            print("[INFO] Initializing Face Swapper model...")
            swapper_instance = Swapper(MODELS_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to init swapper: {e}")
    return swapper_instance

def get_restorer():
    global restorer_instance
    if restorer_instance is None:
        try:
            print("[INFO] Initializing Quality Restorer (GFPGAN)...")
            restorer_instance = Restorer(MODELS_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to init restorer: {e}")
    return restorer_instance

def get_codeformer():
    global codeformer_instance
    if codeformer_instance is None:
        try:
            print("[INFO] Initializing Ultra-Quality Restorer (CodeFormer)...")
            codeformer_instance = CodeFormer(MODELS_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to init codeformer: {e}")
    return codeformer_instance

def run_face_swap_job(job_id: str, src_img_path: str, tgt_vid_path: str, out_vid_path: str, enhance: bool = False, restore_type: str = "gfpgan", fidelity: float = 0.5):
    jobs[job_id]["status"] = "processing"
    
    try:
        swapper = get_swapper()
        if not swapper:
            raise Exception("Swapper model not loaded.")
            
        restorer = None
        if enhance:
            if restore_type == "codeformer":
                try:
                    restorer = get_codeformer()
                except Exception:
                    print("[WARNING] CodeFormer failed to load. Falling back to GFPGAN.")
                    restorer = get_restorer()
            else:
                restorer = get_restorer()
                
            if not restorer:
                print(f"[WARNING] Enhancer failed to load. Proceeding with standard resolution.")

        # Get face from source image
        src_img = cv2.imread(src_img_path)
        source_face = swapper.get_face(src_img)
        
        if source_face is None:
            raise Exception("No face found in source image")
            
        def progress_cb(current, total):
            jobs[job_id]["progress"] = int((current / total) * 100)
            
        process_video(tgt_vid_path, source_face, swapper, out_vid_path, restorer=restorer, fidelity=fidelity, progress_callback=progress_cb)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_url"] = f"/download/{job_id}"
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"[ERROR] Job {job_id} failed: {e}")
    finally:
        import gc
        gc.collect()
        

@app.post("/api/swap-video")
async def swap_video(
    source_image: UploadFile = File(...),
    target_video: UploadFile = File(...),
    enhance: bool = False,
    restore_type: str = "gfpgan",
    fidelity: float = 0.5
):
    job_id = str(uuid.uuid4())
    
    src_img_path = os.path.join(UPLOADS_DIR, f"{job_id}_src.jpg")
    tgt_vid_path = os.path.join(UPLOADS_DIR, f"{job_id}_tgt.mp4")
    out_vid_path = os.path.join(OUTPUTS_DIR, f"{job_id}_out.mp4")
    
    with open(src_img_path, "wb") as f:
        f.write(await source_image.read())
        
    with open(tgt_vid_path, "wb") as f:
        f.write(await target_video.read())
        
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "error": None,
        "result_url": None
    }
    
    # Run in background thread
    thread = threading.Thread(target=run_face_swap_job, args=(job_id, src_img_path, tgt_vid_path, out_vid_path, enhance, restore_type, fidelity))
    thread.start()
    
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/download/{job_id}")
async def get_download(job_id: str):
    out_vid_path = os.path.join(OUTPUTS_DIR, f"{job_id}_out.mp4")
    if not os.path.exists(out_vid_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(out_vid_path, media_type="video/mp4")

# Serve frontend statically
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
