import os
import cv2

def process_video(target_video_path, source_face, swapper, output_path, restorer=None, fidelity=0.5, progress_callback=None):
    """Processes video frame by frame, swaps faces, and restores audio."""
    
    # Reset temporal stabiliser for fresh video
    if hasattr(swapper, 'reset_stabilizer'):
        swapper.reset_stabilizer()
    
    # 1. Open Video
    v_cap = cv2.VideoCapture(target_video_path)
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if video was loaded
    if not v_cap.isOpened() or fps == 0:
        raise Exception("Failed to open target video")

    # 2. Setup Video Writer for processed frames (temp video)
    temp_output_path = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    try:
        frame_idx = 0
        while True:
            ret, frame = v_cap.read()
            if not ret:
                break
            
            # Swap face
            processed_frame = swapper.swap_face(frame, source_face, restorer=restorer, fidelity=fidelity)
            v_writer.write(processed_frame)
            
            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)
    finally:
        v_cap.release()
        v_writer.release()
        
    # 3. Add original audio back and transcode to web-standard H.264 for browser playback
    print("[INFO] Encoding web-compatible H.264 with audio via ffmpeg...")
    import subprocess
    import imageio_ffmpeg
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # Hardware NVENC H.264 encoder (instant ~3s on NVIDIA GPU)
        cmd_nvenc = [
            ffmpeg_exe, "-y",
            "-i", temp_output_path,
            "-i", target_video_path,
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            "-preset", "p4",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
        res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        encoded_ok = (res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000)
        
        if not encoded_ok:
            # Fallback to libx264 CPU encoder
            print("[INFO] NVENC fallback to libx264...")
            cmd_x264 = [
                ffmpeg_exe, "-y",
                "-i", temp_output_path,
                "-i", target_video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ]
            res = subprocess.run(cmd_x264, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            encoded_ok = (res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000)

        if encoded_ok:
            print("[INFO] H.264 video encoded successfully.")
        else:
            # Fallback if ffmpeg failed
            if os.path.exists(temp_output_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output_path, output_path)
    except Exception as e:
        print(f"[WARNING] Fast ffmpeg remux fallback: {e}")
        if os.path.exists(temp_output_path) and not os.path.exists(output_path):
            try:
                os.rename(temp_output_path, output_path)
            except Exception:
                pass
    finally:
        if os.path.exists(temp_output_path) and os.path.exists(output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
