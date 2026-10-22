"""
Real-Time Surgical Tool Tracking System
-----------------------------------------
FastAPI backend that:
  1. Serves a clinical dashboard UI (Jinja2 templates).
  2. Accepts an uploaded MP4 video and saves it to disk (data/uploaded_feed.mp4).
  3. Streams YOLOv8 inference results as an MJPEG feed via /video_feed.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Paths & Constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATH = os.path.join(MODELS_DIR, "detector.pt")
VIDEO_PATH = os.path.join(DATA_DIR, "uploaded_feed.mp4")

CONFIDENCE_THRESHOLD = 0.50

# Ensure the data directory exists (per the required folder structure)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# App Initialization
# --------------------------------------------------------------------------
app = FastAPI(
    title="Real-Time Surgical Tool Tracking System",
    description="YOLOv8 single-stage detector for live surgical instrument tracking.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------
# Loaded once at startup so that inference requests do not pay the load cost.
model = None

# --------------------------------------------------------------------------
# Real-Time FPS Tracking (updated inside generate_frames(), read by /get_fps)
# --------------------------------------------------------------------------
current_fps = 0.0
_fps_smoothing = 0.9  # exponential moving average factor for a stable readout


@app.on_event("startup")
def load_model() -> None:
    """Load the YOLOv8 model weights into memory on server startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        # We don't crash the whole server -- we let it start so the UI
        # still loads, but /video_feed will report a clear error.
        print(f"[WARNING] Model weights not found at: {MODEL_PATH}")
        return
    print(f"[INFO] Loading YOLOv8 model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept an uploaded MP4 file and persist it to disk as
    data/uploaded_feed.mp4 so that /video_feed can stream inference
    directly from local disk (fastest read path, avoids re-buffering
    in memory).
    """
    if file.content_type not in ("video/mp4", "application/octet-stream"):
        # Some browsers send generic octet-stream for mp4; we still validate
        # by extension as a fallback rather than rejecting outright.
        if not file.filename.lower().endswith(".mp4"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an MP4 video.",
            )

    try:
        with open(VIDEO_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}")
    finally:
        file.file.close()

    # Quick sanity check: can OpenCV actually open the file?
    cap_check = cv2.VideoCapture(VIDEO_PATH)
    is_readable = cap_check.isOpened()
    cap_check.release()

    if not is_readable:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be read as a valid video.",
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Video uploaded successfully.",
            "filename": "uploaded_feed.mp4",
            "path": VIDEO_PATH,
        },
    )


def generate_frames():
    """
    Generator that reads the saved video frame-by-frame, runs YOLOv8
    inference on every 3rd frame (frame skipping for high FPS), draws bounding
    boxes, and yields JPEG-encoded bytes wrapped in a multipart/x-mixed-replace
    boundary for MJPEG streaming.
    """
    if model is None:
        error_frame = _make_text_frame("MODEL NOT LOADED")
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
        )
        return

    if not os.path.exists(VIDEO_PATH):
        error_frame = _make_text_frame("NO VIDEO UPLOADED")
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
        )
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    prev_frame_time = time.time()

    if not cap.isOpened():
        error_frame = _make_text_frame("UNABLE TO OPEN VIDEO")
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + error_frame + b"\r\n"
        )
        return

    frame_count = 0
    last_annotated_frame = None

    try:
        while True:
            success, frame = cap.read()

            if not success:
                # Loop the video for a continuous demo-friendly feed.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # --- Frame Skipping Logic (Run YOLO inference every 3rd frame) ---
            if frame_count % 3 == 0 or last_annotated_frame is None:
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=320,
                    verbose=False,
                )
                last_annotated_frame = results[0].plot()

            # --- Real FPS calculation ---
            global current_fps
            new_frame_time = time.time()
            elapsed = new_frame_time - prev_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                current_fps = (_fps_smoothing * current_fps) + ((1 - _fps_smoothing) * instant_fps)
            prev_frame_time = new_frame_time

            # Encode the annotated frame as JPEG.
            ret, buffer = cv2.imencode(".jpg", last_annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()


def _make_text_frame(message: str) -> bytes:
    """Utility: create a simple dark JPEG frame with an overlay message."""
    frame = np.full((480, 640, 3), 30, dtype="uint8")
    cv2.putText(
        frame,
        message,
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    ret, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ret else b""


@app.get("/video_feed")
def video_feed():
    """
    Stream MJPEG frames with YOLOv8 bounding boxes drawn on each frame.
    The multipart/x-mixed-replace content type tells the browser to keep
    replacing the image content with each new part -- this is what gives
    the live, real-time video effect inside a plain <img> tag.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/get_fps")
def get_fps():
    """Return the real, currently-measured inference FPS from the video_feed loop."""
    return {"fps": round(current_fps, 1)}


@app.get("/status")
def status():
    """Simple health/status endpoint used by the frontend to check readiness."""
    return {
        "model_loaded": model is not None,
        "video_available": os.path.exists(VIDEO_PATH),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
