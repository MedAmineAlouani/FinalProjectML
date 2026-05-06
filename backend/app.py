"""
FastAPI server for the Flange-Invariant Acoustic Bolt-Looseness Detector.

Endpoints
---------
GET  /              -> single-page UI (frontend/index.html)
GET  /api/health    -> health probe + model metadata
POST /api/predict   -> multipart upload {audio: file, flange_id: int}
                       returns the segmentation, per-hit probabilities, and
                       averaged Flange-Invariant LR prediction.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import FlangeInvariantLR

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
MODEL_DIR = THIS_DIR / "model"
FRONTEND_DIR = REPO_ROOT / "frontend"

ALLOWED_FLANGES = {1, 2, 3, 4}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB upload cap

app = FastAPI(
    title="Flange-Invariant Acoustic Bolt-Looseness Detector",
    description="UH ML Competition 2026 — percussion-based torque classification.",
    version="1.0.0",
)


@app.on_event("startup")
def _preload_model() -> None:
    """Eager model load so /api/health is meaningful before the first prediction."""
    global _model, _model_load_error
    try:
        _model = FlangeInvariantLR(MODEL_DIR)
        _model_load_error = None
    except Exception as e:
        _model_load_error = (
            f"Model artifacts could not be loaded at startup: {e}. "
            "Run `python backend/train_and_export.py` to generate them."
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the model so the server starts even if the artifacts are missing
# (in which case /api/predict returns a clear error message).
_model: FlangeInvariantLR | None = None
_model_load_error: str | None = None


def get_model() -> FlangeInvariantLR:
    global _model, _model_load_error
    if _model is not None:
        return _model
    try:
        _model = FlangeInvariantLR(MODEL_DIR)
        _model_load_error = None
        return _model
    except FileNotFoundError as e:
        _model_load_error = (
            f"Model artifacts are missing: {e}. "
            "Run `python backend/train_and_export.py` to regenerate them."
        )
        raise HTTPException(status_code=503, detail=_model_load_error)
    except Exception as e:  # pragma: no cover - last-resort surface
        _model_load_error = f"Failed to load model: {e}"
        raise HTTPException(status_code=503, detail=_model_load_error)


@app.get("/api/health")
def health() -> dict:
    """Cheap probe returning model metadata and load status."""
    info = {"status": "ok", "model_loaded": _model is not None}
    if _model_load_error:
        info["model_load_error"] = _model_load_error
    if _model is not None:
        info["model"] = _model.metadata
    return info


@app.post("/api/predict")
async def predict(
    audio: UploadFile = File(..., description="Audio recording of one or more hammer hits."),
    flange_id: int = Form(..., description="Flange identifier: 1, 2, 3, or 4."),
) -> JSONResponse:
    """Run the Flange-Invariant LR pipeline on an uploaded audio file."""
    if flange_id not in ALLOWED_FLANGES:
        raise HTTPException(
            status_code=400,
            detail=f"flange_id must be one of {sorted(ALLOWED_FLANGES)}, got {flange_id}.",
        )

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(raw)} bytes; max {MAX_AUDIO_BYTES}).",
        )

    ext = os.path.splitext(audio.filename or "")[1].lower() or ".webm"

    try:
        result = get_model().predict_from_audio(
            audio_bytes=raw,
            flange_id=int(flange_id),
            file_extension=ext,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(result)


# ---------------------------------------------------------------------- #
# Static frontend
# ---------------------------------------------------------------------- #
if FRONTEND_DIR.exists():
    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    # Mount everything else under /static so we can keep / as the SPA.
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
