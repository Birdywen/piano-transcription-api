"""Piano transcription API (solo piano, ByteDance high-resolution model).

Endpoints:
  POST /api/jobs                  upload audio -> {"job_id": ...}
  GET  /api/jobs/{job_id}         status + result urls
  GET  /api/jobs/{job_id}/midi    download MIDI
  GET  /api/jobs/{job_id}/musicxml  download MusicXML (best effort)
  DELETE /api/jobs/{job_id}       remove job + files
  GET  /health                    device + queue depth
"""
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

DATA_DIR = Path(os.environ.get("PIANO_DATA_DIR", "/data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "jobs.db"
DEVICE = os.environ.get("PIANO_DEVICE", "cuda")
ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="piano-transcription-api")
_executor = ThreadPoolExecutor(max_workers=1)  # GPU: one job at a time
_transcriptor = None
_transcriptor_lock = threading.Lock()


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs(
                 id TEXT PRIMARY KEY, filename TEXT, status TEXT,
                 error TEXT, created_at REAL, finished_at REAL,
                 note_count INTEGER)"""
        )


_init_db()


def _get_transcriptor():
    global _transcriptor
    with _transcriptor_lock:
        if _transcriptor is None:
            import torch
            from piano_transcription_inference import PianoTranscription

            device = DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            _transcriptor = PianoTranscription(device=device)
            _transcriptor.device_used = device
    return _transcriptor


def _convert_to_musicxml(midi_path: Path, xml_path: Path) -> bool:
    try:
        from music21 import converter

        score = converter.parse(str(midi_path))
        score.write("musicxml", fp=str(xml_path))
        return True
    except Exception:
        return False


def _run_job(job_id: str, src_path: Path):
    from piano_transcription_inference import sample_rate
    import soundfile as sf

    started = time.time()
    try:
        with _db() as conn:
            conn.execute("UPDATE jobs SET status='working' WHERE id=?", (job_id,))
        transcriptor = _get_transcriptor()
        audio, _ = sf.read(str(src_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        import librosa

        if sf.info(str(src_path)).samplerate != sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sf.info(str(src_path)).samplerate,
                target_sr=sample_rate,
            )
        midi_path = OUTPUT_DIR / f"{job_id}.mid"
        result = transcriptor.transcribe(audio, str(midi_path))
        note_count = len(result.get("est_note_events", []))
        xml_path = OUTPUT_DIR / f"{job_id}.musicxml"
        _convert_to_musicxml(midi_path, xml_path)
        with _db() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', finished_at=?, note_count=? WHERE id=?",
                (time.time(), note_count, job_id),
            )
    except Exception as exc:  # noqa: BLE001
        with _db() as conn:
            conn.execute(
                "UPDATE jobs SET status='error', error=?, finished_at=? WHERE id=?",
                (f"{type(exc).__name__}: {exc}", time.time(), job_id),
            )
    finally:
        print(f"[job {job_id}] finished in {time.time() - started:.1f}s")


def _job_dict(row) -> dict:
    d = dict(row)
    job_id = d["id"]
    d["midi_url"] = f"/api/jobs/{job_id}/midi" if (OUTPUT_DIR / f"{job_id}.mid").exists() else None
    xml = OUTPUT_DIR / f"{job_id}.musicxml"
    d["musicxml_url"] = f"/api/jobs/{job_id}/musicxml" if xml.exists() else None
    return d


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported format {ext!r}, allowed: {sorted(ALLOWED_EXTS)}")
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    with _db() as conn:
        conn.execute(
            "INSERT INTO jobs(id, filename, status, created_at) VALUES(?, ?, 'queued', ?)",
            (job_id, file.filename, time.time()),
        )
    _executor.submit(_run_job, job_id, dest)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "job not found")
    return _job_dict(row)


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_job_dict(r) for r in rows]


@app.get("/api/jobs/{job_id}/midi")
def download_midi(job_id: str):
    p = OUTPUT_DIR / f"{job_id}.mid"
    if not p.exists():
        raise HTTPException(404, "midi not ready")
    return FileResponse(p, media_type="audio/midi", filename=f"{job_id}.mid")


@app.get("/api/jobs/{job_id}/musicxml")
def download_musicxml(job_id: str):
    p = OUTPUT_DIR / f"{job_id}.musicxml"
    if not p.exists():
        raise HTTPException(404, "musicxml not ready")
    return FileResponse(p, media_type="application/xml", filename=f"{job_id}.musicxml")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "job not found")
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    for p in UPLOAD_DIR.glob(f"{job_id}.*"):
        p.unlink(missing_ok=True)
    for p in OUTPUT_DIR.glob(f"{job_id}.*"):
        p.unlink(missing_ok=True)
    return {"deleted": job_id}


@app.get("/health")
def health():
    import torch

    with _db() as conn:
        queued = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','working')"
        ).fetchone()[0]
    return {
        "ok": True,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _transcriptor is not None,
        "pending_jobs": queued,
    }
