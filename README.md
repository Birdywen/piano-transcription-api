# piano-transcription-api

Solo-piano audio → MIDI + MusicXML, powered by ByteDance's high-resolution
piano transcription model (`piano_transcription_inference`), served as an
async FastAPI service in a CUDA Docker image. Runs on RTX 5090.

## Run

```bash
docker compose up -d --build
```

First build takes ~10–15 min (CUDA base, torch cu128, model checkpoint baked in).

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/jobs` | Upload audio (mp3/wav/flac/m4a/ogg) → `{"job_id": ...}` |
| `GET` | `/api/jobs/{id}` | Status (`queued`/`working`/`done`/`error`) + result URLs |
| `GET` | `/api/jobs/{id}/midi` | Download MIDI |
| `GET` | `/api/jobs/{id}/musicxml` | Download MusicXML (best effort) |
| `GET` | `/api/jobs` | Recent jobs |
| `DELETE` | `/api/jobs/{id}` | Delete job + files |
| `GET` | `/health` | GPU status + queue depth |

Example:

```bash
curl -F file=@nocturne.mp3 http://HOST:7777/api/jobs
curl http://HOST:7777/api/jobs/<job_id>
curl -o out.mid http://HOST:7777/api/jobs/<job_id>/midi
```

## Notes

- One job at a time on GPU (single worker); polling interval 5–10s is plenty.
- Transcription of a 4-min piece takes ~1–2 min on 5090.
- Data lives in `./data/` (uploads, outputs, `jobs.db`).
- Env: `PIANO_DEVICE=cuda|cpu`, `PIANO_DATA_DIR=/data`.
