"""DRIPBL REST API: the single data surface for the dashboard and for anything
customers build on top (interactive docs at /docs, OpenAPI at /openapi.json).

    .venv/bin/python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8000

Reads what the fog node publishes (data/dripbl.db, data/live.jpg, snapshots),
writes node settings to runtime_config.json (the node hot-reloads it within ~1s),
and owns app_settings.json (dashboard, notifications, AI judge, quantum, API).

Auth: every write needs header `X-API-Key` (or `?key=` for <img> tags). The key
comes from $DRIPBL_API_KEY, else data/api_key (generated on first start, 0600).
"""
from __future__ import annotations

import asyncio
import csv
import dataclasses
import hmac
import io
import json
import os
import re
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import store  # noqa: E402
from api import app_settings, quantum_check  # noqa: E402
from api.worker import Worker, send_notifications  # noqa: E402
from config import ConfigStore, Settings, apply_changes  # noqa: E402

CONFIG_PATH = Path(os.environ.get("DRIPBL_CONFIG", ROOT / "runtime_config.json"))
NODE_OFFLINE_AFTER_SEC = 5.0
REPORTS = {  # name -> file under reports/ (whitelist: nothing else is readable)
    "metrics": "metrics.json",
    "experiments": "experiments.jsonl",
    "hardware": "hardware_report.json",
    "shoplifting": "shoplifting_metrics.json",
    "temporal_ab": "temporal_ab_metrics.json",
    "dataset2": "combined_d2_metrics.json",
    "deployed": "deployed_default.json",        # written by src/deploy_model.py
    "deployed_retail": "deployed_retail.json",
    "cross_dataset": "xd_experiments.jsonl",    # every cross-dataset experiment
}


def _paths() -> tuple[Path, Path, Path]:
    data = ROOT / ConfigStore(CONFIG_PATH).settings.data_dir
    (data / "snapshots").mkdir(parents=True, exist_ok=True)
    return data, data / "dripbl.db", data / "snapshots"


DATA_DIR, DB, SNAPS = _paths()


def _api_key() -> str:
    if os.environ.get("DRIPBL_API_KEY"):
        return os.environ["DRIPBL_API_KEY"]
    path = DATA_DIR / "api_key"
    if not path.is_file():
        path.write_text(secrets.token_urlsafe(24))
        os.chmod(path, 0o600)
    return path.read_text().strip()


API_KEY = _api_key()


def _key_ok(request: Request) -> bool:
    given = request.headers.get("x-api-key") or request.query_params.get("key") or ""
    return hmac.compare_digest(given, API_KEY)


def require_write(request: Request) -> None:
    if not _key_ok(request):
        raise HTTPException(401, "missing or invalid X-API-Key")


def require_read(request: Request) -> None:
    if app_settings.load()["api"]["require_key_for_reads"] and not _key_ok(request):
        raise HTTPException(401, "missing or invalid X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.connect(DB).close()  # create schema before the node's first write
    app.state.worker = Worker(DB, SNAPS, DATA_DIR / "worker_state.json").start()
    yield
    app.state.worker.stop()


app = FastAPI(
    title="DRIPBL Fog Node API",
    version="1.0",
    description="Live trials, reviews, video, configuration and research data from the DRIPBL "
                "quantum-kernel theft-detection fog node. Writes require the `X-API-Key` header.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=app_settings.load()["api"]["cors_origins"],
                   allow_methods=["*"], allow_headers=["*"])

read = [Depends(require_read)]
write = [Depends(require_write)]


# --- status -------------------------------------------------------------------

@app.get("/api/health", tags=["status"])
def health() -> dict[str, Any]:
    """Liveness of the API itself (never needs a key)."""
    return {"ok": True, "time": time.time()}


@app.get("/api/status", tags=["status"], dependencies=read)
def status() -> dict[str, Any]:
    """Node heartbeat (fps, latency, model, mode, current gait) plus worker health."""
    hb = store.get_heartbeat(DB)
    online = hb is not None and time.time() - hb["ts"] < NODE_OFFLINE_AFTER_SEC
    worker = app.state.worker
    return {"node_online": online, "node": hb, "api_time": time.time(),
            "worker": {"last_error": worker.last_error, "last_notification": worker.last_results}}


@app.get("/api/stats", tags=["status"], dependencies=read)
def stats(since: float = Query(0.0, description="unix time; 0 = all time")) -> dict[str, Any]:
    """Counts for dashboard tiles and the mock-mode confusion matrix (per trial)."""
    return store.stats(DB, since)


# --- trials -------------------------------------------------------------------

@app.get("/api/trials", tags=["trials"], dependencies=read)
def trials(
    flagged: bool | None = None,
    review_status: Literal["pending", "confirmed", "false_alarm", "dismissed"] | None = None,
    ground_truth: Literal["normal", "theft"] | None = None,
    since: float | None = Query(None, description="only trials after this unix time (poll with the newest ts)"),
    until: float | None = None,
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    full: bool = Query(False, description="include the full per-stage payload"),
) -> list[dict[str, Any]]:
    """Trials newest first. Poll `?since=<newest ts>` for live updates."""
    return store.list_trials(DB, flagged=flagged, review_status=review_status, since=since, until=until,
                             ground_truth=ground_truth, limit=limit, offset=offset, full=full)


@app.get("/api/trials.csv", tags=["trials"], dependencies=read)
def trials_csv(flagged: bool | None = None, since: float | None = None, limit: int = Query(10000, le=100000)):
    rows = store.list_trials(DB, flagged=flagged, since=since, limit=limit)
    cols = ["id", "ts", "flagged", "suspicion", "crime_signal", "object_class", "margin", "source",
            "ground_truth", "review_status", "review_note", "reviewed_at"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols + ["speed_px_s", "jitter_score", "stride_freq_hz", "ai_verdict"],
                       extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({**r, **r.get("gait", {}), "ai_verdict": (r.get("ai_review") or {}).get("verdict")})
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=dripbl_trials.csv"})


@app.get("/api/trials/{trial_id}", tags=["trials"], dependencies=read)
def trial(trial_id: str) -> dict[str, Any]:
    """One trial with every pipeline stage (frame gate, gait, latent, angles, ensemble margins, judge)."""
    t = store.get_trial(DB, trial_id)
    if t is None:
        raise HTTPException(404, "no such trial")
    return t


@app.get("/api/trials/{trial_id}/snapshot.jpg", tags=["trials"], dependencies=read)
def snapshot(trial_id: str):
    t = store.get_trial(DB, trial_id)
    if t is None or not t.get("snapshot") or not (SNAPS / t["snapshot"]).is_file():
        raise HTTPException(404, "no snapshot (only flagged trials keep one)")
    return FileResponse(SNAPS / t["snapshot"], media_type="image/jpeg")


class Review(BaseModel):
    status: Literal["pending", "confirmed", "false_alarm", "dismissed"]
    note: str = Field("", max_length=2000)
    reviewer: str = Field("", max_length=100)


@app.patch("/api/trials/{trial_id}", tags=["trials"], dependencies=write)
def review(trial_id: str, body: Review) -> dict[str, Any]:
    """Record a human decision on a flagged trial (kept in its review history)."""
    if not store.review_trial(DB, trial_id, body.status, body.note, body.reviewer):
        raise HTTPException(404, "no such trial")
    return store.get_trial(DB, trial_id)


# --- live video -----------------------------------------------------------------

@app.get("/api/frame.jpg", tags=["video"], dependencies=read)
def frame():
    path = DATA_DIR / "live.jpg"
    if not path.is_file():
        raise HTTPException(404, "node has not published a frame yet")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/stream", tags=["video"], dependencies=read)
async def stream(fps: float = Query(10.0, gt=0, le=30)):
    """MJPEG stream of the node's annotated live frame — use directly as <img src>.
    Async: an open viewer costs no worker thread while it waits for the next frame, so
    hundreds of open dashboards cannot starve the rest of the API."""
    path, period = DATA_DIR / "live.jpg", 1.0 / fps

    async def frames() -> AsyncIterator[bytes]:
        last = 0.0
        while True:
            try:
                mtime = path.stat().st_mtime
                if mtime != last:
                    last = mtime
                    jpg = path.read_bytes()  # ~50 KB: cheaper inline than a thread hop
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                        str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            except OSError:
                pass
            await asyncio.sleep(period)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


# --- configuration --------------------------------------------------------------

_URL_PASSWORD = re.compile(r"^(\w+://[^/@:]*:)([^/@]*)(@)")


def _mask_camera(src: Any) -> Any:
    """rtsp://user:secret@host -> rtsp://user:********@host (never echo camera passwords)."""
    return _URL_PASSWORD.sub(lambda m: m.group(1) + app_settings.MASK + m.group(3), src) if isinstance(src, str) else src


def _config_dict(s: Settings) -> dict[str, Any]:
    d = dataclasses.asdict(s)
    d["camera_source"] = _mask_camera(d["camera_source"])
    d["roi"] = list(d["roi"])
    d["derived"] = {"flag_threshold": s.flag_threshold, "area_threshold": s.area_threshold}
    return d


@app.get("/api/config", tags=["config"], dependencies=read)
def get_config() -> dict[str, Any]:
    """Node settings (runtime_config.json) plus the sensitivity-derived thresholds."""
    return _config_dict(ConfigStore(CONFIG_PATH).settings)


@app.get("/api/config/defaults", tags=["config"], dependencies=read)
def config_defaults() -> dict[str, Any]:
    return _config_dict(Settings())


@app.put("/api/config", tags=["config"], dependencies=write)
def put_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Partial update of node settings; the running node applies it within ~1 s."""
    cfg = ConfigStore(CONFIG_PATH)
    src = patch.get("camera_source") if isinstance(patch, dict) else None
    if isinstance(src, str) and f":{app_settings.MASK}@" in src:  # masked password sent back: keep the stored one
        old = _URL_PASSWORD.match(str(cfg.settings.camera_source))
        if old:
            patch = {**patch, "camera_source": src.replace(app_settings.MASK, old.group(2), 1)}
    try:
        new, _ = apply_changes(cfg.settings, patch, strict=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    cfg.save(new)
    return _config_dict(new)


@app.post("/api/config/reset", tags=["config"], dependencies=write)
def reset_config() -> dict[str, Any]:
    cfg = ConfigStore(CONFIG_PATH)
    cfg.save(Settings())
    return _config_dict(cfg.settings)


@app.get("/api/app-settings", tags=["config"], dependencies=read)
def get_app_settings() -> dict[str, Any]:
    """Dashboard, notification, AI-judge, quantum, API and retention settings (secrets masked)."""
    return app_settings.masked(app_settings.load())


@app.get("/api/app-settings/defaults", tags=["config"], dependencies=read)
def app_settings_defaults() -> dict[str, Any]:
    return app_settings.DEFAULTS


@app.put("/api/app-settings", tags=["config"], dependencies=write)
def put_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Partial, nested update. Unknown keys and wrong types are rejected."""
    try:
        return app_settings.masked(app_settings.update(patch))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/app-settings/reset", tags=["config"], dependencies=write)
def reset_app_settings(section: str | None = None) -> dict[str, Any]:
    try:
        return app_settings.masked(app_settings.reset(section))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# --- research -------------------------------------------------------------------

@app.get("/api/research", tags=["research"], dependencies=read)
def research_index() -> dict[str, bool]:
    return {name: (ROOT / "reports" / f).is_file() for name, f in REPORTS.items()}


@app.get("/api/research/{name}", tags=["research"], dependencies=read)
def research(name: str) -> Any:
    """Offline results: metrics, experiments (ablation log), hardware (shot-noise study), ..."""
    if name not in REPORTS:
        raise HTTPException(404, f"unknown report; one of {sorted(REPORTS)}")
    path = ROOT / "reports" / REPORTS[name]
    if not path.is_file():
        raise HTTPException(404, f"{REPORTS[name]} has not been generated")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


# --- quantum lab ----------------------------------------------------------------

class QuantumCheck(BaseModel):
    backend: Literal["exact", "fake_brisbane", "ibm"] = "fake_brisbane"
    shots: int = Field(1024, ge=16, le=100000)
    n_trials: int = Field(6, ge=2, le=quantum_check.MAX_TRIALS)
    trial_ids: list[str] | None = Field(None, description="default: the most recent trials")


@app.post("/api/quantum-check", tags=["quantum"], dependencies=write)
def quantum_check_start(body: QuantumCheck) -> dict[str, Any]:
    """Start a background job comparing the exact kernel with circuits on a noisy/real backend."""
    if body.trial_ids:
        rows = [store.get_trial(DB, t) for t in body.trial_ids]
        rows = [r for r in rows if r is not None]
    else:
        rows = store.list_trials(DB, limit=body.n_trials, full=True)
    if len(rows) < 2:
        raise HTTPException(422, "need at least 2 recorded trials — let the node run first")
    feats = [r["payload"]["features"] for r in rows]
    try:
        job = quantum_check.start(feats, [r["id"] for r in rows], body.backend, body.shots,
                                  app_settings.load()["quantum"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"job_id": job}


@app.get("/api/quantum-check", tags=["quantum"], dependencies=read)
def quantum_check_list() -> list[dict[str, Any]]:
    return quantum_check.recent()


@app.get("/api/quantum-check/{job_id}", tags=["quantum"], dependencies=read)
def quantum_check_get(job_id: str) -> dict[str, Any]:
    job = quantum_check.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job (jobs live in memory until the API restarts)")
    return job


# --- notifications ----------------------------------------------------------------

@app.post("/api/notify/test", tags=["notifications"], dependencies=write)
def notify_test() -> dict[str, str]:
    """Send a sample alert through every enabled channel and report each result."""
    cfg = app_settings.load()["notifications"]
    latest = store.list_trials(DB, flagged=True, limit=1, full=True) or store.list_trials(DB, limit=1, full=True)
    if latest:
        sample = latest[0]
    else:
        sample = {"id": "test", "ts": time.time(), "source": "test", "suspicion": 0.9, "ai_review": None,
                  "snapshot": None, "payload": {"verdict": {"suspicion_score": 0.9, "threshold": 0.5,
                                                            "signals": {"crime_signal": 0.9},
                                                            "gait": {"speed_px_s": 120, "jitter_score": 0.4}}}}
    snap = SNAPS / sample["snapshot"] if sample.get("snapshot") else None
    results = send_notifications(sample, snap, cfg)
    if not results:
        raise HTTPException(422, "no notification channel is enabled and configured")
    return results


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=422)
