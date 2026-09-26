"""Render every dashboard page headlessly (Streamlit AppTest) against a real API
server in three states: empty database, populated database, and API down.
A page passes when it renders without an exception.

    .venv/bin/python3 tests/test_dashboard.py
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
(tmp / "runtime_config.json").write_text(json.dumps({"data_dir": str(tmp / "data")}))
with socket.socket() as sck:
    sck.bind(("127.0.0.1", 0))
    PORT = sck.getsockname()[1]
os.environ.update(DRIPBL_CONFIG=str(tmp / "runtime_config.json"), DRIPBL_APP_SETTINGS=str(tmp / "app.json"),
                  DRIPBL_API_KEY="k", DRIPBL_API_URL=f"http://127.0.0.1:{PORT}")
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import store  # noqa: E402
from api.server import DB, app  # noqa: E402

PAGES = ["views/home.py", "views/alerts.py", "views/inspector.py", "views/research.py",
         "views/quantum.py", "views/settings.py"]


def start_api() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError("API did not start")


# Content each page must show once data exists: catches a section silently disappearing
# (a render with no exception is not enough: a deleted tab body renders "fine").
EXPECT = {
    "views/research.py": {"Clip accuracy", "UCF-Crime Test", "Logical qubits"},
    "views/settings.py": {"Sensitivity"},
    "views/inspector.py": {"Crime signal"},
}


def render_all(state: str) -> None:
    for page in PAGES:
        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60)
        at.run()
        assert not at.exception, (state, "app", [e.message for e in at.exception])
        at.switch_page(page)
        at.run()
        assert not at.exception, (state, page, [str(e.message)[:300] for e in at.exception])
        if state == "populated" and page in EXPECT:
            labels = {m.label for m in at.metric} | {w.label for w in at.slider}
            missing = EXPECT[page] - labels
            assert not missing, (page, "missing content", missing)
        print(f"ok: {state:10s} {page:22s} ({len(at.markdown)} markdown, {len(at.error)} errors shown)")


def seed(n: int) -> None:
    for i in range(n):
        flagged = i % 4 == 0
        store.insert_trial(DB, {
            "trial_id": f"t{i}", "timestamp": time.time() - i * 30, "features": [0.01 * i, -0.02, 0.03, 0.0],
            "quantum": {"object_class": "theft" if flagged else "normal", "margin": 0.5 if flagged else -1.0,
                        "trained": True, "class_scores": {"suspicious_signature": 0.7 if flagged else 0.1},
                        "member_margins": [0.4, 0.6, 0.5, 0.3, 0.7], "latent": [0.1] * 128, "backend": "trained_qsvm"},
            "verdict": {"suspicion_score": 0.8 if flagged else 0.2, "threshold": 0.5, "flagged": flagged,
                        "gait": {"speed_px_s": 50.0, "stride_freq_hz": 1.0, "jitter_score": 0.2},
                        "signals": {"crime_signal": 0.7, "nervousness": 0.2, "speed_anomaly": 0.1},
                        "breakdown": {"crime_signal": 0.56, "nervousness": 0.03, "speed_anomaly": 0.01}},
            "stage": {"roi": [1, 2, 3, 4], "area_threshold": 500, "sensitivity": 0.5, "latency_ms": 12,
                      "model": "test", "mode": "mock", "source": "<script>alert(1)</script>"},
        }, source="<b>x</b>", ground_truth="theft" if i % 3 == 0 else "normal")
    store.put_heartbeat(DB, {"started": time.time(), "mode": "mock", "source": "clip.mp4", "source_kind": "offline",
                             "model": "test", "trained_model": True, "fps": 20, "last_infer_ms": 10, "trials": n,
                             "flagged": n // 4, "motion_triggered": False, "gait": {}, "sensitivity": 0.5,
                             "flag_threshold": 0.5, "area_threshold": 500, "roi": [1, 2, 3, 4],
                             "errors": 0, "config_problems": ["sensitivity must be a number"]})


if __name__ == "__main__":
    server = start_api()
    render_all("empty")
    seed(60)
    render_all("populated")
    server.should_exit = True
    time.sleep(1.5)
    render_all("api-down")
    print("ALL DASHBOARD PAGES RENDER")
