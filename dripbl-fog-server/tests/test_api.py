"""API check: one fake trial through list -> detail -> review -> config round trip,
and writes without the key are refused. Temp DB/config, no node or network needed.

    .venv/bin/python3 tests/test_api.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
(tmp / "runtime_config.json").write_text(json.dumps({"data_dir": str(tmp / "data")}))
os.environ.update(DRIPBL_CONFIG=str(tmp / "runtime_config.json"),
                  DRIPBL_APP_SETTINGS=str(tmp / "app_settings.json"), DRIPBL_API_KEY="test-key")
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import store  # noqa: E402
from api.server import DB, app  # noqa: E402

KEY = {"X-API-Key": "test-key"}


def fake_payload(trial_id: str, flagged: bool) -> dict:
    return {"trial_id": trial_id, "timestamp": time.time(), "features": [0.1, -0.2, 0.3, 0.05],
            "quantum": {"object_class": "theft", "margin": 0.4, "trained": True,
                        "class_scores": {"suspicious_signature": 0.8}},
            "verdict": {"suspicion_score": 0.74, "threshold": 0.5, "flagged": flagged,
                        "gait": {"speed_px_s": 90.0, "stride_freq_hz": 1.2, "jitter_score": 0.3},
                        "signals": {"crime_signal": 0.8}, "breakdown": {"crime_signal": 0.64}}}


with TestClient(app) as client:
    store.insert_trial(DB, fake_payload("t-flag", True), source="unit", ground_truth="theft")
    store.insert_trial(DB, fake_payload("t-calm", False), source="unit", ground_truth="normal")

    assert client.get("/api/health").json()["ok"]
    rows = client.get("/api/trials?flagged=true").json()
    assert [r["id"] for r in rows] == ["t-flag"] and rows[0]["review_status"] == "pending"
    assert client.get("/api/trials/t-flag").json()["payload"]["verdict"]["breakdown"]["crime_signal"] == 0.64
    assert client.get("/api/trials/nope").status_code == 404

    # writes need the key
    assert client.patch("/api/trials/t-flag", json={"status": "confirmed"}).status_code == 401
    assert client.put("/api/config", json={"sensitivity": 0.7}).status_code == 401
    r = client.patch("/api/trials/t-flag", json={"status": "confirmed", "note": "on camera"}, headers=KEY)
    assert r.json()["review_status"] == "confirmed" and len(r.json()["review_history"]) == 1
    assert client.patch("/api/trials/t-flag", json={"status": "bogus"}, headers=KEY).status_code == 422

    # node config: validated, persisted, derived thresholds follow sensitivity
    cfg = client.put("/api/config", json={"sensitivity": 0.7, "roi": [10, 20, 300, 200]}, headers=KEY).json()
    assert cfg["sensitivity"] == 0.7 and cfg["roi"] == [10, 20, 300, 200]
    assert abs(cfg["derived"]["flag_threshold"] - 0.3) < 1e-9
    assert client.put("/api/config", json={"sensitivity": 3}, headers=KEY).status_code == 422
    assert client.put("/api/config", json={"no_such": 1}, headers=KEY).status_code == 422

    # camera passwords are masked on read and survive a round trip of the masked value
    client.put("/api/config", json={"camera_source": "rtsp://admin:hunter2@10.0.0.5/s1"}, headers=KEY)
    shown = client.get("/api/config").json()["camera_source"]
    assert shown == "rtsp://admin:********@10.0.0.5/s1", shown
    client.put("/api/config", json={"camera_source": shown, "sensitivity": 0.6}, headers=KEY)
    import config as _c
    assert _c.ConfigStore().settings.camera_source == "rtsp://admin:hunter2@10.0.0.5/s1"

    # app settings: nested partial update, secrets masked on read
    s = client.put("/api/app-settings", json={"dashboard": {"refresh_sec": 5},
                                              "ai_judge": {"api_key": "sk-secret"}}, headers=KEY).json()
    assert s["dashboard"]["refresh_sec"] == 5.0 and s["ai_judge"]["api_key"] == "********"
    assert "sk-secret" not in client.get("/api/app-settings").text
    assert client.put("/api/app-settings", json={"dashboard": {"refresh_sec": "x"}}, headers=KEY).status_code == 422

    st = client.get("/api/stats").json()
    assert st["n"] == 2 and st["mock_confusion"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert client.get("/api/trials.csv").text.startswith("id,ts,flagged")
    assert client.get("/api/research/metrics").json()["clip"]["accuracy"] > 0.8

print("ok")
