"""Hostile-input checks: odd camera frames, broken configs, concurrent DB access,
and junk sent to every API endpoint (nothing may produce a 500 or kill a process).

    .venv/bin/python3 tests/test_robustness.py
"""
import json
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
(tmp / "runtime_config.json").write_text(json.dumps({"data_dir": str(tmp / "data"), "log_path": str(tmp / "t.log")}))
os.environ.update(DRIPBL_CONFIG=str(tmp / "runtime_config.json"),
                  DRIPBL_APP_SETTINGS=str(tmp / "app_settings.json"), DRIPBL_API_KEY="k")
sys.path.insert(0, str(ROOT))

import store  # noqa: E402
from camera import to_bgr8  # noqa: E402
from config import ConfigStore, Settings, apply_changes  # noqa: E402
from edge_pipeline import EdgePipeline  # noqa: E402


def test_frames() -> None:
    """Every frame shape/dtype a camera can produce flows through the pipeline."""
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320), dtype=np.uint8),           # grayscale
              rng.integers(0, 255, (240, 320, 1), dtype=np.uint8),
              rng.integers(0, 255, (240, 320, 4), dtype=np.uint8),        # BGRA
              rng.integers(0, 65535, (240, 320, 3), dtype=np.uint16),     # 16-bit sensor
              rng.random((240, 320, 3), dtype=np.float32),                # float frames
              rng.integers(0, 255, (2160, 3840, 3), dtype=np.uint8),      # 4K
              rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)]          # absurdly small
    pipe = EdgePipeline(Settings(roi=(5000, 5000, 100, 100)))              # ROI entirely off-frame
    from theft_detector import TheftDetector
    det = TheftDetector(Settings())
    for f in frames:
        bgr = to_bgr8(f)
        assert bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, f.shape
        triggered, gait = pipe.observe(bgr)
        assert set(gait) == {"speed_px_s", "stride_freq_hz", "jitter_score"}
        out = det.classify(bgr)
        assert 0.0 <= out["class_scores"]["suspicious_signature"] <= 1.0
        assert all(np.isfinite(out["features"]))
    print("ok: frames")


def test_camera_reconnect() -> None:
    """An IP camera (MJPEG over HTTP) drops and comes back: NO SIGNAL while down
    (identical frames -> zero motion -> no trials), then automatic reconnection."""
    import http.server
    import socketserver
    import cv2
    from camera import FrameGrabber

    power = threading.Event()  # the camera's power switch

    class Cam(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                for i in range(100000):
                    if not power.is_set():
                        break
                    img = np.full((240, 320, 3), 40, np.uint8)
                    cv2.circle(img, (40 + i % 240, 120), 20, (0, 200, 255), -1)
                    jpg = cv2.imencode(".jpg", img)[1].tobytes()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    def serve():
        power.set()
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 8799), Cam)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def wait_kind(g, kinds, timeout):
        t = time.time()
        while g.kind not in kinds and time.time() - t < timeout:
            time.sleep(0.1)
        return g.kind

    srv = serve()
    g = FrameGrabber(Settings(camera_source="http://127.0.0.1:8799/video.mjpg"))
    assert wait_kind(g, ("stream",), 15) == "stream", g.kind
    time.sleep(0.5)
    live = g.read()
    assert live is not None and live.shape == (240, 320, 3)
    power.clear()
    srv.shutdown()
    srv.server_close()
    assert wait_kind(g, ("offline",), 20) == "offline", f"drop not detected: {g.kind}"
    time.sleep(1.2)
    a, b2 = g.read(), None
    time.sleep(1.2)
    b2 = g.read()
    assert a is not None and b2 is not None and np.array_equal(a, b2), "offline frames must be static"
    srv = serve()
    assert wait_kind(g, ("stream",), 40) == "stream", f"did not reconnect: {g.kind}"
    g.release()
    srv.shutdown()
    srv.server_close()
    print("ok: camera drop -> NO SIGNAL -> automatic reconnect")


def test_config() -> None:
    path = tmp / "cfg_test.json"
    # broken JSON at startup -> defaults, reason recorded, file left alone for a human
    path.write_text("{ not json")
    c = ConfigStore(path)
    assert c.settings == Settings() and c.problems and path.read_text() == "{ not json"
    # bad values -> those keys keep defaults, good keys apply
    path.write_text(json.dumps({"sensitivity": "high", "roi": [1, 2], "debounce_sec": 5, "zzz": 1,
                                "mode": "party", "live_jpeg_quality": 1000, "camera_source": 2}))
    c = ConfigStore(path)
    assert c.settings.sensitivity == 0.5 and c.settings.roi == Settings().roi and c.settings.mode == "live"
    assert c.settings.debounce_sec == 5.0 and c.settings.camera_source == 2 and len(c.problems) == 5
    # hot reload: a broken edit keeps the running settings
    good = c.settings
    time.sleep(0.02)
    path.write_text("[1, 2")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert c.reload_if_changed() is None and c.settings == good and "JSON" in c.problems[0]
    path.write_text(json.dumps({"sensitivity": 0.9}))
    os.utime(path, (time.time() + 10, time.time() + 10))
    reloaded = c.reload_if_changed()
    assert reloaded.sensitivity == 0.9 and not c.problems
    assert reloaded.debounce_sec == Settings().debounce_sec, "a key removed from the file must revert to default"
    # strict mode (API): every bad input rejected, including NaN, bools-as-numbers and floats-as-ints
    for bad in ({"sensitivity": float("nan")}, {"sensitivity": True}, {"mog2_history": 3.5},
                {"roi": [0, 0, 0, 10]}, {"roi": [-1, 0, 10, 10]}, {"camera_source": None},
                {"camera_source": [1]}, {"nope": 1}, {"frame_width": 10**9}):
        try:
            apply_changes(Settings(), bad, strict=True)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert apply_changes(Settings(), {"mog2_history": 300.0}, strict=True)[0].mog2_history == 300
    print("ok: config")


def test_store_concurrency() -> None:
    """Node writing while several API readers hammer the DB: no 'database is locked'."""
    db = tmp / "conc.db"
    errors: list[BaseException] = []
    stop = threading.Event()

    def payload(i: int) -> dict:
        return {"trial_id": f"t{i}", "timestamp": time.time(), "features": [0, 0, 0, 0],
                "quantum": {"object_class": "normal", "margin": 0.1},
                "verdict": {"suspicion_score": random.random(), "threshold": 0.5, "flagged": i % 7 == 0,
                            "gait": {"speed_px_s": 1.0}, "signals": {"crime_signal": 0.2}}}

    def writer() -> None:
        try:
            for i in range(1500):
                store.insert_trial(db, payload(i), source="x")
                if i % 50 == 0:
                    store.put_heartbeat(db, {"fps": 20})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                store.list_trials(db, flagged=True, limit=50)
                store.stats(db, 0)
                rows = store.list_trials(db, limit=1, full=True)
                if rows:
                    store.review_trial(db, rows[0]["id"], "confirmed", "n")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    store.connect(db).close()
    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert store.stats(db, 0)["n"] == 1500
    print("ok: store concurrency (1500 writes vs 4 readers)")


def test_api_fuzz() -> None:
    from fastapi.testclient import TestClient
    from api.server import DB, app
    rng = random.Random(1)
    junk = [None, 0, -1, 1e308, "", "x" * 10000, "../../etc/passwd", "'; DROP TABLE trials;--",
            [], {}, [1, 2, 3], {"a": {"b": {"c": [None]}}}, True, "🔥", "\x00"]
    with TestClient(app, raise_server_exceptions=False) as client:
        store.insert_trial(DB, {"trial_id": "real", "timestamp": time.time(), "features": [0.1, 0.2, 0.3, 0.4],
                                "quantum": {"object_class": "theft", "margin": 1.0},
                                "verdict": {"suspicion_score": 0.9, "threshold": 0.5, "flagged": True,
                                            "gait": {}, "signals": {}}}, source="fuzz")
        H = {"X-API-Key": "k"}
        calls = 0
        for path in ("/api/trials", "/api/trials.csv", "/api/stats", "/api/research/x", "/api/quantum-check/zz"):
            for v in junk[:10]:
                for key in ("limit", "offset", "since", "flagged", "review_status", "full", "ground_truth"):
                    r = client.get(path, params={key: v})
                    calls += 1
                    assert r.status_code < 500, (path, key, v, r.status_code, r.text[:200])
        for name in ("../../runtime_config", "..%2F..%2Fmain.py", "metrics", "experiments"):
            r = client.get(f"/api/research/{name}")
            assert r.status_code in (200, 404), (name, r.status_code)
            assert "import" not in r.text[:200]
        for tid in ("real", "nope", "../x", "x" * 5000, "%00"):
            for method, body in (("GET", None), ("PATCH", {"status": "confirmed"}), ("PATCH", {"status": 5})):
                r = client.request(method, f"/api/trials/{tid}", json=body, headers=H)
                calls += 1
                assert r.status_code < 500, (tid, method, r.status_code)
            assert client.get(f"/api/trials/{tid}/snapshot.jpg").status_code in (404, 200)
        for _ in range(300):  # random nested junk into the config writers
            body = {rng.choice(["sensitivity", "roi", "mode", "camera_source", "x", "debounce_sec", "w_crime"]):
                    rng.choice(junk)}
            for path in ("/api/config", "/api/app-settings"):
                r = client.put(path, json=body, headers=H)
                calls += 1
                assert r.status_code < 500, (path, body, r.status_code, r.text[:200])
            r = client.put("/api/app-settings", json={rng.choice(["dashboard", "api", "notifications", "z"]):
                                                      {rng.choice(["title", "refresh_sec", "tiles", "q"]): rng.choice(junk)}},
                           headers=H)
            assert r.status_code < 500, r.text[:200]
        for body in ([], "x", None, {"backend": "nope"}, {"n_trials": 999}, {"shots": -5}, {"trial_ids": ["nope"]}):
            r = client.post("/api/quantum-check", json=body, headers=H)
            assert r.status_code < 500, (body, r.status_code, r.text[:200])
        for raw in (b"\xff\xfe not json", b'{"sensitivity": NaN}', b'{"sensitivity": Infinity}',
                    b'{"roi": [0, 0, 1e999, 5]}', b'[' * 5000):
            for path in ("/api/config", "/api/app-settings"):
                r = client.put(path, content=raw, headers={**H, "content-type": "application/json"})
                assert r.status_code in (400, 422), (path, raw[:30], r.status_code, r.text[:200])
        # config stays valid after all that
        cfg = client.get("/api/config").json()
        assert 0.0 <= cfg["sensitivity"] <= 1.0 and len(cfg["roi"]) == 4
        assert client.get("/api/app-settings").status_code == 200
    print(f"ok: api fuzz ({calls}+ junk requests, no 5xx)")


if __name__ == "__main__":
    test_frames()
    test_camera_reconnect()
    test_config()
    test_store_concurrency()
    test_api_fuzz()
    print("ALL ROBUSTNESS TESTS PASSED")
