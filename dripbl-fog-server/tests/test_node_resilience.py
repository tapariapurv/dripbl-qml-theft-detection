"""Sabotage a running fog node and check it keeps going.

Starts the real main.py on a generated video (isolated config + data dir via
DRIPBL_CONFIG), then while it runs: corrupts its config, sends invalid values,
points it at a missing camera, a missing mock folder, deletes its whole data
directory, and finally SIGTERMs it. After every step the node must still be
alive and sending heartbeats; it must also exit cleanly.

    .venv/bin/python3 tests/test_node_resilience.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import store  # noqa: E402

tmp = Path(tempfile.mkdtemp())
video, cfg_path, data = tmp / "walk.mp4", tmp / "runtime_config.json", tmp / "data"
db = data / "dripbl.db"


def make_video(path: Path, seconds: int = 6) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    rng = np.random.default_rng(0)
    for i in range(30 * seconds):
        f = np.full((480, 640, 3), 60, np.uint8) + rng.integers(0, 8, (480, 640, 3), dtype=np.uint8)
        x = 150 + int(150 * np.sin(i / 9))
        cv2.rectangle(f, (x, 120 + (i % 20)), (x + 90, 330), (200, 180, 160), -1)  # a "person" pacing
        w.write(f)
    w.release()


def write_cfg(obj) -> None:
    text = obj if isinstance(obj, str) else json.dumps(obj)
    tmp_file = cfg_path.with_suffix(".w")
    tmp_file.write_text(text)
    os.replace(tmp_file, cfg_path)


BASE = {"camera_source": str(video), "data_dir": str(data), "log_path": str(tmp / "trials.log"),
        "sensitivity": 0.9, "debounce_sec": 0.5, "live_frame_fps": 5.0, "heartbeat_sec": 0.5}


def heartbeat_after(t: float, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            hb = store.get_heartbeat(db)
            if hb and hb["ts"] > t:
                return hb
        except Exception:  # noqa: BLE001 - the DB may be mid-recreation during sabotage
            pass
        time.sleep(0.3)
    raise AssertionError(f"no heartbeat within {timeout}s")


def step(name: str, proc: subprocess.Popen, action) -> dict:
    t = time.time()
    action()
    time.sleep(2.5)
    assert proc.poll() is None, f"node died during: {name}"
    hb = heartbeat_after(t)
    print(f"ok: {name}  (fps={hb['fps']:.0f} trials={hb['trials']} errors={hb.get('errors')} "
          f"problems={hb.get('config_problems')})", flush=True)
    return hb


def main() -> None:
    make_video(video)
    write_cfg(BASE)
    env = {**os.environ, "DRIPBL_CONFIG": str(cfg_path), "PYTHONUNBUFFERED": "1"}
    log = open(tmp / "node.out", "w")
    proc = subprocess.Popen([sys.executable, "-u", "main.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        hb = heartbeat_after(0, timeout=180)  # cold start imports torch/sklearn/qiskit
        print(f"ok: node up (model: {hb['model']})", flush=True)
        deadline = time.time() + 60
        while store.stats(db, 0)["n"] == 0 and time.time() < deadline:
            time.sleep(0.5)
        assert store.stats(db, 0)["n"] > 0, "no trials recorded from the moving-person video"
        print(f"ok: trials recorded ({store.stats(db, 0)['n']})", flush=True)

        hb = step("config becomes invalid JSON", proc, lambda: write_cfg("{ oops"))
        assert hb["config_problems"] and hb["sensitivity"] == 0.9
        hb = step("config with wrong types", proc,
                  lambda: write_cfg({**BASE, "sensitivity": "max", "roi": "everything", "mode": 3}))
        assert hb["sensitivity"] == 0.9 and len(hb["config_problems"]) == 3
        hb = step("valid edit applies live", proc, lambda: write_cfg({**BASE, "sensitivity": 0.3}))
        assert hb["sensitivity"] == 0.3 and not hb["config_problems"]
        step("ROI far outside the frame", proc, lambda: write_cfg({**BASE, "roi": [5000, 5000, 50, 50]}))
        step("camera that does not exist", proc,
             lambda: write_cfg({**BASE, "camera_source": str(tmp / "missing.mp4")}))
        n_off = store.stats(db, 0)["n"]
        time.sleep(4)
        hb = heartbeat_after(time.time() - 1)
        assert hb["source_kind"] == "offline", hb["source_kind"]
        assert store.stats(db, 0)["n"] == n_off, "an offline camera produced trials (fake alerts)"
        print("ok: offline camera -> NO SIGNAL, zero trials", flush=True)
        n_off = store.stats(db, 0)["n"]
        hb = step("stream URL that does not resolve", proc,
                  lambda: write_cfg({**BASE, "camera_source": "rtsp://user:pw@nonexistent.invalid/s"}))
        assert "pw" not in hb["source"], hb["source"]
        assert store.stats(db, 0)["n"] == n_off, "switching to a dead camera produced a trial"
        step("mock mode with missing clip folder", proc,
             lambda: write_cfg({**BASE, "mode": "mock", "mock_dir": str(tmp / "nope")}))
        step("back to the real video", proc, lambda: write_cfg(BASE))
        step("whole data directory deleted", proc, lambda: shutil.rmtree(data))
        n = store.stats(db, 0)["n"]
        time.sleep(4)
        assert store.stats(db, 0)["n"] >= n, "trial recording did not resume"
        assert (data / "live.jpg").exists(), "live frame not republished"
        print("ok: recording resumed after data loss", flush=True)

        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=20)
        assert code == 0, f"exit code {code}"
        print(f"ok: clean SIGTERM shutdown in {time.time() - t0:.1f}s", flush=True)
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
        out = (tmp / "node.out").read_text()
        assert "Traceback" not in out, out[-3000:]
    print("ALL NODE RESILIENCE TESTS PASSED")


if __name__ == "__main__":
    main()
