"""Soak test: run node + API + dashboard for a long time and watch for leaks and stalls.

Isolated (temp config/data dir, its own ports). While it runs, a client keeps an
MJPEG viewer attached and polls the API like an open dashboard would. Every 30 s it
records each process's memory (RSS) and the node's heartbeat. It fails on: a process
exiting, heartbeats stopping, API errors, or memory still growing in the last third.

    .venv/bin/python3 tests/soak.py --minutes 30
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--api-port", type=int, default=8030)
    ap.add_argument("--dash-port", type=int, default=8630)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="dripbl_soak_"))
    cfg = tmp / "runtime_config.json"
    cfg.write_text(json.dumps({"mode": "mock", "mock_dir": str(ROOT.parent / "training-data" / "dataset-1"),
                               "data_dir": str(tmp / "data"), "log_path": str(tmp / "trials.log"),
                               "mock_clip_sec": 8.0, "sensitivity": 0.6, "debounce_sec": 2.0}))
    env = {**os.environ, "DRIPBL_CONFIG": str(cfg), "DRIPBL_APP_SETTINGS": str(tmp / "app.json"),
           "DRIPBL_API_KEY": "soak", "DRIPBL_API_URL": f"http://127.0.0.1:{args.api_port}", "PYTHONUNBUFFERED": "1"}
    logs = {n: open(tmp / f"{n}.log", "w") for n in ("node", "api", "dash")}
    procs = {
        "node": subprocess.Popen([PY, "-u", "main.py"], cwd=ROOT, env=env, stdout=logs["node"], stderr=subprocess.STDOUT),
        "api": subprocess.Popen([PY, "-m", "uvicorn", "api.server:app", "--port", str(args.api_port), "--log-level",
                                 "warning"], cwd=ROOT, env=env, stdout=logs["api"], stderr=subprocess.STDOUT),
        "dash": subprocess.Popen([PY, "-m", "streamlit", "run", "dashboard/app.py", "--server.port",
                                  str(args.dash_port), "--server.headless", "true"], cwd=ROOT, env=env,
                                 stdout=logs["dash"], stderr=subprocess.STDOUT),
    }
    api = f"http://127.0.0.1:{args.api_port}"
    stop = threading.Event()
    stats = {"frames": 0, "polls": 0, "errors": []}

    def viewer() -> None:  # an open browser tab watching the live stream
        while not stop.is_set():
            try:
                with requests.get(f"{api}/api/stream?fps=10", stream=True, timeout=10) as r:
                    for chunk in r.iter_content(65536):
                        stats["frames"] += chunk.count(b"--frame")
                        if stop.is_set():
                            return
            except requests.RequestException:
                time.sleep(1)

    def poller() -> None:  # an open dashboard polling every 2 s
        while not stop.is_set():
            for path in ("/api/status", "/api/stats", "/api/trials?limit=50", "/api/trials?flagged=true&full=true&limit=6"):
                try:
                    r = requests.get(api + path, timeout=10)
                    if r.status_code >= 500:
                        stats["errors"].append(f"{path} -> {r.status_code}")
                    stats["polls"] += 1
                except requests.RequestException as exc:
                    if time.time() - t_start > 60:  # ignore the API's own start-up
                        stats["errors"].append(f"{path} -> {type(exc).__name__}")
            stop.wait(2)

    t_start = time.time()
    samples = []
    try:
        for _ in range(240):  # wait for the node's first heartbeat (cold imports)
            try:
                if requests.get(f"{api}/api/status", timeout=2).json().get("node_online"):
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(1)
        threads = [threading.Thread(target=viewer, daemon=True), threading.Thread(target=poller, daemon=True)]
        for t in threads:
            t.start()
        requests.get(f"http://127.0.0.1:{args.dash_port}", timeout=10)
        end = time.time() + args.minutes * 60
        while time.time() < end:
            for name, p in procs.items():
                assert p.poll() is None, f"{name} exited with {p.returncode}"
            st = requests.get(f"{api}/api/status", timeout=10).json()
            hb = st["node"] or {}
            samples.append({"t": round((time.time() - t_start) / 60, 1), **{n: round(rss_mb(p.pid), 1) for n, p in procs.items()},
                            "online": st["node_online"], "fps": round(hb.get("fps", 0), 1), "trials": hb.get("trials"),
                            "errors": hb.get("errors"), "stream_frames": stats["frames"], "polls": stats["polls"]})
            print(json.dumps(samples[-1]), flush=True)
            assert st["node_online"], "node heartbeat stopped"
            time.sleep(30)
    finally:
        stop.set()
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs.values():
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in logs.values():
            f.close()

    third = max(1, len(samples) // 3)
    report = {"minutes": args.minutes, "samples": len(samples), "api_errors": stats["errors"][:20],
              "node_errors": samples[-1]["errors"], "stream_frames": stats["frames"], "trials": samples[-1]["trials"]}
    for n in procs:
        mid = [s[n] for s in samples[third:2 * third]] or [samples[0][n]]
        last = [s[n] for s in samples[-third:]]
        report[f"{n}_rss_mb"] = {"start": samples[0][n], "end": samples[-1][n], "growth_last_third_mb":
                                 round(max(last) - max(mid), 1)}
    (ROOT / "reports" / "soak_report.json").write_text(json.dumps({"report": report, "samples": samples}, indent=2))
    print(json.dumps(report, indent=2))
    tb = [n for n in logs if "Traceback" in (tmp / f"{n}.log").read_text()]
    assert not tb, f"tracebacks in {tb} logs: see {tmp}"
    assert not stats["errors"], stats["errors"][:5]
    assert samples[-1]["errors"] == 0, "node logged errors"
    for n in procs:
        assert report[f"{n}_rss_mb"]["growth_last_third_mb"] < 60, f"{n} memory still growing: {report[f'{n}_rss_mb']}"
    shutil.rmtree(tmp, ignore_errors=True)
    print("SOAK PASSED")


if __name__ == "__main__":
    main()
