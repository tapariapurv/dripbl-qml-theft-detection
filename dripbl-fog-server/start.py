"""One command for the whole system: fog node + API server + dashboard.

    .venv/bin/python3 start.py                 # live mode, this machine only
    .venv/bin/python3 start.py --mock          # research mode: replay labelled dataset-1 clips
    .venv/bin/python3 start.py --host 0.0.0.0  # fog server: reachable from other devices on the network

Ctrl-C stops all three. Each line of output is prefixed with the process it came from.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
COLOURS = {"node": "\033[35m", "api": "\033[36m", "dash": "\033[32m"}


def _pump(name: str, proc: subprocess.Popen) -> None:
    for line in iter(proc.stdout.readline, ""):
        print(f"{COLOURS[name]}[{name:>4}]\033[0m {line}", end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="switch the node to mock (research) mode")
    ap.add_argument("--live", action="store_true", help="switch the node to live mode")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for API + dashboard (0.0.0.0 = network)")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--dash-port", type=int, default=8501)
    ap.add_argument("--no-node", action="store_true", help="API + dashboard only (node runs elsewhere/separately)")
    args = ap.parse_args()

    for port, flag in ((args.api_port, "--api-port"), (args.dash_port, "--dash-port")):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already in use by another program; pick another with {flag}")

    sys.path.insert(0, str(ROOT))
    from config import ConfigStore
    cfg = ConfigStore()
    if args.mock:
        # mock mode replays retail footage (dataset-1 hold-out), so run the retail expert on it
        cfg.save(dataclasses.replace(cfg.settings, mode="mock", site_profile="retail"))
        print("mock mode: site profile set to 'retail' (the clips are retail footage)")
    elif args.live:
        cfg.save(dataclasses.replace(cfg.settings, mode="live"))

    # Create the API key before anything starts, so the dashboard and API agree on it.
    data = ROOT / cfg.settings.data_dir
    data.mkdir(parents=True, exist_ok=True)
    key_file = data / "api_key"
    if not os.environ.get("DRIPBL_API_KEY") and not key_file.is_file():
        key_file.write_text(secrets.token_urlsafe(24))
        os.chmod(key_file, 0o600)
    api_key = os.environ.get("DRIPBL_API_KEY") or key_file.read_text().strip()

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "DRIPBL_API_KEY": api_key,
           "DRIPBL_API_URL": f"http://127.0.0.1:{args.api_port}"}
    cmds = {
        "api": [PY, "-m", "uvicorn", "api.server:app", "--host", args.host, "--port", str(args.api_port),
                "--log-level", "warning"],
        "dash": [PY, "-m", "streamlit", "run", "dashboard/app.py", "--server.address", args.host,
                 "--server.port", str(args.dash_port)],
    }
    if not args.no_node:
        cmds = {"node": [PY, "-u", "main.py"], **cmds}

    procs: dict[str, subprocess.Popen] = {}
    for name, cmd in cmds.items():
        procs[name] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
        threading.Thread(target=_pump, args=(name, procs[name]), daemon=True).start()

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"\n  Dashboard  http://{shown}:{args.dash_port}\n  API docs   http://{shown}:{args.api_port}/docs\n"
          f"  Mode       {cfg.settings.mode if not (args.mock or args.live) else ('mock' if args.mock else 'live')}"
          f"\n  Ctrl-C to stop everything\n", flush=True)

    try:
        while True:
            for name, p in procs.items():
                if p.poll() is not None:
                    raise SystemExit(f"[{name}] exited with code {p.returncode}; stopping the rest")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()  # SIGTERM: every child handles it as a clean shutdown
        deadline = time.time() + 8
        for p in procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        print("stopped.")


def _sigterm(*_: object) -> None:
    raise KeyboardInterrupt  # `kill` / systemd: stop the children too, same as Ctrl-C


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    main()
