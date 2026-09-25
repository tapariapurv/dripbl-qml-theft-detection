"""Load: many open live-video viewers must not starve the rest of the API.

    .venv/bin/python3 tests/test_load.py [viewers=500]
"""
import json, os, socket, sys, tempfile, threading, time
from pathlib import Path
import numpy as np, cv2, requests
ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
(tmp / "rc.json").write_text(json.dumps({"data_dir": str(tmp / "data")}))
(tmp / "data").mkdir()
cv2.imwrite(str(tmp / "data" / "live.jpg"), np.zeros((480, 640, 3), np.uint8))
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
os.environ.update(DRIPBL_CONFIG=str(tmp / "rc.json"), DRIPBL_APP_SETTINGS=str(tmp / "a.json"), DRIPBL_API_KEY="k")
sys.path.insert(0, str(ROOT))
import uvicorn
from api.server import app
srv = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
threading.Thread(target=srv.run, daemon=True).start()
while not srv.started: time.sleep(0.1)
stop = threading.Event()
def touch():  # the node publishing frames
    while not stop.is_set():
        os.utime(tmp / "data" / "live.jpg", None); time.sleep(0.1)
threading.Thread(target=touch, daemon=True).start()
def viewer():
    try:
        with requests.get(f"http://127.0.0.1:{port}/api/stream", stream=True, timeout=30) as r:
            for _ in r.iter_content(4096):
                if stop.is_set(): return
    except Exception: pass
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
for _ in range(N): threading.Thread(target=viewer, daemon=True).start()
time.sleep(3)
t = time.time()
try:
    r = requests.get(f"http://127.0.0.1:{port}/api/trials", timeout=10)
    took = time.time() - t
finally:
    stop.set()
assert r.status_code == 200 and took < 1.0, f"API slow/failing with {N} viewers: {r.status_code} in {took:.2f}s"
print(f"ok: {N} open live-video viewers, /api/trials answered in {took * 1000:.0f} ms")
