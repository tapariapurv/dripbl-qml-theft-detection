"""Run every test suite and print a summary.

    .venv/bin/python3 tests/run_all.py            # everything that needs no datasets (~10 min)
    .venv/bin/python3 tests/run_all.py --data     # + model accuracy on the real datasets
    .venv/bin/python3 tests/run_all.py --soak     # + 30-minute soak (leaks / stalls)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ("smoke: pipeline maths", ["test_pipeline.py"]),
    ("model self-check", ["theft_detector.py"]),
    ("API contract", ["tests/test_api.py"]),
    ("robustness: frames, camera drop, config, DB, retention, API fuzz", ["tests/test_robustness.py"]),
    ("node sabotage (live process)", ["tests/test_node_resilience.py"]),
    ("dashboard: every page x 3 states", ["tests/test_dashboard.py"]),
    ("load: 500 live-video viewers", ["tests/test_load.py"]),
]
DATA = [("model accuracy on held-out data", ["tests/test_pipeline_e2e.py"])]
SOAK = [("30-minute soak", ["tests/soak.py", "--minutes", "30"])]


def main() -> None:
    suites = SUITES + (DATA if "--data" in sys.argv else []) + (SOAK if "--soak" in sys.argv else [])
    results = []
    for name, args in suites:
        t0 = time.time()
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        ok = proc.returncode == 0
        results.append((name, ok, time.time() - t0))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            print((proc.stdout + proc.stderr)[-3000:])
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} suites passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
