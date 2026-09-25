"""SQLite store shared by the fog node (writer) and the API server (reader/reviewer).

One file, stdlib only. WAL mode lets the node keep inserting while the API reads.
Every call opens its own short-lived connection, so it is safe from any thread
or process.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

REVIEW_STATUSES = ("pending", "confirmed", "false_alarm", "dismissed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    flagged       INTEGER NOT NULL,
    suspicion     REAL NOT NULL,
    object_class  TEXT,
    margin        REAL,
    source        TEXT,
    ground_truth  TEXT,             -- mock mode only: "normal" | "theft"
    review_status TEXT,             -- flagged trials only, see REVIEW_STATUSES
    review_note   TEXT,
    reviewed_at   REAL,
    ai_review     TEXT,             -- JSON from the optional Claude second opinion
    snapshot      TEXT,             -- file name under data/snapshots/
    payload       TEXT NOT NULL     -- full per-stage record (JSON)
);
CREATE INDEX IF NOT EXISTS trials_ts ON trials(ts);
CREATE INDEX IF NOT EXISTS trials_flagged ON trials(flagged, ts);
CREATE TABLE IF NOT EXISTS reviews (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    ts       REAL NOT NULL,
    status   TEXT NOT NULL,
    note     TEXT,
    reviewer TEXT
);
CREATE TABLE IF NOT EXISTS heartbeat (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    ts      REAL NOT NULL,
    payload TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    return conn


def insert_trial(path: str | Path, payload: dict[str, Any], *, source: str = "",
                 ground_truth: str | None = None, snapshot: str | None = None) -> None:
    v, q = payload["verdict"], payload["quantum"]
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO trials (id, ts, flagged, suspicion, object_class, margin, source, ground_truth,"
            " review_status, snapshot, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (payload["trial_id"], payload["timestamp"], int(v["flagged"]), v["suspicion_score"],
             q.get("object_class"), q.get("margin"), source, ground_truth,
             "pending" if v["flagged"] else None, snapshot, json.dumps(payload)),
        )


def put_heartbeat(path: str | Path, status: dict[str, Any]) -> None:
    with connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO heartbeat (id, ts, payload) VALUES (1, ?, ?)",
                     (time.time(), json.dumps(status)))


def get_heartbeat(path: str | Path) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT ts, payload FROM heartbeat WHERE id = 1").fetchone()
    return None if row is None else {"ts": row["ts"], **json.loads(row["payload"])}


def _row(row: sqlite3.Row, full: bool) -> dict[str, Any]:
    d = dict(row)
    d["flagged"] = bool(d["flagged"])
    d["ai_review"] = json.loads(d["ai_review"]) if d["ai_review"] else None
    payload = json.loads(d.pop("payload"))
    if full:
        d["payload"] = payload
    else:  # list view: the few fields a table needs, not the 128-d latent
        d["crime_signal"] = payload["verdict"].get("signals", {}).get("crime_signal")
        d["gait"] = payload["verdict"].get("gait", {})
    return d


def list_trials(path: str | Path, *, flagged: bool | None = None, review_status: str | None = None,
                since: float | None = None, until: float | None = None, ground_truth: str | None = None,
                limit: int = 100, offset: int = 0, full: bool = False) -> list[dict[str, Any]]:
    where, args = [], []
    for clause, val in (("flagged = ?", None if flagged is None else int(flagged)),
                        ("review_status = ?", review_status), ("ts > ?", since),
                        ("ts <= ?", until), ("ground_truth = ?", ground_truth)):
        if val is not None:
            where.append(clause)
            args.append(val)
    sql = "SELECT * FROM trials" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
    with connect(path) as conn:
        rows = conn.execute(sql, (*args, limit, offset)).fetchall()
    return [_row(r, full) for r in rows]


def get_trial(path: str | Path, trial_id: str) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,)).fetchone()
        if row is None:
            return None
        history = conn.execute("SELECT ts, status, note, reviewer FROM reviews WHERE trial_id = ? ORDER BY ts",
                               (trial_id,)).fetchall()
    d = _row(row, full=True)
    d["review_history"] = [dict(h) for h in history]
    return d


def review_trial(path: str | Path, trial_id: str, status: str, note: str = "", reviewer: str = "") -> bool:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"status must be one of {REVIEW_STATUSES}")
    now = time.time()
    with connect(path) as conn:
        cur = conn.execute("UPDATE trials SET review_status = ?, review_note = ?, reviewed_at = ? WHERE id = ?",
                           (status, note, now, trial_id))
        if cur.rowcount == 0:
            return False
        conn.execute("INSERT INTO reviews (trial_id, ts, status, note, reviewer) VALUES (?,?,?,?,?)",
                     (trial_id, now, status, note, reviewer))
    return True


def set_ai_review(path: str | Path, trial_id: str, review: dict[str, Any]) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE trials SET ai_review = ? WHERE id = ?", (json.dumps(review), trial_id))


def stats(path: str | Path, since: float) -> dict[str, Any]:
    """Counts for the dashboard tiles, plus the mock-mode confusion matrix."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(flagged),0) flagged,"
            " COALESCE(SUM(review_status = 'pending'),0) pending,"
            " COALESCE(SUM(review_status = 'confirmed'),0) confirmed,"
            " COALESCE(SUM(review_status = 'false_alarm'),0) false_alarm,"
            " AVG(suspicion) avg_suspicion FROM trials WHERE ts > ?", (since,)).fetchone()
        cm = conn.execute(
            "SELECT ground_truth gt, flagged, COUNT(*) n FROM trials"
            " WHERE ground_truth IS NOT NULL AND ts > ? GROUP BY ground_truth, flagged", (since,)).fetchall()
        pending_all = conn.execute("SELECT COUNT(*) FROM trials WHERE review_status = 'pending'").fetchone()[0]
    out = dict(row)
    out["pending_all_time"] = pending_all
    m = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for r in cm:
        key = ("t" if (r["gt"] == "theft") == bool(r["flagged"]) else "f") + ("p" if r["flagged"] else "n")
        m[key] += r["n"]
    out["mock_confusion"] = m
    return out


def delete_before(path: str | Path, ts: float) -> list[str]:
    """Retention: drop trials older than ts; returns their snapshot file names to delete."""
    with connect(path) as conn:
        snaps = [r[0] for r in conn.execute(
            "SELECT snapshot FROM trials WHERE ts < ? AND snapshot IS NOT NULL", (ts,))]
        conn.execute("DELETE FROM reviews WHERE trial_id IN (SELECT id FROM trials WHERE ts < ?)", (ts,))
        conn.execute("DELETE FROM trials WHERE ts < ?", (ts,))
    return snaps
