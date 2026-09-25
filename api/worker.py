"""API-side background worker: for each newly flagged trial, optionally get a
Claude second opinion, then send notifications (ntfy / webhook / email).
Also applies snapshot/trial retention once an hour.

Runs in the API process, not the node, so a slow network call can never stall
frame processing. Progress is persisted (data/worker_state.json) so a restart
neither re-sends old alerts nor floods you with the backlog.
"""
from __future__ import annotations

import base64
import json
import logging
import smtplib
import threading
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests

import store
from api import app_settings

log = logging.getLogger("dripbl.worker")
_POLL_SEC = 2.0
_RETENTION_EVERY_SEC = 3600.0

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["theft", "not_theft", "uncertain"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You review alerts from a retail loss-prevention camera system. A fast on-device "
    "model (a quantum-kernel SVM on 64x64 frames plus gait heuristics) has flagged a "
    "moment as possible theft. You get its signals and, when available, the camera frame. "
    "Decide whether the evidence supports theft. The on-device model is known to react to "
    "scene appearance rather than the act itself, so weigh what is actually visible. "
    "Be concise: one or two sentences of reasoning a store manager can act on."
)


def claude_review(trial: dict[str, Any], snapshot: Path | None, cfg: dict[str, Any]) -> dict[str, Any]:
    import anthropic  # optional dependency, only needed in rules+claude mode

    client = anthropic.Anthropic(api_key=cfg["api_key"]) if cfg["api_key"] else anthropic.Anthropic()
    v = trial["payload"]["verdict"]
    facts = {
        "suspicion_score": round(v["suspicion_score"], 3),
        "flag_threshold": round(v["threshold"], 3),
        "signals": v.get("signals"),
        "weighted_breakdown": v.get("breakdown"),
        "gait": v.get("gait"),
        "classifier": {k: trial["payload"]["quantum"].get(k) for k in ("object_class", "margin", "backend")},
    }
    content: list[dict[str, Any]] = []
    if cfg["send_snapshot"] and snapshot is not None and snapshot.is_file():
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(snapshot.read_bytes()).decode()}})
    text = "Flagged event signals:\n" + json.dumps(facts, indent=2)
    if cfg["extra_instructions"]:
        text += "\n\nOperator instructions: " + cfg["extra_instructions"]
    content.append({"type": "text", "text": text})

    kwargs: dict[str, Any] = dict(
        model=cfg["model"], max_tokens=2000, system=_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": cfg["effort"], "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
    )
    if cfg["use_fallbacks"]:
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        response = client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "Model declined to review this frame.",
                "model": response.model, "ts": time.time()}
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    return {**data, "model": response.model, "ts": time.time()}


def _message(trial: dict[str, Any], base_url: str) -> tuple[str, str]:
    v = trial["payload"]["verdict"]
    title = f"Possible theft · suspicion {v['suspicion_score']:.2f}"
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trial["ts"]))
    lines = [f"Time: {when}", f"Source: {trial.get('source') or '-'}",
             f"Crime signal: {v.get('signals', {}).get('crime_signal', 0):.2f}",
             f"Speed: {v['gait'].get('speed_px_s', 0):.0f} px/s · jitter {v['gait'].get('jitter_score', 0):.2f}"]
    if trial.get("ai_review"):
        r = trial["ai_review"]
        lines.append(f"AI review: {r['verdict']} ({r['confidence']:.0%}) — {r['reason']}")
    lines.append(f"Review: {base_url.rstrip('/')}/Alerts?trial={trial['id']}")
    return title, "\n".join(lines)


def send_notifications(trial: dict[str, Any], snapshot: Path | None, cfg: dict[str, Any]) -> dict[str, str]:
    """Returns {channel: "ok" | error text} for every enabled channel."""
    title, body = _message(trial, cfg["dashboard_url"])
    results: dict[str, str] = {}

    ntfy = cfg["ntfy"]
    if ntfy["enabled"] and ntfy["topic"]:
        try:
            url = f"{ntfy['server'].rstrip('/')}/{ntfy['topic']}"
            headers = {"Title": title, "Priority": ntfy["priority"], "Tags": "rotating_light"}
            if snapshot is not None and snapshot.is_file():
                headers.update({"Filename": "snapshot.jpg", "Message": body.replace("\n", " · ")})
                r = requests.put(url, data=snapshot.read_bytes(), headers=headers, timeout=10)
            else:
                r = requests.post(url, data=body.encode(), headers=headers, timeout=10)
            r.raise_for_status()
            results["ntfy"] = "ok"
        except requests.RequestException as exc:
            results["ntfy"] = str(exc)[:200]

    hook = cfg["webhook"]
    if hook["enabled"] and hook["url"]:
        try:
            if hook["format"] == "slack":
                payload: dict[str, Any] = {"text": f"*{title}*\n{body}"}
            elif hook["format"] == "discord":
                payload = {"content": f"**{title}**\n{body}"}
            else:
                payload = {"event": "trial.flagged", "title": title, "message": body,
                           "trial": {k: trial[k] for k in ("id", "ts", "suspicion", "source", "ai_review")}}
            requests.post(hook["url"], json=payload, timeout=10).raise_for_status()
            results["webhook"] = "ok"
        except requests.RequestException as exc:
            results["webhook"] = str(exc)[:200]

    mail = cfg["email"]
    if mail["enabled"] and mail["smtp_host"] and mail["recipients"]:
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"] = title, mail["sender"] or mail["username"]
            msg["To"] = mail["recipients"]
            msg.set_content(body)
            if snapshot is not None and snapshot.is_file():
                msg.add_attachment(snapshot.read_bytes(), maintype="image", subtype="jpeg", filename="snapshot.jpg")
            with smtplib.SMTP(mail["smtp_host"], mail["smtp_port"], timeout=15) as smtp:
                if mail["use_tls"]:
                    smtp.starttls()
                if mail["username"]:
                    smtp.login(mail["username"], mail["password"])
                smtp.send_message(msg)
            results["email"] = "ok"
        except (OSError, smtplib.SMTPException) as exc:
            results["email"] = str(exc)[:200]
    return results


class Worker:
    def __init__(self, db: Path, snap_dir: Path, state_path: Path) -> None:
        self._db, self._snaps, self._state_path = db, snap_dir, state_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="dripbl-worker")
        self.last_error: str | None = None
        self.last_results: dict[str, str] = {}
        try:
            self._state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            self._state = {"last_ts": time.time()}  # first start: don't replay history
        self._last_sent = 0.0
        self._last_retention = 0.0

    def start(self) -> "Worker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _save(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state))
        tmp.replace(self._state_path)

    def _loop(self) -> None:
        while not self._stop.wait(_POLL_SEC):
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - the worker must survive anything
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.exception("worker tick failed")

    def _tick(self) -> None:
        cfg = app_settings.load()
        new = store.list_trials(self._db, flagged=True, since=self._state["last_ts"], limit=50, full=True)
        for trial in sorted(new, key=lambda t: t["ts"]):
            snap = self._snaps / trial["snapshot"] if trial.get("snapshot") else None
            if cfg["ai_judge"]["mode"] == "rules+claude" and not trial.get("ai_review"):
                try:
                    trial["ai_review"] = claude_review(trial, snap, cfg["ai_judge"])
                except Exception as exc:  # noqa: BLE001
                    trial["ai_review"] = {"verdict": "error", "confidence": 0.0,
                                          "reason": f"{type(exc).__name__}: {exc}"[:300], "ts": time.time()}
                store.set_ai_review(self._db, trial["id"], trial["ai_review"])
            self._maybe_notify(trial, snap, cfg["notifications"])
            self._state["last_ts"] = trial["ts"]
            self._save()

        days = cfg["retention"]["days"]
        if days > 0 and time.time() - self._last_retention > _RETENTION_EVERY_SEC:
            self._last_retention = time.time()
            for name in store.delete_before(self._db, time.time() - days * 86400):
                (self._snaps / name).unlink(missing_ok=True)

    def _maybe_notify(self, trial: dict[str, Any], snap: Path | None, cfg: dict[str, Any]) -> None:
        if not cfg["enabled"] or trial["suspicion"] < cfg["min_suspicion"]:
            return
        if cfg["only_if_ai_agrees"] and (trial.get("ai_review") or {}).get("verdict") == "not_theft":
            return
        if time.time() - self._last_sent < cfg["cooldown_sec"]:
            return
        self.last_results = send_notifications(trial, snap, cfg)
        self._last_sent = time.time()
