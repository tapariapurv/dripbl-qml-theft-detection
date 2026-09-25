"""Command-center settings (dashboard look, notifications, AI judge, quantum, API,
retention), persisted to app_settings.json next to runtime_config.json.

Node/pipeline tunables stay in config.Settings (runtime_config.json) because the
node hot-reloads them; everything here is only read by the API and dashboard.
DEFAULTS is the schema: a PUT may only set keys that exist here, with a value
of the same type. Secrets are never returned in clear text.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from config import ROOT

PATH = Path(os.environ.get("DRIPBL_APP_SETTINGS", ROOT / "app_settings.json"))
MASK = "********"

DEFAULTS: dict[str, Any] = {
    "dashboard": {
        "title": "DRIPBL Command Center",
        "subtitle": "Quantum-kernel theft detection · fog node",
        "accent_color": "#7C5CFF",
        "alert_color": "#FF4D6D",
        "ok_color": "#2ED3A0",
        "warn_color": "#FFB547",
        "refresh_sec": 2.0,            # how often live panels poll the API
        "timeline_minutes": 30,        # window of the suspicion timeline
        "inbox_size": 6,               # flagged alerts shown on the home page
        "table_page_size": 50,
        "timezone": "Asia/Kolkata",
        "time_format": "%H:%M:%S",
        "date_format": "%d %b %Y",
        "show_video": True,
        "video_max_width_px": 900,
        "tiles": ["node", "camera", "mode", "fps", "latency", "trials", "flagged", "pending"],
        "show_ground_truth": True,     # mock mode: show the true label next to verdicts
        "show_thresholds": True,       # draw the flag threshold on charts
        "suspicion_decimals": 2,
        "chart_height_px": 280,
        "hidden_pages": [],           # page titles to hide from the sidebar (Settings is always shown)
    },
    "notifications": {
        "enabled": False,
        "min_suspicion": 0.0,          # extra floor on top of the judge's own threshold
        "cooldown_sec": 30.0,          # at most one notification per this many seconds
        "only_if_ai_agrees": False,    # rules+claude mode: skip when Claude says not theft
        "dashboard_url": "http://localhost:8501",
        "ntfy": {"enabled": False, "server": "https://ntfy.sh", "topic": "", "priority": "high"},
        "webhook": {"enabled": False, "url": "", "format": "json"},   # "json" | "slack" | "discord"
        "email": {"enabled": False, "smtp_host": "", "smtp_port": 587, "use_tls": True,
                  "username": "", "password": "", "sender": "", "recipients": ""},
    },
    "ai_judge": {
        "mode": "rules",               # "rules" | "rules+claude" (second opinion on flagged trials)
        "model": "claude-opus-5",
        "effort": "low",               # low | medium | high
        "use_fallbacks": True,         # server-side refusal fallbacks
        "send_snapshot": True,         # include the flagged frame as an image
        "api_key": "",                 # blank = ANTHROPIC_API_KEY / `ant auth login` profile
        "extra_instructions": "",
    },
    "quantum": {
        "default_trials": 6,           # recent trials whose features the check compares
        "default_shots": 1024,
        "ibm_token": "",               # enables the real-hardware backend in Quantum Lab
        "ibm_instance": "",
        "ibm_backend": "",             # blank = least busy device
    },
    "api": {
        "require_key_for_reads": False,  # writes always need X-API-Key
        "cors_origins": ["*"],
    },
    "retention": {
        "days": 0,                     # delete trials + snapshots older than this; 0 = keep forever
    },
}

# Bounds for numeric settings (dotted path): out-of-range values would break the dashboard
# (a negative refresh interval, an inbox of -1 alerts) so they are rejected up front.
RANGES: dict[str, tuple[float, float]] = {
    "dashboard.refresh_sec": (0.5, 60), "dashboard.timeline_minutes": (1, 1440), "dashboard.inbox_size": (1, 50),
    "dashboard.table_page_size": (10, 1000), "dashboard.video_max_width_px": (200, 4000),
    "dashboard.suspicion_decimals": (0, 6), "dashboard.chart_height_px": (150, 1200),
    "notifications.min_suspicion": (0, 1), "notifications.cooldown_sec": (0, 86400),
    "notifications.email.smtp_port": (1, 65535),
    "quantum.default_trials": (2, 12), "quantum.default_shots": (16, 100000), "retention.days": (0, 36500),
}
CHOICES: dict[str, tuple[str, ...]] = {
    "ai_judge.mode": ("rules", "rules+claude"), "ai_judge.effort": ("low", "medium", "high"),
    "notifications.webhook.format": ("json", "slack", "discord"),
    "notifications.ntfy.priority": ("min", "low", "default", "high", "urgent"),
}

SECRETS = (("notifications", "email", "password"), ("ai_judge", "api_key"), ("quantum", "ibm_token"))


def _merge(base: dict, over: dict, path: str = "") -> dict:
    """Deep-merge `over` into a copy of `base`, rejecting unknown keys and wrong types."""
    out = copy.deepcopy(base)
    for key, val in over.items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"unknown setting: {where}")
        ref = base[key]
        if isinstance(ref, dict):
            if not isinstance(val, dict):
                raise ValueError(f"{where} must be an object")
            out[key] = _merge(ref, val, where)
        elif isinstance(ref, bool):
            if not isinstance(val, bool):
                raise ValueError(f"{where} must be true/false")
            out[key] = val
        elif isinstance(ref, (int, float)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{where} must be a number")
            if isinstance(ref, int) and not float(val).is_integer():
                raise ValueError(f"{where} must be a whole number")
            lo, hi = RANGES.get(where, (-1e9, 1e9))
            if not lo <= val <= hi:
                raise ValueError(f"{where} must be between {lo:g} and {hi:g}")
            out[key] = float(val) if isinstance(ref, float) else int(val)
        elif isinstance(ref, list):
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val) or len(val) > 100:
                raise ValueError(f"{where} must be a list of strings")
            out[key] = val
        else:
            if not isinstance(val, str) or len(val) > 5000:
                raise ValueError(f"{where} must be a string (max 5000 characters)")
            if key.endswith("_color") and not re.fullmatch(r"#[0-9a-fA-F]{6}", val):
                raise ValueError(f"{where} must be a hex colour like #7C5CFF")
            if where in CHOICES and val not in CHOICES[where]:
                raise ValueError(f"{where} must be one of {'|'.join(CHOICES[where])}")
            out[key] = val
    return out


def load() -> dict[str, Any]:
    """Full settings including secrets (server-side use only)."""
    try:
        saved = json.loads(PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    try:
        return _merge(DEFAULTS, saved)
    except ValueError:
        # a hand-edited file with a stale key: keep what still validates, section by section
        out = copy.deepcopy(DEFAULTS)
        for section, vals in saved.items():
            if section in DEFAULTS and isinstance(vals, dict):
                for k, v in vals.items():
                    try:
                        out[section] = _merge(out[section], {k: v}, section)
                    except ValueError:
                        pass
        return out


def masked(settings: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(settings)
    for path in SECRETS:
        node = out
        for key in path[:-1]:
            node = node[key]
        if node[path[-1]]:
            node[path[-1]] = MASK
    return out


def update(patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update; a secret sent back as MASK keeps its stored value."""
    current = load()
    merged = _merge(current, patch)
    for path in SECRETS:
        node, old = merged, current
        for key in path[:-1]:
            node, old = node[key], old[key]
        if node[path[-1]] == MASK:
            node[path[-1]] = old[path[-1]]
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    os.chmod(tmp, 0o600)  # holds secrets
    os.replace(tmp, PATH)
    return merged


def reset(section: str | None = None) -> dict[str, Any]:
    current = load()
    if section is None:
        fresh = copy.deepcopy(DEFAULTS)
    elif section in DEFAULTS:
        fresh = {**current, section: copy.deepcopy(DEFAULTS[section])}
    else:
        raise ValueError(f"unknown section: {section}")
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, PATH)
    return fresh
