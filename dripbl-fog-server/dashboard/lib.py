"""Shared dashboard plumbing: the API client (the dashboard's ONLY data source),
theme CSS built from the user's colour settings, and small render helpers."""
from __future__ import annotations

import html
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.environ.get("DRIPBL_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_PORT = API_URL.rsplit(":", 1)[-1].split("/")[0] if API_URL.count(":") >= 2 else "8000"


def _key() -> str:
    if os.environ.get("DRIPBL_API_KEY"):
        return os.environ["DRIPBL_API_KEY"]
    try:  # same machine as the API: read the key it generated
        return (ROOT / "data" / "api_key").read_text().strip()
    except OSError:
        return ""


@st.cache_resource
def _session() -> requests.Session:
    return requests.Session()  # pooled connections; the key is sent per request (see api())


class APIError(RuntimeError):
    pass


def api(method: str, path: str, **kwargs: Any) -> Any:
    try:
        # key read per request: on boot the API may create data/api_key after the dashboard starts
        r = _session().request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 10),
                               headers={"X-API-Key": _key()}, **kwargs)
    except requests.RequestException as exc:
        raise APIError(f"API unreachable at {API_URL} — is it running? ({type(exc).__name__})") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise APIError(f"{r.status_code}: {detail}")
    return r.json() if "json" in r.headers.get("content-type", "") else r.content


def get(path: str, **params: Any) -> Any:
    return api("GET", path, params={k: v for k, v in params.items() if v is not None})


def browser_api_url() -> str:
    """The API as the *viewer's browser* reaches it (same host the dashboard was opened on),
    so the <img> video stream works from another device on the network too."""
    host = (st.context.headers.get("host") or "localhost").rsplit(":", 1)[0]
    return os.environ.get("DRIPBL_PUBLIC_API_URL", f"http://{host}:{API_PORT}").rstrip("/")


def media_url(path: str) -> str:
    """URL for <img> tags, which cannot send headers: add ?key= only when reads need it."""
    sep = "&" if "?" in path else "?"
    return f"{browser_api_url()}{path}" + (f"{sep}key={_key()}" if settings()["api"]["require_key_for_reads"] else "")


# --- settings (cached briefly so every widget doesn't hit the API) --------------

@st.cache_data(ttl=3, show_spinner=False)
def _app_settings() -> dict[str, Any]:
    return get("/api/app-settings")


def settings() -> dict[str, Any]:
    try:
        return _app_settings()
    except APIError:
        return get_defaults_offline()


@st.cache_data(show_spinner=False)
def get_defaults_offline() -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(ROOT))
    from api.app_settings import DEFAULTS
    return DEFAULTS


def dash() -> dict[str, Any]:
    return settings()["dashboard"]


def clear_caches() -> None:
    _app_settings.clear()


# --- formatting -------------------------------------------------------------------

def tz() -> ZoneInfo:
    try:
        return ZoneInfo(dash()["timezone"])
    except Exception:  # noqa: BLE001 - bad timezone string in settings
        return ZoneInfo("UTC")


def fmt_time(ts: float | None, date: bool = False) -> str:
    if not ts:
        return "—"
    d = datetime.fromtimestamp(ts, tz())
    fmt = (dash()["date_format"] + " " if date else "") + dash()["time_format"]
    return d.strftime(fmt)


def ago(ts: float | None) -> str:
    if not ts:
        return "—"
    s = max(0, time.time() - ts)
    if s < 60:
        return f"{s:.0f}s ago"
    if s < 3600:
        return f"{s / 60:.0f}m ago"
    if s < 86400:
        return f"{s / 3600:.1f}h ago"
    return f"{s / 86400:.1f}d ago"


def score(x: float | None) -> str:
    return "—" if x is None else f"{x:.{dash()['suspicion_decimals']}f}"


def today_start() -> float:
    return datetime.now(tz()).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# --- theme ------------------------------------------------------------------------

def inject_css() -> None:
    d = dash()
    st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {{
  --accent: {d['accent_color']}; --alert: {d['alert_color']}; --ok: {d['ok_color']}; --warn: {d['warn_color']};
  --bg: #0B0D12; --surface: #12151C; --surface-2: #171B24; --border: #232838; --muted: #8A91A5; --text: #E6E8EE;
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
[data-testid="stSidebar"] {{ background: #0E1117; border-right: 1px solid var(--border); }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

.dp-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap;
  border-bottom:1px solid var(--border); padding-bottom:14px; margin-bottom:18px; }}
.dp-title {{ font-size:1.55rem; font-weight:700; color:var(--text); line-height:1.1; }}
.dp-sub {{ color:var(--muted); font-size:.88rem; margin-top:4px; }}

.pill {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px;
  font-size:.74rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase; border:1px solid var(--border);
  background:var(--surface-2); color:var(--muted); white-space:nowrap; }}
.pill .dot {{ width:7px; height:7px; border-radius:50%; background:currentColor; }}
.pill.ok {{ color:var(--ok); border-color:color-mix(in srgb, var(--ok) 35%, transparent); }}
.pill.alert {{ color:var(--alert); border-color:color-mix(in srgb, var(--alert) 40%, transparent);
  background:color-mix(in srgb, var(--alert) 10%, var(--surface-2)); }}
.pill.warn {{ color:var(--warn); border-color:color-mix(in srgb, var(--warn) 35%, transparent); }}
.pill.accent {{ color:var(--accent); border-color:color-mix(in srgb, var(--accent) 40%, transparent); }}
.pill.live .dot {{ animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 color-mix(in srgb, currentColor 60%, transparent); }}
  70% {{ box-shadow:0 0 0 7px transparent; }} 100% {{ box-shadow:0 0 0 0 transparent; }} }}

.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-bottom:6px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }}
.tile .lbl {{ color:var(--muted); font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }}
.tile .val {{ font-family:'JetBrains Mono', monospace; font-size:1.45rem; font-weight:600; color:var(--text); margin-top:4px; }}
.tile .hint {{ color:var(--muted); font-size:.72rem; margin-top:2px; }}
.tile.alert .val {{ color:var(--alert); }} .tile.ok .val {{ color:var(--ok); }} .tile.warn .val {{ color:var(--warn); }}

.video {{ position:relative; background:#05060A; border:1px solid var(--border); border-radius:14px; overflow:hidden; }}
.video img {{ width:100%; display:block; }}
.video .tag {{ position:absolute; top:10px; left:10px; }}

.alert-card {{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--alert);
  border-radius:12px; padding:12px 14px; }}
.alert-card.reviewed {{ border-left-color: var(--muted); opacity:.85; }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:3px 12px; font-size:.84rem; }}
.kv .k {{ color:var(--muted); }} .kv .v {{ font-family:'JetBrains Mono', monospace; }}

.stage {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px 16px; height:100%; }}
.stage .num {{ display:inline-flex; width:24px; height:24px; border-radius:50%; align-items:center; justify-content:center;
  background:color-mix(in srgb, var(--accent) 22%, transparent); color:var(--accent); font-weight:700; font-size:.8rem; margin-right:8px; }}
.stage .name {{ font-weight:600; }}
.stage .desc {{ color:var(--muted); font-size:.8rem; margin:4px 0 10px; }}
.flow-arrow {{ text-align:center; color:var(--muted); font-size:1.2rem; margin:2px 0; }}

.bar {{ height:8px; background:var(--surface-2); border-radius:6px; overflow:hidden; border:1px solid var(--border); }}
.bar > span {{ display:block; height:100%; background:var(--accent); }}
.callout {{ background:color-mix(in srgb, var(--accent) 8%, var(--surface)); border:1px solid color-mix(in srgb, var(--accent) 30%, var(--border));
  border-radius:12px; padding:12px 14px; font-size:.88rem; color:var(--text); }}
div[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:10px 14px; }}
</style>""")


def esc(text: object) -> str:
    """Anything user- or data-controlled goes through this before landing in st.html."""
    return html.escape(str(text), quote=True)


def header(title: str, subtitle: str = "", right_html: str = "") -> None:
    st.html(f'<div class="dp-header"><div><div class="dp-title">{esc(title)}</div>'
            f'<div class="dp-sub">{esc(subtitle)}</div></div><div>{right_html}</div></div>')


def pill(text: str, kind: str = "", live: bool = False) -> str:
    return f'<span class="pill {kind}{" live" if live else ""}"><span class="dot"></span>{esc(text)}</span>'


def tile(label: str, value: str, hint: str = "", kind: str = "") -> str:
    return f'<div class="tile {kind}"><div class="lbl">{esc(label)}</div><div class="val">{esc(value)}</div>' + \
        (f'<div class="hint">{esc(hint)}</div>' if hint else "") + "</div>"


def fig_layout(fig: go.Figure, height: int | None = None, **kw: Any) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height or dash()["chart_height_px"],
        margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#C9CDD8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#171B24", bordercolor="#232838", font=dict(family="JetBrains Mono, monospace")),
        **kw,
    )
    fig.update_xaxes(gridcolor="#1C2130", zerolinecolor="#232838")
    fig.update_yaxes(gridcolor="#1C2130", zerolinecolor="#232838")
    return fig


def status_or_none() -> dict[str, Any] | None:
    try:
        return get("/api/status")
    except APIError:
        return None


def api_down_banner(exc: Exception) -> None:
    st.error(f"{exc}\n\nStart everything with `.venv/bin/python3 start.py`.", icon=":material/cloud_off:")
