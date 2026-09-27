"""Command Center: live feed, status tiles, theft-alert inbox, suspicion timeline."""
from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st

import lib

d = lib.dash()
lib.header(d["title"], d["subtitle"])

try:
    lib.get("/api/health")
except lib.APIError as exc:
    lib.api_down_banner(exc)
    st.stop()

REFRESH = max(0.5, float(d["refresh_sec"]))


@st.fragment(run_every=REFRESH)
def tiles() -> None:
    status = lib.status_or_none() or {"node_online": False, "node": None}
    node = status["node"] or {}
    stats = lib.get("/api/stats", since=lib.today_start())
    online = status["node_online"]
    all_tiles = {
        "node": lib.tile("Fog node", "ONLINE" if online else "OFFLINE",
                         lib.ago(node.get("ts")) if node else "no heartbeat yet", "ok" if online else "alert"),
        "camera": lib.tile("Camera", {"offline": "NO SIGNAL", "connecting": "CONNECTING"}.get(
                               node.get("source_kind"), "LIVE") if online else "—",
                           {"offline": "reconnecting automatically", "connecting": "opening source"}.get(
                               node.get("source_kind"), node.get("source_kind") or ""),
                           {"offline": "alert", "connecting": "warn"}.get(node.get("source_kind"), "ok")
                           if online else ""),
        "mode": lib.tile("Mode", (node.get("mode") or "—").upper(),
                         (node.get("source") or "")[:28] or "—"),
        "fps": lib.tile("Throughput", f"{node.get('fps', 0):.0f} fps" if online else "—", "frames processed"),
        "latency": lib.tile("Inference", f"{node.get('last_infer_ms', 0):.0f} ms" if node else "—",
                            "CNN → PLS → 5× QSVM"),
        "trials": lib.tile("Trials today", f"{stats['n']:,}", "motion-gated events"),
        "flagged": lib.tile("Flagged today", f"{stats['flagged']:,}",
                            f"{stats['flagged'] / stats['n']:.0%} of trials" if stats["n"] else "—",
                            "alert" if stats["flagged"] else ""),
        "pending": lib.tile("Awaiting review", f"{stats['pending_all_time']:,}", "all time",
                            "warn" if stats["pending_all_time"] else "ok"),
        "threshold": lib.tile("Flag threshold", lib.score(node.get("flag_threshold")),
                              f"sensitivity {node.get('sensitivity', 0):.2f}" if node else ""),
        "gait": lib.tile("Live gait", f"{node.get('gait', {}).get('speed_px_s', 0):.0f} px/s" if node else "—",
                         f"jitter {node.get('gait', {}).get('jitter_score', 0):.2f}" if node else ""),
    }
    st.html('<div class="tiles">' + "".join(all_tiles[k] for k in d["tiles"] if k in all_tiles) + "</div>")


tiles()

left, right = st.columns([1.55, 1], gap="large")

with left:
    if d["show_video"]:
        st.html(f'<div class="video" style="max-width:{d["video_max_width_px"]}px">'
                f'<span class="tag">{lib.pill("Live", "alert", live=True)}</span>'
                f'<img src="{lib.media_url("/api/stream?fps=10")}" alt="Live feed — waiting for the fog node"/></div>')
        st.caption("Green box = monitored region (ROI). The frame border turns red for 2 s after a flag.")

    @st.fragment(run_every=REFRESH)
    def timeline() -> None:
        window = d["timeline_minutes"] * 60
        rows = lib.get("/api/trials", since=time.time() - window, limit=5000)
        st.markdown(f"##### Suspicion · last {d['timeline_minutes']} min")
        if not rows:
            st.caption("No trials in this window yet. A trial is recorded each time motion in the ROI passes the gate.")
            return
        from datetime import datetime
        xs = [datetime.fromtimestamp(r["ts"], lib.tz()) for r in rows]
        colors = [d["alert_color"] if r["flagged"] else d["accent_color"] for r in rows]
        fig = go.Figure(go.Scatter(
            x=xs, y=[r["suspicion"] for r in rows], mode="markers+lines",
            line=dict(color="rgba(124,92,255,.25)", width=1), marker=dict(color=colors, size=8),
            customdata=[[r["id"][:8], r.get("crime_signal") or 0, r["gait"].get("speed_px_s", 0),
                         r.get("ground_truth") or "—"] for r in rows],
            hovertemplate="%{x|%H:%M:%S}<br>suspicion %{y:.2f}<br>crime signal %{customdata[1]:.2f}"
                          "<br>speed %{customdata[2]:.0f} px/s<br>truth %{customdata[3]}<br>%{customdata[0]}"
                          "<extra></extra>"))
        node = (lib.status_or_none() or {}).get("node") or {}
        if d["show_thresholds"] and node.get("flag_threshold") is not None:
            fig.add_hline(y=node["flag_threshold"], line=dict(color=d["alert_color"], dash="dot", width=1),
                          annotation_text="flag threshold", annotation_font_color=d["alert_color"])
        fig.update_yaxes(range=[0, 1.02], title=None)
        st.plotly_chart(lib.fig_layout(fig), width="stretch", config={"displayModeBar": False})

    timeline()

with right:
    @st.fragment(run_every=REFRESH)
    def inbox() -> None:
        pending = lib.get("/api/trials", flagged=True, review_status="pending", limit=d["inbox_size"], full=True)
        n = f"{len(pending)}+" if len(pending) >= d["inbox_size"] else str(len(pending))  # list is capped
        st.html(f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<h5 style="margin:0">Theft alerts</h5>'
                f'{lib.pill(f"{n} pending", "alert" if pending else "ok")}</div>')
        if not pending:
            st.caption("Nothing waiting. When the AI judge flags a trial it appears here for review.")
        for t in pending:
            v = t["payload"]["verdict"]
            with st.container(border=True):
                c1, c2 = st.columns([1, 1.35], gap="small")
                if t.get("snapshot"):
                    c1.image(lib.media_url(f"/api/trials/{t['id']}/snapshot.jpg"), width="stretch")
                truth = t.get("ground_truth")
                truth_html = (f'<span class="k">truth</span><span class="v">{truth}</span>'
                              if truth and d["show_ground_truth"] else "")
                c2.html(f'<div class="kv"><span class="k">time</span><span class="v">{lib.fmt_time(t["ts"])}</span>'
                        f'<span class="k">suspicion</span><span class="v" style="color:var(--alert)">'
                        f'{lib.score(v["suspicion_score"])}</span>'
                        f'<span class="k">crime</span><span class="v">{lib.score(v.get("signals", {}).get("crime_signal"))}</span>'
                        f'<span class="k">speed</span><span class="v">{v["gait"].get("speed_px_s", 0):.0f} px/s</span>'
                        f'{truth_html}</div>')
                if t.get("ai_review"):
                    r = t["ai_review"]
                    kind = {"theft": "alert", "not_theft": "ok"}.get(r["verdict"], "warn")
                    c2.html(lib.pill(f"AI: {r['verdict'].replace('_', ' ')} {r.get('confidence', 0):.0%}", kind))
                    c2.caption(r.get("reason", ""))
                (b1, b2), (b3, b4) = st.columns(2, gap="small"), st.columns(2, gap="small")  # 4 in a row truncate labels
                for col, label, status_, icon in ((b1, "Confirm", "confirmed", ":material/check:"),
                                                  (b2, "False", "false_alarm", ":material/close:"),
                                                  (b3, "Dismiss", "dismissed", ":material/archive:")):
                    if col.button(label, key=f"{status_}-{t['id']}", icon=icon, width="stretch",
                                  type="primary" if status_ == "confirmed" else "secondary"):
                        try:
                            lib.api("PATCH", f"/api/trials/{t['id']}", json={"status": status_})
                            st.toast(f"Marked {status_.replace('_', ' ')}", icon=":material/done:")
                        except lib.APIError as exc:
                            st.error(str(exc))
                        st.rerun(scope="fragment")
                if b4.button("Inspect", key=f"insp-{t['id']}", icon=":material/account_tree:", width="stretch"):
                    st.session_state["inspect_trial"] = t["id"]
                    st.switch_page("views/inspector.py")

    inbox()
