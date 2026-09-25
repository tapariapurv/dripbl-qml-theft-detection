"""Alerts: every flagged claim, filterable, with review + notes + history and CSV export."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import lib

d = lib.dash()
lib.header("Alerts", "Every trial the AI judge flagged. Select a row to review it.")

STATUSES = {"pending": "Pending", "confirmed": "Confirmed theft", "false_alarm": "False alarm", "dismissed": "Dismissed"}

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1, 1])
    status_sel = f1.pills("Status", list(STATUSES), format_func=STATUSES.get, selection_mode="multi",
                          default=["pending"], key="al_status")
    now = datetime.now(lib.tz())
    span = f2.selectbox("Period", ["Last hour", "Today", "Last 7 days", "Last 30 days", "All time"], index=2)
    since = {"Last hour": time.time() - 3600, "Today": lib.today_start(),
             "Last 7 days": (now - timedelta(days=7)).timestamp(),
             "Last 30 days": (now - timedelta(days=30)).timestamp(), "All time": None}[span]
    min_s = f3.slider("Min suspicion", 0.0, 1.0, 0.0, 0.05)
    truth = f4.selectbox("Ground truth", ["any", "theft", "normal"], help="Mock mode only")
    include_all = st.toggle("Include non-flagged trials", help="Show every recorded trial, not only alerts")

try:
    rows = lib.get("/api/trials", flagged=None if include_all else True, since=since, limit=5000,
                   ground_truth=None if truth == "any" else truth)
except lib.APIError as exc:
    lib.api_down_banner(exc)
    st.stop()

rows = [r for r in rows if r["suspicion"] >= min_s and
        (include_all or not status_sel or r.get("review_status") in status_sel)]

top = st.columns([1, 1, 1, 1, 1.2])
top[0].metric("Shown", f"{len(rows):,}")
top[1].metric("Pending", sum(r.get("review_status") == "pending" for r in rows))
top[2].metric("Confirmed", sum(r.get("review_status") == "confirmed" for r in rows))
top[3].metric("False alarms", sum(r.get("review_status") == "false_alarm" for r in rows))
with top[4]:
    try:
        csv = lib.api("GET", "/api/trials.csv", params={"flagged": None if include_all else True, "since": since})
        st.download_button("Export CSV", csv, "dripbl_alerts.csv", "text/csv", icon=":material/download:",
                           width="stretch")
    except lib.APIError as exc:
        st.caption(str(exc))

if not rows:
    st.info("No alerts match these filters.")
    st.stop()

page_size = d["table_page_size"]
pages = max(1, -(-len(rows) // page_size))
page = st.number_input("Page", 1, pages, 1, width=120) if pages > 1 else 1
chunk = rows[(page - 1) * page_size: page * page_size]

df = pd.DataFrame([{
    "time": lib.fmt_time(r["ts"], date=True),
    "suspicion": r["suspicion"],
    "crime": r.get("crime_signal"),
    "speed px/s": r["gait"].get("speed_px_s", 0.0),
    "jitter": r["gait"].get("jitter_score", 0.0),
    "status": STATUSES.get(r.get("review_status"), "—") if r["flagged"] else "not flagged",
    "AI review": (r.get("ai_review") or {}).get("verdict", "—"),
    "truth": r.get("ground_truth") or "—",
    "source": r.get("source") or "—",
    "note": r.get("review_note") or "",
} for r in chunk])
if not d["show_ground_truth"]:
    df = df.drop(columns="truth")

event = st.dataframe(
    df, hide_index=True, width="stretch", height=min(36 * (len(df) + 1) + 12, 520),
    on_select="rerun", selection_mode="single-row", key="al_table",
    column_config={
        "suspicion": st.column_config.ProgressColumn("suspicion", min_value=0, max_value=1, format="%.2f"),
        "crime": st.column_config.NumberColumn(format="%.2f"),
        "speed px/s": st.column_config.NumberColumn(format="%.0f"),
        "jitter": st.column_config.NumberColumn(format="%.2f"),
    },
)

sel = event.selection.rows
wanted = st.query_params.get("trial")
chosen = chunk[sel[0]]["id"] if sel else (wanted if wanted else None)
if not chosen:
    st.caption("Select a row to open its review panel.")
    st.stop()

try:
    t = lib.get(f"/api/trials/{chosen}")
except lib.APIError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
v = t["payload"]["verdict"]
a, b = st.columns([1, 1.2], gap="large")
with a:
    st.markdown(f"#### {lib.fmt_time(t['ts'], date=True)}")
    if t.get("snapshot"):
        st.image(lib.media_url(f"/api/trials/{t['id']}/snapshot.jpg"), width="stretch")
    st.html(f'<div class="kv">'
            f'<span class="k">trial</span><span class="v">{t["id"]}</span>'
            f'<span class="k">suspicion</span><span class="v">{v["suspicion_score"]:.3f} (threshold {v["threshold"]:.3f})</span>'
            f'<span class="k">crime signal</span><span class="v">{v.get("signals", {}).get("crime_signal", 0):.3f}</span>'
            f'<span class="k">speed</span><span class="v">{v["gait"].get("speed_px_s", 0):.1f} px/s</span>'
            f'<span class="k">stride</span><span class="v">{v["gait"].get("stride_freq_hz", 0):.2f} Hz</span>'
            f'<span class="k">jitter</span><span class="v">{v["gait"].get("jitter_score", 0):.2f}</span>'
            f'<span class="k">source</span><span class="v">{t.get("source") or "—"}</span></div>')
    if st.button("Open in Pipeline Inspector", icon=":material/account_tree:"):
        st.session_state["inspect_trial"] = t["id"]
        st.switch_page("views/inspector.py")
with b:
    if t.get("ai_review"):
        r = t["ai_review"]
        with st.container(border=True):
            st.html(lib.pill(f"Claude second opinion: {r['verdict'].replace('_', ' ')} · {r.get('confidence', 0):.0%}",
                             {"theft": "alert", "not_theft": "ok"}.get(r["verdict"], "warn")))
            st.write(r.get("reason", ""))
    if t["flagged"]:
        with st.form(f"review-{t['id']}", border=True):
            st.markdown("**Review**")
            current = t.get("review_status") or "pending"
            new_status = st.segmented_control("Decision", list(STATUSES), format_func=STATUSES.get, default=current)
            note = st.text_area("Note", t.get("review_note") or "", placeholder="What did you see on camera?")
            reviewer = st.text_input("Reviewer", st.session_state.get("reviewer", ""), placeholder="Your name")
            if st.form_submit_button("Save review", type="primary", icon=":material/save:"):
                st.session_state["reviewer"] = reviewer
                try:
                    lib.api("PATCH", f"/api/trials/{t['id']}",
                            json={"status": new_status or current, "note": note, "reviewer": reviewer})
                    st.toast("Review saved", icon=":material/done:")
                    st.rerun()
                except lib.APIError as exc:
                    st.error(str(exc))
        if t.get("review_history"):
            st.markdown("**History**")
            for h in reversed(t["review_history"]):
                st.caption(f"{lib.fmt_time(h['ts'], date=True)} · **{STATUSES.get(h['status'], h['status'])}**"
                           + (f" by {h['reviewer']}" if h.get("reviewer") else "")
                           + (f" — {h['note']}" if h.get("note") else ""))
    else:
        st.info("This trial was not flagged, so there is nothing to review.")
