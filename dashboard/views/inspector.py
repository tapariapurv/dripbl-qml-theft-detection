"""Pipeline Inspector: one trial, every stage, in the order the fog node ran them."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import lib

d = lib.dash()
lib.header("Pipeline Inspector", "Follow one trial through every stage: camera → motion gate → gait → "
                                 "CNN → PLS angles → quantum kernel ensemble → AI judge → verdict")

try:
    recent = lib.get("/api/trials", limit=200)
except lib.APIError as exc:
    lib.api_down_banner(exc)
    st.stop()

if not recent:
    st.info("No trials recorded yet. Start the node (`start.py --mock` for research mode) and come back.")
    st.stop()

ids = [r["id"] for r in recent]
labels = {r["id"]: f"{lib.fmt_time(r['ts'], date=True)} · suspicion {lib.score(r['suspicion'])}"
                   f"{' · FLAGGED' if r['flagged'] else ''}"
                   f"{' · truth ' + r['ground_truth'] if r.get('ground_truth') and d['show_ground_truth'] else ''}"
          for r in recent}
wanted = st.session_state.pop("inspect_trial", None) or st.query_params.get("trial")
if wanted and wanted not in labels:  # older than the list: fetch it so it can still be selected
    try:
        t = lib.get(f"/api/trials/{wanted}")
        ids.insert(0, wanted)
        labels[wanted] = f"{lib.fmt_time(t['ts'], date=True)} · suspicion {lib.score(t['suspicion'])}"
    except lib.APIError:
        wanted = None

c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
follow = c2.toggle("Follow latest", value=not wanted, help="Always show the newest trial")
choice = ids[0] if follow else c1.selectbox("Trial", ids, index=ids.index(wanted) if wanted in ids else 0,
                                            format_func=labels.get)
if follow:
    c1.selectbox("Trial", [ids[0]], format_func=labels.get, disabled=True)
st.query_params["trial"] = choice
t = lib.get(f"/api/trials/{choice}")
p = t["payload"]
q, v, stage = p["quantum"], p["verdict"], p.get("stage", {})


def stage_card(num: int, name: str, desc: str) -> st.delta_generator.DeltaGenerator:
    box = st.container(border=False)
    box.html(f'<div class="stage" style="padding-bottom:4px;border-bottom-left-radius:0;border-bottom-right-radius:0">'
             f'<span class="num">{num}</span><span class="name">{name}</span><div class="desc">{desc}</div></div>')
    return box


def arrow() -> None:
    st.html('<div class="flow-arrow">↓</div>')


# 1 + 2: frame and motion gate ---------------------------------------------------------------
a, b = st.columns([1.2, 1], gap="large")
with a:
    stage_card(1, "Camera frame", f"Source <b>{stage.get('source') or t.get('source') or '—'}</b> · "
                                  f"mode {stage.get('mode', '—')} · {lib.fmt_time(t['ts'], date=True)}")
    if t.get("snapshot"):
        st.image(lib.media_url(f"/api/trials/{t['id']}/snapshot.jpg"), width="stretch")
    else:
        st.caption("Only flagged trials keep a snapshot (saves disk on the fog node).")
with b:
    stage_card(2, "Motion gate (MOG2 in ROI)",
               "Background subtraction inside the monitored region; a trial starts only if the moving "
               "area clears the sensitivity-scaled threshold.")
    roi = stage.get("roi") or [0, 0, 0, 0]
    st.html(f'<div class="kv"><span class="k">triggered</span><span class="v">yes</span>'
            f'<span class="k">ROI</span><span class="v">x={roi[0]} y={roi[1]} {roi[2]}×{roi[3]}</span>'
            f'<span class="k">area threshold</span><span class="v">{stage.get("area_threshold", 0):.0f} px²</span>'
            f'<span class="k">sensitivity</span><span class="v">{stage.get("sensitivity", 0):.2f}</span></div>')
    arrow()
    stage_card(3, "Gait analysis", "Rolling 20-frame track of the moving blob's centroid.")
    g = v["gait"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Speed", f"{g.get('speed_px_s', 0):.0f} px/s")
    m2.metric("Stride", f"{g.get('stride_freq_hz', 0):.2f} Hz")
    m3.metric("Jitter", f"{g.get('jitter_score', 0):.2f}", help="Heading variance, 0–1: the nervousness proxy")

arrow()

# 4 + 5: CNN latent -> PLS angles --------------------------------------------------------------
a, b = st.columns([1.4, 1], gap="large")
with a:
    stage_card(4, "LightweightCNN → 128-d latent", "Frame resized to 64×64 RGB, 3 conv blocks, "
                                                 "single-threaded CPU. First ensemble member shown.")
    latent = q.get("latent")
    if latent:
        lat = np.asarray(latent)
        fig = go.Figure(go.Bar(y=lat, marker_color=[d["accent_color"] if x >= 0 else "#5B6275" for x in lat],
                               hovertemplate="dim %{x}: %{y:.3f}<extra></extra>"))
        fig.update_xaxes(title="latent dimension")
        st.plotly_chart(lib.fig_layout(fig, height=230, bargap=0.1), width="stretch",
                        config={"displayModeBar": False})
    else:
        st.caption("Latent not recorded (prototype fallback classifier, or a trial from before this dashboard).")
with b:
    stage_card(5, "PLS projection → 4 qubit angles", "Supervised 128→4 compression, scaled into the "
                                                   "ZZFeatureMap's encoding range. One angle per qubit.")
    ang = p.get("features") or []
    if ang:
        fig = go.Figure(go.Barpolar(r=[abs(x) for x in ang], theta=[f"q{i}" for i in range(len(ang))],
                                    marker_color=[d["accent_color"] if x >= 0 else d["warn_color"] for x in ang],
                                    customdata=ang, hovertemplate="%{theta}: %{customdata:.4f} rad<extra></extra>"))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(showticklabels=False, gridcolor="#1C2130"),
                                     angularaxis=dict(gridcolor="#1C2130")))
        st.plotly_chart(lib.fig_layout(fig, height=230), width="stretch", config={"displayModeBar": False})
        st.html('<div class="kv">' + "".join(f'<span class="k">q{i}</span><span class="v">{x:+.4f} rad</span>'
                                             for i, x in enumerate(ang)) + "</div>")

arrow()

# 6: quantum kernel ensemble -----------------------------------------------------------------
stage_card(6, "Quantum kernel SVM · 5-fold ensemble",
           "Each member encodes the angles with ZZFeatureMap(4 qubits, reps=2) and scores fidelity "
           "|⟨φ(x)|φ(sv)⟩|² against its support vectors. Margin > 0 leans theft; the ensemble averages them.")
mm = q.get("member_margins")
a, b = st.columns([1.4, 1], gap="large")
with a:
    if mm:
        names = [f"member {i + 1}" for i in range(len(mm))] + ["ensemble mean"]
        vals = list(mm) + [q.get("margin", float(np.mean(mm)))]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                               marker_color=[d["alert_color"] if x > 0 else d["ok_color"] for x in vals],
                               hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#8A91A5", width=1))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(lib.fig_layout(fig, height=230), width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Per-member margins not recorded for this trial.")
with b:
    crime = v.get("signals", {}).get("crime_signal", q.get("class_scores", {}).get("suspicious_signature", 0))
    st.metric("Classifier output", (q.get("object_class") or "—").upper())
    st.metric("Crime signal", lib.score(crime), help="Anomaly probability from the ensemble margin (0–1)")
    st.caption(f"Backend: {q.get('backend', '—')} · model: {stage.get('model', '—')} · "
               f"{stage.get('latency_ms', 0):.1f} ms for stages 4–7")

arrow()

# 7: AI judge ------------------------------------------------------------------------------------
stage_card(7, "AI judge", "Weighted vote of the crime signal and the gait signals, gated by the "
                          "sensitivity-derived threshold. Weights are editable in Settings → Detection.")
breakdown = v.get("breakdown") or {}
a, b = st.columns([1.4, 1], gap="large")
with a:
    if breakdown:
        names = list(breakdown)
        fig = go.Figure()
        palette = [d["accent_color"], d["warn_color"], "#4CC9F0", "#B5179E"]
        for i, n in enumerate(names):
            fig.add_bar(x=[breakdown[n]], y=["suspicion"], orientation="h", name=n.replace("_", " "),
                        marker_color=palette[i % len(palette)], hovertemplate=f"{n}: %{{x:.3f}}<extra></extra>")
        fig.add_vline(x=v["threshold"], line=dict(color=d["alert_color"], dash="dot", width=2),
                      annotation_text=f"threshold {v['threshold']:.2f}", annotation_font_color=d["alert_color"])
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(lib.fig_layout(fig, height=170, barmode="stack"), width="stretch",
                        config={"displayModeBar": False})
    else:
        st.caption("No weighted breakdown recorded for this trial.")
with b:
    flagged = v["flagged"]
    st.html(f'<div class="stage" style="border-color:{d["alert_color"] if flagged else d["ok_color"]}">'
            f'<div class="lbl" style="color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.07em">'
            f'Verdict</div><div style="font-size:1.6rem;font-weight:700;color:'
            f'{d["alert_color"] if flagged else d["ok_color"]}">{"FLAGGED" if flagged else "CLEAR"}</div>'
            f'<div class="mono" style="color:var(--muted)">suspicion {v["suspicion_score"]:.3f} '
            f'{"≥" if flagged else "<"} {v["threshold"]:.3f}</div></div>')
    if t.get("ground_truth") and d["show_ground_truth"]:
        correct = (t["ground_truth"] == "theft") == flagged
        st.html(lib.pill(f"ground truth: {t['ground_truth']} · {'correct' if correct else 'wrong'}",
                         "ok" if correct else "alert"))

if t.get("ai_review") or t.get("review_status"):
    arrow()
    stage_card(8, "Second opinion & human review", "Optional Claude review of flagged trials, then the operator's decision.")
    a, b = st.columns(2, gap="large")
    if t.get("ai_review"):
        r = t["ai_review"]
        a.html(lib.pill(f"Claude: {r['verdict'].replace('_', ' ')} · {r.get('confidence', 0):.0%}",
                        {"theft": "alert", "not_theft": "ok"}.get(r["verdict"], "warn")))
        a.write(r.get("reason", ""))
        a.caption(f"{r.get('model', '')} · {lib.fmt_time(r.get('ts'), date=True)}")
    if t.get("review_status"):
        b.html(lib.pill(f"review: {t['review_status'].replace('_', ' ')}",
                        {"confirmed": "alert", "false_alarm": "ok", "pending": "warn"}.get(t["review_status"], "")))
        for h in t.get("review_history", []):
            b.caption(f"{lib.fmt_time(h['ts'], date=True)} · {h['status']}" + (f" — {h['note']}" if h.get("note") else ""))

with st.expander("Raw trial JSON (exactly what the API returns)"):
    st.json(t, expanded=False)
