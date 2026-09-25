"""Quantum Lab: run the fidelity kernel for recent live trials on real circuits and compare."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import lib

d = lib.dash()
qcfg = lib.settings()["quantum"]
lib.header("Quantum Lab", "Cross-check the kernel the fog node computes analytically against real quantum circuits")

st.html('<div class="callout">The node evaluates the ZZFeatureMap fidelity kernel <b>exactly</b> (closed-form '
        'statevector, ~15 ms/frame). Here you take the 4-qubit angles of recent live trials and compute the same '
        'kernel with actual circuits: on a <b>noisy digital twin of IBM Brisbane</b> (127-qubit heavy-hex topology, '
        'native gates, T1/T2 and gate errors) or on <b>real IBM hardware</b> if a token is set in Settings → Quantum.</div>')

BACKENDS = {"fake_brisbane": "FakeBrisbane digital twin (noisy)", "exact": "Exact statevector only",
            "ibm": "Real IBM Quantum hardware"}

with st.form("qc", border=True):
    c1, c2, c3 = st.columns([1.6, 1, 1])
    backend = c1.selectbox("Backend", list(BACKENDS), format_func=BACKENDS.get,
                           help="Real hardware needs quantum.ibm_token and may queue for a long time")
    shots = c2.select_slider("Shots per circuit", [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                             value=qcfg["default_shots"] if qcfg["default_shots"] in
                             [128, 256, 512, 1024, 2048, 4096, 8192, 16384] else 1024)
    n = c3.slider("Recent trials", 2, 12, int(qcfg["default_trials"]),
                  help="n trials → n(n−1)/2 circuits")
    if backend == "ibm" and not qcfg["ibm_token"]:
        st.warning("No IBM token configured — add one in Settings → Quantum first.")
    if st.form_submit_button("Run check", type="primary", icon=":material/play_arrow:"):
        try:
            job = lib.api("POST", "/api/quantum-check", json={"backend": backend, "shots": shots, "n_trials": n})
            st.session_state["qc_job"] = job["job_id"]
        except lib.APIError as exc:
            st.error(str(exc))

jobs = []
try:
    jobs = lib.get("/api/quantum-check")
except lib.APIError as exc:
    lib.api_down_banner(exc)
    st.stop()

job_id = st.session_state.get("qc_job") or (jobs[0]["id"] if jobs else None)
if jobs:
    job_id = st.selectbox("Job", [j["id"] for j in jobs], index=[j["id"] for j in jobs].index(job_id)
                          if job_id in [j["id"] for j in jobs] else 0,
                          format_func=lambda i: next(f"{lib.fmt_time(j['started'])} · {BACKENDS.get(j['backend'])} · "
                                                     f"{j['shots']} shots" for j in jobs if j["id"] == i))


def heat(m: np.ndarray, title: str, scale, zmin: float, zmax: float, col: int, fig) -> None:
    fig.add_trace(go.Heatmap(z=m, zmin=zmin, zmax=zmax, colorscale=scale, showscale=False,
                             hovertemplate="trial %{y} × %{x}: %{z:.3f}<extra></extra>"), row=1, col=col)
    fig.layout.annotations[col - 1].update(text=title)


@st.fragment(run_every=2.0)
def show(job_id: str) -> None:
    job = lib.get(f"/api/quantum-check/{job_id}")
    if job["status"] not in ("done", "error"):
        st.info(f"Job {job_id}: {job['status']}…", icon=":material/hourglass_top:")
        return
    if job["status"] == "error":
        st.error(job["error"])
        return
    exact = np.asarray(job["exact"])
    if "circuit" not in job:
        st.plotly_chart(lib.fig_layout(go.Figure(go.Heatmap(z=exact, colorscale="Viridis")), height=320),
                        width="stretch")
        return
    circ = np.asarray(job["circuit"])
    info = job["backend_info"]
    k = st.columns(5)
    k[0].metric("Circuits run", job["circuits_run"])
    k[1].metric("Transpiled depth", info["transpiled_depth"])
    k[2].metric("Mean |error|", f"{job['mean_abs_error']:.3f}")
    k[3].metric("Correlation", "—" if job.get("correlation") is None else f"{job['correlation']:.3f}")
    k[4].metric("Circuit time", f"{job['circuit_seconds']:.1f} s")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("a", "b", "c"), horizontal_spacing=0.06)
    viridis = [[0, "#12151C"], [1, d["accent_color"]]]
    heat(exact, "Exact (what the node uses)", viridis, 0, 1, 1, fig)
    heat(circ, f"Circuits · {info['backend']}", viridis, 0, 1, 2, fig)
    heat(circ - exact, "Difference", [[0, d["ok_color"]], [0.5, "#12151C"], [1, d["alert_color"]]], -0.5, 0.5, 3, fig)
    for c in (1, 2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=c)
    st.plotly_chart(lib.fig_layout(fig, height=320), width="stretch", config={"displayModeBar": False})
    st.html('<div class="callout"><b>How to read it.</b> Each cell is the fidelity |⟨φ(xᵢ)|φ(xⱼ)⟩|² between two '
            'trials. Finite shots add random scatter; hardware noise (decoherence over a depth-~110 circuit) pulls '
            'every value toward the fully-mixed baseline 1/16, which is why the error is systematic. A high '
            'correlation means the kernel still <i>ranks</i> similarity correctly, even when absolute values drift.</div>')


if job_id:
    show(job_id)
else:
    st.caption("No checks run yet in this API session.")
