"""Research: the offline results (from reports/ via the API) and the live mock-mode scoreboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

d = lib.dash()
lib.header("Research", "Offline benchmark results and live mock-mode evaluation")


@st.cache_data(ttl=300, show_spinner=False)
def report(name: str):
    try:
        return lib.get(f"/api/research/{name}")
    except lib.APIError:
        return None


def confusion(cm: dict, title: str, height: int = 260) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted normal", "Predicted theft"], y=["Actual normal", "Actual theft"],
        text=[[str(v) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=20),
        colorscale=[[0, "#12151C"], [1, d["accent_color"]]], showscale=False,
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return lib.fig_layout(fig, height=height, title=dict(text=title, font=dict(size=13)))


def rates(cm: dict) -> dict[str, float | None]:
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = tn + fp + fn + tp
    return {
        "accuracy": (tp + tn) / n if n else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false positive rate": fp / (fp + tn) if fp + tn else None,
    }


tab_model, tab_xd, tab_mock, tab_hw, tab_exp = st.tabs(
    [":material/verified: Deployed model", ":material/public: Cross-dataset", ":material/play_circle: Mock-mode scoreboard",
     ":material/memory: Hardware feasibility", ":material/lab_panel: Experiment log"])

NAMES = {"dataset3_test": "UCF-Crime Test (CCTV)", "dataset1_holdout": "Retail hold-out (dataset-1)",
         "dataset2_holdout": "Second store camera (dataset-2)"}

with tab_xd:
    st.markdown("Each camera domain runs its own expert, chosen by the node's **site profile**. Every number "
                "below is measured once on videos no model or threshold ever saw.")
    experts = [(slot, report(f"deployed{'' if slot == 'default' else '_' + slot}")) for slot in ("default", "retail")]
    experts = [(slot, e) for slot, e in experts if e]
    if not experts:
        st.info("No cross-dataset expert deployed yet (src/train_xd.py, then src/deploy_model.py).")
    for slot, e in experts:
        st.markdown(f"##### {'Retail expert' if slot == 'retail' else 'Default / CCTV model'} · `{e.get('experiment')}`")
        res = e.get("holdout_site_calibrated") or e.get("holdout") or {}
        cols = st.columns(len(res) or 1)
        for col, (name, r) in zip(cols, res.items()):
            col.metric(NAMES.get(name, name), f"{r['accuracy']:.1%}",
                       help=f"AUC {r['auc']:.3f} · recall {r['recall']:.1%} · {r['clips']} clips · fp {r['fp']} fn {r['fn']}")
        oof = (e.get("training") or {}).get("oof_clip_auc")
        if oof:
            st.caption("Out-of-fold clip AUC on training videos: " + ", ".join(f"{k} {v:.3f}" for k, v in oof.items()))
    st.caption("Before this work the single deployed model scored 87.1% on UCF-Crime but 47.2% on retail "
               "(recall 0) and 50% on the second store camera.")

with tab_model:
    m = report("metrics")
    if not m:
        st.info("reports/metrics.json not found — run `src/train_final.py`.")
    else:
        c = m["clip"]
        st.caption(f"{m['model']} · {m['protocol']}")
        k = st.columns(6)
        k[0].metric("Clip accuracy", f"{c['accuracy']:.1%}", help=f"{c['n_clips']} unseen test clips")
        k[1].metric("ROC AUC", f"{c['roc_auc']:.3f}")
        k[2].metric("Balanced acc.", f"{c['balanced_accuracy']:.1%}")
        k[3].metric("Recall", f"{c['recall']:.1%}", help="Share of theft clips caught")
        k[4].metric("Precision", f"{c['precision']:.1%}", help="Share of alerts that were theft")
        k[5].metric("Frame accuracy", f"{m['frame_accuracy']:.1%}", help=f"{m['n_test_frames']:,} frames")
        a, b = st.columns([1, 1.2], gap="large")
        a.plotly_chart(confusion(c["confusion_matrix"], "Test clips (dataset-3, UCF-Crime)"), width="stretch",
                       config={"displayModeBar": False})
        with b:
            st.html(f'<div class="callout"><b>Reading this.</b> Train and Test share zero source videos, so this '
                    f'is a real generalisation test. The oracle threshold would reach '
                    f'{c.get("oracle_threshold_accuracy", 0):.1%} — only about a point above what the tuned '
                    f'threshold achieves, so calibration is close to exhausted: a better representation '
                    f'(temporal models) is what would move the number, not more tuning.</div>')
            sl = report("shoplifting")
            if sl:
                st.markdown("##### Second dataset · retail shoplifting (dataset-1)")
                s = st.columns(3)
                s[0].metric("Clip accuracy", f"{sl['clip_accuracy']:.1%}")
                s[1].metric("ROC AUC", f"{sl['roc_auc']:.3f}")
                s[2].metric("Recall", f"{sl['recall']:.1%}")
                st.caption(sl["protocol"])

with tab_mock:
    st.markdown("Mock mode replays labelled **dataset-1** clips through the live pipeline, so every trial "
                "carries ground truth. This is a *per-trial* score of the full live system (motion gate + "
                "classifier + judge + threshold), not the clip-level benchmark.")
    try:
        stats = lib.get("/api/stats")
    except lib.APIError as exc:
        lib.api_down_banner(exc)
        st.stop()
    cm = stats["mock_confusion"]
    if sum(cm.values()) == 0:
        st.info("No mock-mode trials yet. Switch to mock in Settings → Source & mode, or run `start.py --mock`.")
    else:
        r = rates(cm)
        k = st.columns(5)
        k[0].metric("Scored trials", f"{sum(cm.values()):,}")
        for col, (name, val) in zip(k[1:], r.items()):
            col.metric(name.capitalize(), "—" if val is None else f"{val:.1%}")
        a, b = st.columns([1, 1.2], gap="large")
        a.plotly_chart(confusion(cm, "Live pipeline vs ground truth"), width="stretch",
                       config={"displayModeBar": False})
        with b:
            st.html('<div class="callout"><b>Why this can be lower than the benchmark.</b> The deployed model was '
                    'trained on UCF-Crime (dataset-3) plus dataset-2. dataset-1 is a different camera and store, '
                    'so this scoreboard measures <i>cross-source transfer</i> — the weakness the research log '
                    'already identified (a dataset-3-only model scored chance on unseen footage). Tune the '
                    'sensitivity in Settings and watch recall and false positives trade off here.</div>')

with tab_hw:
    hw = report("hardware")
    if not hw:
        st.info("reports/hardware_report.json not found — run `src/hardware_report.py`.")
    else:
        circ, cost = hw["circuit"], hw["cost"]
        k = st.columns(5)
        k[0].metric("Logical qubits", circ["logical_qubits"])
        k[1].metric("Transpiled depth", circ.get("transpiled_depth", "—"), help=f"on {circ.get('backend')}")
        k[2].metric("2-qubit gates", circ.get("two_qubit_gates", "—"))
        k[3].metric("Circuits / frame", f"{cost['circuits_per_frame']:,}", help="one per support vector")
        k[4].metric("Device", f"{circ.get('device_qubits', '—')} qubits")
        acc = hw["accuracy_vs_shots"]
        shots = [s for s in acc if s != "exact"]
        fig = go.Figure()
        fig.add_scatter(x=[int(s) for s in shots], y=[acc[s]["clip_accuracy"] for s in shots], name="clip accuracy",
                        mode="lines+markers", line=dict(color=d["accent_color"], width=3))
        fig.add_scatter(x=[int(s) for s in shots], y=[acc[s]["roc_auc"] for s in shots], name="ROC AUC",
                        mode="lines+markers", line=dict(color=d["warn_color"], width=2, dash="dash"))
        fig.add_hline(y=acc["exact"]["clip_accuracy"], line=dict(color=d["ok_color"], dash="dot"),
                      annotation_text=f"exact statevector {acc['exact']['clip_accuracy']:.1%} (deployed)",
                      annotation_font_color=d["ok_color"])
        fig.add_hline(y=0.5, line=dict(color="#5B6275", dash="dot"), annotation_text="chance")
        fig.update_xaxes(type="log", title="measurement shots per kernel entry")
        fig.update_yaxes(range=[0.45, 0.95])
        a, b = st.columns([1.5, 1], gap="large")
        a.plotly_chart(lib.fig_layout(fig, height=340, title=dict(text="Accuracy vs shot budget", font=dict(size=13))),
                       width="stretch", config={"displayModeBar": False})
        b.html(f'<div class="callout"><b>Takeaway.</b> On real hardware each kernel entry is estimated from finite '
               f'shots. Classifying one frame needs {cost["circuits_per_frame"]:,} circuits, and accuracy only '
               f'recovers to {acc["16384"]["clip_accuracy"]:.0%} at 16,384 shots each. Exact statevector '
               f'simulation on the fog node is the production path; QPUs are for periodic verification '
               f'(see Quantum Lab).</div>')

with tab_exp:
    exps = report("experiments")
    if not exps:
        st.info("reports/experiments.jsonl not found.")
    else:
        df = pd.DataFrame([{
            "run": e["config"].get("name"),
            "clip acc.": e["clip"]["accuracy"], "clip AUC": e["clip"]["roc_auc"],
            "recall": e["clip"]["recall"], "frame acc.": e.get("frame_accuracy"),
            "projector": e.get("projector_mode"), "angle scale": e.get("angle_scale"), "C": e.get("C"),
            "epochs": e["config"].get("epochs"), "frames/video": e["config"].get("frames_per_video"),
            "seeds": len(e["config"].get("seeds", [])), "minutes": (e.get("runtime_sec") or 0) / 60,
        } for e in exps])
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ("clip acc.", "clip AUC", "recall", "frame acc.")
        } | {"minutes": st.column_config.NumberColumn(format="%.0f")})
        st.markdown("##### Documented design decisions")
        st.dataframe(pd.DataFrame([
            ("PCA → supervised PLS for the 4-d bottleneck", 0.887, "kept", "+2.6 pts accuracy"),
            ("3-to-5 member ensemble (mean margin)", 0.912, "kept", "variance down, circuit depth unchanged"),
            ("Mix in dataset-2 (cross-source)", 0.871, "kept", "holdout clips 0/2 → 2/2"),
            ("RGB + inter-frame difference", 0.875, "rejected", "frames 0.33 s apart: difference = scene drift"),
            ("45 frames/clip, 16 epochs", 0.869, "rejected", "memorises scenes"),
            ("MIL ranking (Sultani)", 0.646, "rejected", "needs precomputed 3D-CNN features"),
            ("Frozen ImageNet backbone", 0.711, "rejected", "collapses to 2×2 at 64×64"),
            ("dataset-4 temporal masks", 0.841, "rejected", "model has no within-clip temporal resolution"),
        ], columns=["change", "clip AUC", "verdict", "why"]), hide_index=True, width="stretch")
