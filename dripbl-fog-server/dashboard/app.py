"""DRIPBL Command Center — Streamlit entry point.

    .venv/bin/python3 -m streamlit run dashboard/app.py

Every number on every page comes from the API server (dashboard/lib.py), so the
dashboard can run on a different machine than the fog node. Set
DRIPBL_DASHBOARD_PASSWORD to require a password (do this before exposing it on a network).
"""
from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

st.set_page_config(page_title="DRIPBL Command Center", page_icon=":material/shield:", layout="wide")
lib.inject_css()

password = os.environ.get("DRIPBL_DASHBOARD_PASSWORD", "")
if password and not st.session_state.get("authed"):
    lib.header("DRIPBL Command Center", "Sign in to continue")
    with st.form("login", border=True, width=380):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            if hmac.compare_digest(pw, password):
                st.session_state.authed = True
                st.rerun()
            st.error("Wrong password")
    st.stop()

views = Path(__file__).resolve().parent / "views"
PAGES = [
    st.Page(views / "home.py", title="Command Center", icon=":material/radar:", default=True),
    st.Page(views / "alerts.py", title="Alerts", icon=":material/notification_important:", url_path="Alerts"),
    st.Page(views / "inspector.py", title="Pipeline Inspector", icon=":material/account_tree:", url_path="Inspector"),
    st.Page(views / "research.py", title="Research", icon=":material/science:", url_path="Research"),
    st.Page(views / "quantum.py", title="Quantum Lab", icon=":material/memory:", url_path="Quantum"),
    st.Page(views / "settings.py", title="Settings", icon=":material/tune:", url_path="Settings"),
]
hidden = set(lib.dash().get("hidden_pages", []))
nav = st.navigation([p for p in PAGES if p.title not in hidden or p.title == "Settings"])

with st.sidebar:
    status = lib.status_or_none()
    if status is None:
        st.html(lib.pill("API offline", "alert"))
    elif status["node_online"]:
        node = status["node"]
        st.html(lib.pill("Node live", "ok", live=True) + " " +
                lib.pill("Mock" if node["mode"] == "mock" else "Live feed", "accent"))
        st.caption(f"{node['model']}  \n{node['fps']:.0f} fps · {node['last_infer_ms']:.0f} ms/frame")
    else:
        st.html(lib.pill("Node offline", "warn"))
        st.caption("The API is up but the fog node isn't sending heartbeats.")

nav.run()
