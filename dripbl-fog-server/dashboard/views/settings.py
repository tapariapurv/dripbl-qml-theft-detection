"""Settings: every node tunable (hot-reloaded by the fog node within ~1 s) and every
command-center preference, each with its default shown and a reset."""
from __future__ import annotations

import io
import json
import zoneinfo

import streamlit as st

import lib

lib.header("Settings", "Node settings apply live within a second. Dashboard settings apply on the next refresh.")

try:
    node = lib.get("/api/config")
    node_def = lib.get("/api/config/defaults")
    app = lib.get("/api/app-settings")
    app_def = lib.get("/api/app-settings/defaults")
except lib.APIError as exc:
    lib.api_down_banner(exc)
    st.stop()


def save_node(patch: dict) -> None:
    changed = {k: v for k, v in patch.items() if node.get(k) != v}
    if not changed:
        st.toast("Nothing changed")
        return
    try:
        lib.api("PUT", "/api/config", json=changed)
        st.toast(f"Saved {len(changed)} node setting(s) — the node applies them within ~1 s", icon=":material/done:")
        st.rerun()
    except lib.APIError as exc:
        st.error(str(exc))


def save_app(section: str, values: dict) -> None:
    try:
        lib.api("PUT", "/api/app-settings", json={section: values})
        lib.clear_caches()
        st.toast("Saved", icon=":material/done:")
        st.rerun()
    except lib.APIError as exc:
        st.error(str(exc))


def reset_app(section: str) -> None:
    """Button callback (Streamlit reruns after it on its own)."""
    try:
        lib.api("POST", "/api/app-settings/reset", params={"section": section})
        lib.clear_caches()
        st.toast(f"{section} reset to defaults")
    except lib.APIError as exc:
        st.toast(str(exc), icon=":material/error:")


def dflt(value) -> str:
    return f"Default: {value}"


tabs = st.tabs([":material/visibility: Detection", ":material/videocam: Source & mode", ":material/gavel: AI judge",
                ":material/notifications: Notifications", ":material/dashboard: Dashboard",
                ":material/memory: Quantum", ":material/settings: System & API", ":material/data_object: Advanced"])

# --- Detection -----------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1.2, 1], gap="large")
    with left, st.form("detection"):
        st.markdown("##### Sensitivity")
        sens = st.slider("Sensitivity", 0.0, 1.0, float(node["sensitivity"]), 0.01, help=dflt(node_def["sensitivity"]) +
                         ". Higher = flags more readily (lower judge threshold, smaller motion needed).")
        st.caption(f"→ flag threshold **{1 - sens:.2f}** · motion area gate "
                   f"**{node['min_contour_area'] * (1.6 - 1.2 * sens):.0f} px²**")
        st.markdown("##### Motion gate")
        c1, c2 = st.columns(2)
        debounce = c1.number_input("Debounce (s)", 0.0, 3600.0, float(node["debounce_sec"]), 0.5,
                                   help=dflt(node_def["debounce_sec"]) + ". Minimum gap between trials.")
        area = c2.number_input("Base contour area (px²)", 1.0, 1e6, float(node["min_contour_area"]), 50.0,
                               help=dflt(node_def["min_contour_area"]))
        hist = c1.number_input("MOG2 history (frames)", 1, 10000, int(node["mog2_history"]),
                               help=dflt(node_def["mog2_history"]))
        var = c2.number_input("MOG2 variance threshold", 1.0, 1000.0, float(node["mog2_var_threshold"]),
                              help=dflt(node_def["mog2_var_threshold"]))
        st.markdown("##### Camera site profile")
        profiles = {"default": "Default — general model", "cctv": "CCTV — wide surveillance scenes",
                    "retail": "Retail — store / counter camera (retail expert model)"}
        prof = st.selectbox("Site profile", list(profiles), format_func=profiles.get,
                            index=list(profiles).index(node["site_profile"]),
                            help="Picks the model and decision threshold trained on footage like yours. "
                                 "Switching applies live; the active model shows in the sidebar.")
        st.markdown("##### Clip-level judgement")
        st.caption("While there is motion the model scores frames continuously; a trial is judged on the "
                   "mean over the last window, which is how its accuracy was measured.")
        c1, c2 = st.columns(2)
        sfps = c1.number_input("Scoring rate (frames/s)", 0.5, 30.0, float(node["score_fps"]), 0.5,
                               help=dflt(node_def["score_fps"]) + ". Lower it on a slow Pi.")
        jwin = c2.number_input("Judgement window (s)", 0.25, 30.0, float(node["judge_window_sec"]), 0.25,
                               help=dflt(node_def["judge_window_sec"]))
        st.markdown("##### AI judge weights · trained model")
        st.caption("suspicion = w_crime·crime + w_nervous·jitter + w_speed·speed_anomaly")
        w1, w2, w3 = st.columns(3)
        wc = w1.number_input("Crime signal", 0.0, 1.0, float(node["w_crime"]), 0.01, help=dflt(node_def["w_crime"]))
        wn = w2.number_input("Nervousness", 0.0, 1.0, float(node["w_nervous"]), 0.01, help=dflt(node_def["w_nervous"]))
        ws = w3.number_input("Speed", 0.0, 1.0, float(node["w_speed"]), 0.01, help=dflt(node_def["w_speed"]))
        if abs(wc + wn + ws - 1) > 1e-6:
            st.caption(f":orange[Weights sum to {wc + wn + ws:.2f}; 1.00 keeps suspicion on the 0–1 scale.]")
        with st.expander("Fallback weights (prototype classifier, used only without a trained model)"):
            f1, f2, f3, f4 = st.columns(4)
            fwc = f1.number_input("Crime", 0.0, 1.0, float(node["fb_w_crime"]), 0.01)
            fwn = f2.number_input("Nervous", 0.0, 1.0, float(node["fb_w_nervous"]), 0.01)
            fws = f3.number_input("Speed", 0.0, 1.0, float(node["fb_w_speed"]), 0.01)
            fwb = f4.number_input("Person bonus", 0.0, 1.0, float(node["fb_w_person_bonus"]), 0.01)
        nsp = st.number_input("Speed counted as fully anomalous (px/s)", 1.0, 1e5, float(node["nervous_speed_px_s"]),
                              10.0, help=dflt(node_def["nervous_speed_px_s"]) + ". Depends on camera distance.")
        if st.form_submit_button("Save detection settings", type="primary"):
            save_node({"site_profile": prof, "score_fps": sfps, "judge_window_sec": jwin, "sensitivity": sens, "debounce_sec": debounce, "min_contour_area": area, "mog2_history": hist,
                       "mog2_var_threshold": var, "w_crime": wc, "w_nervous": wn, "w_speed": ws, "fb_w_crime": fwc,
                       "fb_w_nervous": fwn, "fb_w_speed": fws, "fb_w_person_bonus": fwb, "nervous_speed_px_s": nsp})
    with right:
        st.markdown("##### Region of interest")
        x, y, w, h = node["roi"]
        fw, fh = int(node["frame_width"]), int(node["frame_height"])
        with st.form("roi"):
            c1, c2 = st.columns(2)
            nx = c1.number_input("x", 0, 10000, int(x))
            ny = c2.number_input("y", 0, 10000, int(y))
            nw = c1.number_input("width", 1, 10000, int(w))
            nh = c2.number_input("height", 1, 10000, int(h))
            b1, b2 = st.columns(2)
            preview = b1.form_submit_button("Preview", icon=":material/preview:", width="stretch")
            if b2.form_submit_button("Save ROI", type="primary", width="stretch"):
                save_node({"roi": [nx, ny, nw, nh]})
        try:
            from PIL import Image, ImageDraw
            img = Image.open(io.BytesIO(lib.api("GET", "/api/frame.jpg"))).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle([nx, ny, nx + nw, ny + nh], outline=lib.dash()["warn_color"], width=3)
            st.image(img, caption="Orange = ROI being edited (the live frame already shows the saved one in green)",
                     width="stretch")
        except lib.APIError:
            st.caption("Start the node to preview the ROI on a live frame.")

        st.markdown("##### Calibrate to this camera")
        mode_now = node["threshold_mode"]
        st.caption("Every camera's normal activity looks a little different. Calibration learns the alert "
                   "threshold from a period of **normal** activity on this camera, so everyday movement stops "
                   "raising alerts. Only use a period with no theft in it.")
        st.html(lib.pill(f"threshold: {'calibrated ' + format(node['calibrated_threshold'], '+.3f') if mode_now == 'calibrated' else 'model default'}",
                         "accent" if mode_now == "calibrated" else ""))
        with st.form("calibrate"):
            c1, c2 = st.columns(2)
            minutes = c1.number_input("Use the last (minutes)", 1.0, 10080.0, 30.0, 5.0)
            quant = c2.slider("Stay above this share of normal windows", 0.80, 1.0, 0.98, 0.01)
            b1, b2 = st.columns(2)
            preview_cal = b1.form_submit_button("Preview", width="stretch")
            apply_cal = b2.form_submit_button("Calibrate & apply", type="primary", width="stretch")
            if preview_cal or apply_cal:
                try:
                    r = lib.api("POST", "/api/calibrate", json={"minutes": minutes, "quantile": quant, "apply": apply_cal})
                    prev = r["normal_windows_above_previous"]
                    st.success(f"{r['windows']} normal windows → threshold {r['threshold']:+.3f}. Normal windows that "
                               f"would alert: {'—' if prev is None else f'{prev:.0%}'} before, "
                               f"{r['normal_windows_above_new']:.0%} after." + (" Applied." if r["applied"] else ""))
                except lib.APIError as exc:
                    st.error(str(exc))
        if mode_now == "calibrated" and st.button("Use the model's own threshold again"):
            save_node({"threshold_mode": "model"})

# --- Source & mode ------------------------------------------------------------------------------
with tabs[1], st.form("source"):
    mode = st.segmented_control("Mode", ["live", "mock"], default=node["mode"],
                                format_func={"live": "Live camera / stream / file",
                                             "mock": "Mock (labelled research clips)"}.get)
    st.markdown("##### Live source")
    c1, c2 = st.columns([2, 1])
    src = c1.text_input("Camera source", str(node["camera_source"]),
                        help="Camera index (0, 1…), rtsp://… / http(s)://… URL, or a video file path")
    kind = c2.selectbox("Source kind", ["auto", "camera", "stream", "file"],
                        index=["auto", "camera", "stream", "file"].index(node["source_kind"]))
    c1, c2, c3 = st.columns(3)
    fw = c1.number_input("Frame width", 64, 7680, int(node["frame_width"]))
    fh = c2.number_input("Frame height", 64, 4320, int(node["frame_height"]))
    loop = c3.toggle("Loop video files", node["loop_file"])
    st.markdown("##### Mock mode")
    c1, c2 = st.columns([2, 1])
    mdir = c1.text_input("Mock clip folder", node["mock_dir"], help="Must contain normal/*.mp4 and shoplifting/*.mp4")
    mclip = c2.number_input("Max seconds per clip", 1.0, 3600.0, float(node["mock_clip_sec"]))
    mhold = st.toggle("Replay only hold-out clips (honest scores)", node["mock_holdout_only"],
                      help="The retail model trained on the other dataset-1 clips; replaying those would "
                           "inflate the Research scoreboard.")
    if st.form_submit_button("Save source settings", type="primary"):
        save_node({"mode": mode or node["mode"], "camera_source": int(src) if src.strip().isdigit() else src.strip(),
                   "source_kind": kind, "frame_width": fw, "frame_height": fh, "loop_file": loop,
                   "mock_dir": mdir, "mock_clip_sec": mclip, "mock_holdout_only": mhold})

# --- AI judge ------------------------------------------------------------------------------------
with tabs[2]:
    aj, ajd = app["ai_judge"], app_def["ai_judge"]
    st.markdown("The rule-based judge always runs on the node (fast, offline). Optionally each **flagged** trial "
                "also gets a Claude second opinion with a written reason, shown on the alert.")
    with st.form("ai_judge"):
        jm = st.segmented_control("Judge mode", ["rules", "rules+claude"], default=aj["mode"],
                                  format_func={"rules": "Rules only", "rules+claude": "Rules + Claude review"}.get)
        c1, c2 = st.columns(2)
        model = c1.text_input("Model", aj["model"], help=dflt(ajd["model"]))
        effort = c2.selectbox("Effort", ["low", "medium", "high"], index=["low", "medium", "high"].index(aj["effort"]))
        c1, c2 = st.columns(2)
        fb = c1.toggle("Server-side refusal fallbacks", aj["use_fallbacks"])
        snap = c2.toggle("Send the flagged frame", aj["send_snapshot"])
        key = st.text_input("Anthropic API key", aj["api_key"], type="password",
                            help="Blank = use ANTHROPIC_API_KEY or an `ant auth login` profile on the API host")
        extra = st.text_area("Extra instructions for the reviewer", aj["extra_instructions"],
                             placeholder="e.g. Staff wear blue vests; the back door is staff-only.")
        if st.form_submit_button("Save AI judge settings", type="primary"):
            save_app("ai_judge", {"mode": jm or aj["mode"], "model": model, "effort": effort, "use_fallbacks": fb,
                                  "send_snapshot": snap, "api_key": key, "extra_instructions": extra})
    st.button("Reset AI judge settings", on_click=reset_app, args=("ai_judge",))

# --- Notifications --------------------------------------------------------------------------------
with tabs[3]:
    n = app["notifications"]
    with st.form("notify"):
        en = st.toggle("Send notifications for flagged trials", n["enabled"])
        c1, c2, c3 = st.columns(3)
        mins = c1.slider("Minimum suspicion", 0.0, 1.0, float(n["min_suspicion"]), 0.05)
        cool = c2.number_input("Cooldown (s)", 0.0, 86400.0, float(n["cooldown_sec"]), 5.0)
        agree = c3.toggle("Only if Claude agrees", n["only_if_ai_agrees"], help="Needs Rules + Claude mode")
        durl = st.text_input("Dashboard URL (used in links)", n["dashboard_url"])
        st.markdown("##### Phone push · ntfy")
        c1, c2, c3, c4 = st.columns([0.7, 1.4, 1.2, 1])
        ne = c1.toggle("On", n["ntfy"]["enabled"], key="ntfy_on")
        nserver = c2.text_input("Server", n["ntfy"]["server"])
        ntopic = c3.text_input("Topic", n["ntfy"]["topic"], help="Subscribe to the same topic in the ntfy app")
        nprio = c4.selectbox("Priority", ["min", "low", "default", "high", "urgent"],
                             index=["min", "low", "default", "high", "urgent"].index(n["ntfy"]["priority"]))
        st.markdown("##### Webhook · Slack / Discord / any HTTP endpoint")
        c1, c2, c3 = st.columns([0.7, 2.6, 1])
        we = c1.toggle("On", n["webhook"]["enabled"], key="hook_on")
        wurl = c2.text_input("URL", n["webhook"]["url"])
        wfmt = c3.selectbox("Format", ["json", "slack", "discord"], index=["json", "slack", "discord"].index(n["webhook"]["format"]))
        st.markdown("##### Email · SMTP")
        m = n["email"]
        c1, c2, c3, c4 = st.columns([0.7, 2, 0.8, 0.8])
        me = c1.toggle("On", m["enabled"], key="mail_on")
        host = c2.text_input("SMTP host", m["smtp_host"])
        port = c3.number_input("Port", 1, 65535, int(m["smtp_port"]))
        tls = c4.toggle("STARTTLS", m["use_tls"])
        c1, c2 = st.columns(2)
        user = c1.text_input("Username", m["username"])
        pw = c2.text_input("Password", m["password"], type="password")
        sender = c1.text_input("From", m["sender"])
        rcpt = c2.text_input("To (comma-separated)", m["recipients"])
        if st.form_submit_button("Save notification settings", type="primary"):
            save_app("notifications", {
                "enabled": en, "min_suspicion": mins, "cooldown_sec": cool, "only_if_ai_agrees": agree,
                "dashboard_url": durl,
                "ntfy": {"enabled": ne, "server": nserver, "topic": ntopic, "priority": nprio},
                "webhook": {"enabled": we, "url": wurl, "format": wfmt},
                "email": {"enabled": me, "smtp_host": host, "smtp_port": port, "use_tls": tls, "username": user,
                          "password": pw, "sender": sender, "recipients": rcpt}})
    c1, c2 = st.columns([1, 3])
    if c1.button("Send test notification", icon=":material/send:"):
        try:
            res = lib.api("POST", "/api/notify/test", timeout=30)
            for ch, r in res.items():
                (st.success if r == "ok" else st.error)(f"{ch}: {r}")
        except lib.APIError as exc:
            st.error(str(exc))
    c2.button("Reset notification settings", on_click=reset_app, args=("notifications",))

# --- Dashboard -------------------------------------------------------------------------------------
with tabs[4]:
    dsh, dd = app["dashboard"], app_def["dashboard"]
    TILES = {"node": "Node status", "camera": "Camera", "mode": "Mode", "fps": "Throughput", "latency": "Inference time",
             "trials": "Trials today", "flagged": "Flagged today", "pending": "Awaiting review",
             "threshold": "Flag threshold", "gait": "Live gait"}
    PAGES = ["Command Center", "Alerts", "Pipeline Inspector", "Research", "Quantum Lab"]
    with st.form("dash"):
        st.markdown("##### Branding")
        c1, c2 = st.columns(2)
        title = c1.text_input("Title", dsh["title"])
        subtitle = c2.text_input("Subtitle", dsh["subtitle"])
        c = st.columns(4)
        accent = c[0].color_picker("Accent", dsh["accent_color"], help=dflt(dd["accent_color"]))
        alert = c[1].color_picker("Alert", dsh["alert_color"], help=dflt(dd["alert_color"]))
        ok = c[2].color_picker("OK", dsh["ok_color"], help=dflt(dd["ok_color"]))
        warn = c[3].color_picker("Warning", dsh["warn_color"], help=dflt(dd["warn_color"]))
        st.markdown("##### Layout")
        tiles = st.multiselect("Status tiles (in order)", list(TILES), dsh["tiles"], format_func=TILES.get)
        hidden = st.multiselect("Hide pages", PAGES, dsh["hidden_pages"])
        c = st.columns(4)
        show_video = c[0].toggle("Show live video", dsh["show_video"])
        vwidth = c[1].number_input("Video max width (px)", 200, 4000, int(dsh["video_max_width_px"]), 50)
        chart_h = c[2].number_input("Chart height (px)", 150, 1200, int(dsh["chart_height_px"]), 10)
        inbox = c[3].number_input("Alerts in home inbox", 1, 50, int(dsh["inbox_size"]))
        c = st.columns(4)
        refresh = c[0].number_input("Refresh every (s)", 0.5, 60.0, float(dsh["refresh_sec"]), 0.5,
                                    help="Lower = snappier but more load on a Pi")
        tl = c[1].number_input("Timeline window (min)", 1, 1440, int(dsh["timeline_minutes"]))
        psize = c[2].number_input("Table page size", 10, 1000, int(dsh["table_page_size"]), 10)
        dec = c[3].number_input("Score decimals", 0, 6, int(dsh["suspicion_decimals"]))
        c = st.columns(2)
        gt = c[0].toggle("Show ground truth (mock mode)", dsh["show_ground_truth"])
        thr = c[1].toggle("Draw thresholds on charts", dsh["show_thresholds"])
        st.markdown("##### Time")
        zones = sorted(zoneinfo.available_timezones())
        c = st.columns(3)
        tzname = c[0].selectbox("Timezone", zones, index=zones.index(dsh["timezone"]) if dsh["timezone"] in zones else 0)
        tfmt = c[1].text_input("Time format", dsh["time_format"], help="strftime, e.g. %H:%M:%S or %I:%M %p")
        dfmt = c[2].text_input("Date format", dsh["date_format"], help="strftime, e.g. %d %b %Y or %Y-%m-%d")
        if st.form_submit_button("Save dashboard settings", type="primary"):
            save_app("dashboard", {
                "title": title, "subtitle": subtitle, "accent_color": accent, "alert_color": alert, "ok_color": ok,
                "warn_color": warn, "tiles": tiles, "hidden_pages": hidden, "show_video": show_video,
                "video_max_width_px": vwidth, "chart_height_px": chart_h, "inbox_size": inbox,
                "refresh_sec": refresh, "timeline_minutes": tl, "table_page_size": psize,
                "suspicion_decimals": dec, "show_ground_truth": gt, "show_thresholds": thr,
                "timezone": tzname, "time_format": tfmt, "date_format": dfmt})
    st.button("Reset dashboard settings", on_click=reset_app, args=("dashboard",))

# --- Quantum ----------------------------------------------------------------------------------------
with tabs[5]:
    qc = app["quantum"]
    with st.form("quantum"):
        c1, c2 = st.columns(2)
        qt = c1.number_input("Default trials per check", 2, 12, int(qc["default_trials"]))
        qs = c2.number_input("Default shots", 16, 100000, int(qc["default_shots"]))
        st.markdown("##### IBM Quantum (real hardware)")
        token = st.text_input("API token", qc["ibm_token"], type="password")
        c1, c2 = st.columns(2)
        inst = c1.text_input("Instance (CRN, optional)", qc["ibm_instance"])
        bk = c2.text_input("Backend (blank = least busy)", qc["ibm_backend"])
        if st.form_submit_button("Save quantum settings", type="primary"):
            save_app("quantum", {"default_trials": qt, "default_shots": qs, "ibm_token": token,
                                 "ibm_instance": inst, "ibm_backend": bk})

# --- System & API -------------------------------------------------------------------------------------
with tabs[6]:
    a, b = st.columns(2, gap="large")
    with a, st.form("system"):
        st.markdown("##### Fog node")
        c1, c2 = st.columns(2)
        lfps = c1.number_input("Live frame publish rate (fps)", 0.0, 30.0, float(node["live_frame_fps"]), 1.0,
                               help="0 disables the live video (saves CPU on a Pi)")
        lq = c2.number_input("Live frame JPEG quality", 10, 100, int(node["live_jpeg_quality"]))
        hb = c1.number_input("Heartbeat (s)", 0.2, 60.0, float(node["heartbeat_sec"]), 0.1)
        model_path = c2.text_input("Model artifact", node["model_path"], help="Restart the node after changing")
        st.markdown("##### Hardware alert (GPIO)")
        c1, c2 = st.columns(2)
        bz = c1.number_input("Buzzer pin (BCM)", 0, 40, int(node["buzzer_pin"]))
        led = c2.number_input("LED pin (BCM)", 0, 40, int(node["led_pin"]))
        st.markdown("##### Extra POST per flagged trial")
        aep = st.text_input("Alert endpoint", node["alert_endpoint"], help="Optional; blank = off")
        c1, c2, c3 = st.columns(3)
        thumb = c1.toggle("Include thumbnail", node["send_thumbnail"])
        tq = c2.number_input("Thumbnail quality", 10, 100, int(node["thumbnail_jpeg_quality"]))
        tout = c3.number_input("Timeout (s)", 0.1, 60.0, float(node["request_timeout_sec"]))
        if st.form_submit_button("Save node system settings", type="primary"):
            save_node({"live_frame_fps": lfps, "live_jpeg_quality": lq, "heartbeat_sec": hb, "model_path": model_path,
                       "buzzer_pin": bz, "led_pin": led, "alert_endpoint": aep, "send_thumbnail": thumb,
                       "thumbnail_jpeg_quality": tq, "request_timeout_sec": tout})
    with b:
        with st.form("api"):
            st.markdown("##### API")
            rk = st.toggle("Require API key for reads too", app["api"]["require_key_for_reads"],
                           help="Writes always need it. Turn on before exposing the API beyond this machine.")
            cors = st.text_input("CORS origins (comma-separated)", ", ".join(app["api"]["cors_origins"]),
                                 help="Which web apps may call the API from a browser. Applies after an API restart.")
            st.markdown("##### Retention")
            days = st.number_input("Delete trials older than (days, 0 = never)", 0, 36500, int(app["retention"]["days"]))
            if st.form_submit_button("Save API & retention", type="primary"):
                try:
                    lib.api("PUT", "/api/app-settings", json={
                        "api": {"require_key_for_reads": rk,
                                "cors_origins": [o.strip() for o in cors.split(",") if o.strip()]},
                        "retention": {"days": days}})
                    lib.clear_caches()
                    st.toast("Saved", icon=":material/done:")
                    st.rerun()
                except lib.APIError as exc:
                    st.error(str(exc))
        with st.container(border=True):
            st.markdown("##### Build on the API")
            st.markdown(f"Interactive docs: [{lib.browser_api_url()}/docs]({lib.browser_api_url()}/docs)")
            st.code(f'curl -H "X-API-Key: $DRIPBL_API_KEY" {lib.browser_api_url()}/api/trials?flagged=true', language="bash")
            if st.toggle("Reveal API key"):
                st.code(lib._key() or "(not available on this machine)", language=None)

# --- Advanced: raw JSON -----------------------------------------------------------------------------------
with tabs[7]:
    st.caption("Edit the full JSON directly. Unknown keys and wrong types are rejected by the API, "
               "so a typo can't break the node.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Node · runtime_config.json")
        raw = st.text_area("node json", json.dumps({k: v for k, v in node.items() if k != "derived"}, indent=2),
                           height=480, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        if c1.button("Apply node JSON", type="primary"):
            try:
                save_node(json.loads(raw))
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
        if c2.button("Reset node to defaults"):
            try:
                lib.api("POST", "/api/config/reset")
                st.rerun()
            except lib.APIError as exc:
                st.error(str(exc))
    with b:
        st.markdown("##### Command center · app_settings.json")
        raw_app = st.text_area("app json", json.dumps(app, indent=2), height=480, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        if c1.button("Apply app JSON", type="primary"):
            try:
                lib.api("PUT", "/api/app-settings", json=json.loads(raw_app))
                lib.clear_caches()
                st.rerun()
            except (json.JSONDecodeError, lib.APIError) as exc:
                st.error(str(exc))
        if c2.button("Reset everything to defaults"):
            try:
                lib.api("POST", "/api/app-settings/reset")
                lib.clear_caches()
                st.rerun()
            except lib.APIError as exc:
                st.error(str(exc))
