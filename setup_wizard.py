"""
setup_wizard.py — First-run setup wizard.

State machine:
    admin       → no users in DB
    camera      → no primary camera selected yet
    extras      → primary set, asking if user wants live-only cameras
    discord     → optional Discord webhook
    done        → flag setup_complete=true and bounce to dashboard

Validation: each step writes to the DB only after type-specific fields pass
validation, so a half-finished form never leaves the DB in a broken state.
A self-heal in /setup/ flips setup_complete=true if every step's data is
present but the flag never landed.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

from flask import (
    Blueprint, render_template, redirect, url_for, request,
    flash, session, jsonify,
)

import config
import models
from auth import hash_password, login_user
from database import get_db, set_setting, get_setting
from utils import audit, get_client_ip

logger = logging.getLogger(__name__)

setup_bp = Blueprint("setup", __name__, url_prefix="/setup")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def setup_complete() -> bool:
    try:
        return get_setting("setup_complete", "").strip().lower() == "true"
    except Exception:
        return False


def _has_users() -> bool:
    try:
        db = get_db()
        return bool(models.get_all_users(db))
    except Exception:
        return False


def _has_primary_camera() -> bool:
    return bool(get_setting("camera_preferred_source", ""))


def _extras_decided() -> bool:
    """User has either added an extra camera or explicitly said 'no extras'."""
    return get_setting("setup_extras_done", "").strip().lower() == "true"


def _discord_decided() -> bool:
    if get_setting("setup_skip_discord", "").strip().lower() == "true":
        return True
    return bool(get_setting("discord_webhook_url", ""))


def _wizard_state() -> str:
    if not _has_users():
        return "admin"
    if not _has_primary_camera():
        return "camera"
    if not _extras_decided():
        return "extras"
    if not _discord_decided():
        return "discord"
    return "done"


def _start_camera_loop() -> None:
    try:
        from camera import get_camera_loop
        get_camera_loop().start()
    except Exception as exc:
        logger.warning("Camera loop start failed: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@setup_bp.route("/", methods=["GET"])
def wizard():
    if setup_complete():
        return redirect(url_for("admin.dashboard"))

    state = _wizard_state()
    if state == "done":
        # Self-heal: every step satisfied, flip the flag.
        set_setting("setup_complete", "true", user_id=session.get("user_id"))
        _start_camera_loop()
        return redirect(url_for("admin.dashboard"))

    extras = _list_extra_cameras() if state in {"extras", "discord"} else []
    return render_template(
        "setup/wizard.html",
        step=state,
        primary_kind=get_setting("camera_preferred_source", ""),
        primary_label=_describe_primary(),
        extras=extras,
    )


# --------- ADMIN ----------

@setup_bp.route("/admin", methods=["POST"])
def submit_admin():
    if _has_users():
        return redirect(url_for("setup.wizard"))

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    confirm  = request.form.get("password_confirm") or ""

    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        flash("Username must be 3-32 chars (letters, digits, _ . -).", "danger")
        return redirect(url_for("setup.wizard"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "danger")
        return redirect(url_for("setup.wizard"))
    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("setup.wizard"))

    db = get_db()
    role = models.get_role_by_name(db, "admin")
    if not role:
        flash("Default admin role missing — DB not initialised correctly.", "danger")
        return redirect(url_for("setup.wizard"))

    uid = models.create_user(
        db,
        username=username,
        password_hash=hash_password(password),
        role_id=role["id"],
        display_name=username,
    )
    user_row = models.get_user_by_id(db, uid)
    if user_row:
        login_user(user_row)

    audit("SETUP_ADMIN_CREATED", user_id=uid, username=username,
          ip_address=get_client_ip())
    flash(f"Welcome, {username}.", "success")
    return redirect(url_for("setup.wizard"))


# --------- PRIMARY CAMERA ----------

@setup_bp.route("/camera", methods=["POST"])
def submit_camera():
    if not _has_users():
        return redirect(url_for("setup.wizard"))

    cam_type = (request.form.get("cam_type") or "").strip().lower()
    if cam_type not in {"webcam", "ip", "kinect", "tapo"}:
        flash("Pick a camera type to continue.", "danger")
        return redirect(url_for("setup.wizard"))

    uid = session.get("user_id")
    pending: list[tuple[str, str]] = _validate_camera_fields(cam_type, request.form)
    if pending is None:
        return redirect(url_for("setup.wizard"))

    set_setting("camera_preferred_source", cam_type, user_id=uid)
    for k, v in pending:
        set_setting(k, v, user_id=uid)

    audit("SETUP_CAMERA_CONFIGURED", user_id=uid,
          username=session.get("username", ""),
          detail=f"type={cam_type}", ip_address=get_client_ip())
    return redirect(url_for("setup.wizard"))


# --------- EXTRA (live-only) CAMERAS ----------

@setup_bp.route("/extras/add", methods=["POST"])
def add_extra_camera():
    if not _has_primary_camera():
        return redirect(url_for("setup.wizard"))

    label = (request.form.get("label") or "").strip()
    url = (request.form.get("url") or "").strip()
    if not url:
        flash("Stream URL is required for an extra camera.", "danger")
        return redirect(url_for("setup.wizard"))

    if not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
        flash("Stream URL must start with rtsp://, http://, or https://", "danger")
        return redirect(url_for("setup.wizard"))

    extras = _list_extra_cameras_raw()
    line = f"{label}|{url}" if label else url
    extras.append(line)
    set_setting("ip_camera_urls", "\n".join(extras), user_id=session.get("user_id"))
    flash(f"Added camera: {label or url}", "success")
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/extras/remove", methods=["POST"])
def remove_extra_camera():
    try:
        idx = int(request.form.get("idx", "-1"))
    except (TypeError, ValueError):
        idx = -1
    extras = _list_extra_cameras_raw()
    if 0 <= idx < len(extras):
        removed = extras.pop(idx)
        set_setting("ip_camera_urls", "\n".join(extras), user_id=session.get("user_id"))
        flash(f"Removed: {removed}", "info")
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/extras/done", methods=["POST"])
def extras_done():
    if not _has_primary_camera():
        return redirect(url_for("setup.wizard"))
    set_setting("setup_extras_done", "true", user_id=session.get("user_id"))
    return redirect(url_for("setup.wizard"))


# --------- DISCORD ----------

@setup_bp.route("/discord", methods=["POST"])
def submit_discord():
    if not _has_users():
        return redirect(url_for("setup.wizard"))

    uid = session.get("user_id")
    skip = (request.form.get("skip") or "").strip().lower() == "true"
    if skip:
        set_setting("setup_skip_discord", "true", user_id=uid)
    else:
        webhook = (request.form.get("discord_webhook_url") or "").strip()
        if webhook and not webhook.startswith("https://discord.com/api/webhooks/"):
            flash("That doesn't look like a Discord webhook URL.", "danger")
            return redirect(url_for("setup.wizard"))
        if not webhook:
            flash("Webhook URL is required (or click Skip).", "danger")
            return redirect(url_for("setup.wizard"))
        set_setting("discord_webhook_url", webhook, user_id=uid)

    set_setting("setup_complete", "true", user_id=uid)
    audit("SETUP_COMPLETE", user_id=uid,
          username=session.get("username", ""),
          ip_address=get_client_ip())
    flash("Setup complete. Welcome to CacheSec.", "success")
    _start_camera_loop()
    return redirect(url_for("admin.dashboard"))


# ---------------------------------------------------------------------------
# Camera test endpoint (used by the wizard's auto-probe)
# ---------------------------------------------------------------------------

@setup_bp.route("/test-camera", methods=["POST"])
def test_camera():
    if not _has_users():
        return jsonify(ok=False, error="Create the admin account first."), 403

    cam_type = (request.form.get("cam_type") or "").strip().lower()
    try:
        if cam_type == "webcam":
            return _probe_webcam()
        if cam_type == "ip":
            return _probe_url((request.form.get("ip_camera_url") or "").strip())
        if cam_type == "tapo":
            host = (request.form.get("tapo_host") or "").strip()
            user = (request.form.get("tapo_username") or "admin").strip() or "admin"
            pw = request.form.get("tapo_password") or ""
            stream = (request.form.get("tapo_stream") or "stream1").strip().lower()
            if stream not in {"stream1", "stream2"}:
                stream = "stream1"
            if not host or not pw:
                return jsonify(ok=False, error="Host and password required.")
            url = f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{host}:554/{stream}"
            return _probe_url(url)
        if cam_type == "kinect":
            return _probe_kinect()
        if cam_type == "extra":
            return _probe_url((request.form.get("url") or "").strip())
        return jsonify(ok=False, error="Unknown camera type.")
    except Exception as exc:
        logger.exception("Camera probe failed")
        return jsonify(ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_camera_fields(cam_type: str, form) -> "list[tuple[str,str]] | None":
    """Return list of (key, value) settings to persist, or None on validation
    failure (after flashing an error)."""
    pending: list[tuple[str, str]] = []
    if cam_type == "ip":
        url = (form.get("ip_camera_url") or "").strip()
        if not url:
            flash("Stream URL is required.", "danger")
            return None
        if not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
            flash("URL must start with rtsp://, http://, or https://", "danger")
            return None
        transport = (form.get("ip_camera_rtsp_transport") or "tcp").strip().lower()
        if transport not in {"tcp", "udp", "udp_multicast", "http"}:
            transport = "tcp"
        pending.append(("ip_camera_url", url))
        pending.append(("ip_camera_rtsp_transport", transport))
    elif cam_type == "tapo":
        host = (form.get("tapo_host") or "").strip()
        username = (form.get("tapo_username") or "admin").strip() or "admin"
        password = form.get("tapo_password") or ""
        stream = (form.get("tapo_stream") or "stream1").strip().lower()
        if stream not in {"stream1", "stream2"}:
            stream = "stream1"
        if not host:
            flash("Tapo host (IP) is required.", "danger")
            return None
        if not password:
            flash("Tapo password is required.", "danger")
            return None
        pending.append(("tapo_host", host))
        pending.append(("tapo_username", username))
        pending.append(("tapo_password", password))
        pending.append(("tapo_stream", stream))
    return pending


def _list_extra_cameras_raw() -> list[str]:
    raw = get_setting("ip_camera_urls", "")
    return [line.strip() for line in raw.replace(",", "\n").splitlines() if line.strip()]


def _list_extra_cameras() -> list[dict]:
    out = []
    for idx, line in enumerate(_list_extra_cameras_raw()):
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        if len(parts) == 1:
            out.append({"idx": idx, "label": "", "url": parts[0]})
        else:
            label = parts[0]
            url = parts[1]
            # If first part looks like a URL, swap.
            if "://" in label and "://" not in url:
                label, url = "", parts[0]
            out.append({"idx": idx, "label": label, "url": url})
    return out


def _describe_primary() -> str:
    src = get_setting("camera_preferred_source", "")
    if src == "tapo":
        host = get_setting("tapo_host", "")
        return f"Tapo · {host}" if host else "Tapo"
    if src == "ip":
        url = get_setting("ip_camera_url", "")
        return f"IP / RTSP · {url[:48]}" if url else "IP / RTSP"
    if src == "kinect":
        return "Kinect v1"
    if src == "webcam":
        return "USB / Pi camera"
    return src or ""


# ---- probes ----

def _probe_webcam():
    import cv2
    idx = config.CAMERA_INDEX
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return jsonify(ok=False, error=f"Cannot open /dev/video{idx}.")
        ok, _ = cap.read()
        if not ok:
            return jsonify(ok=False, error="Camera opened but returned no frames.")
        return jsonify(ok=True, message=f"Frame received from /dev/video{idx}.")
    finally:
        cap.release()


def _probe_url(url: str):
    import cv2, os
    if not url:
        return jsonify(ok=False, error="URL is empty.")
    prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            return jsonify(ok=False, error="Could not open the stream (auth, URL, or network).")
        for _ in range(20):
            ok, _ = cap.read()
            if ok:
                return jsonify(ok=True, message="Stream opened and produced frames.")
        return jsonify(ok=False, error="Stream opened but produced no frames in time.")
    finally:
        cap.release()
        if prev is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev


def _probe_kinect():
    try:
        from kinect import kinect_available
    except Exception as exc:
        return jsonify(ok=False, error=f"Kinect module unavailable: {exc}")
    if not kinect_available():
        return jsonify(ok=False, error="Kinect device not detected on USB.")
    return jsonify(ok=True, message="Kinect detected.")
