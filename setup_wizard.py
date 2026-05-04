"""
setup_wizard.py — First-run setup wizard.

Triggers when the database has no users and the `setup_complete` flag is not
set. Walks the operator through:

  1. Admin account creation (replaces the legacy auto-`admin/changeme123`).
  2. Primary detection camera (USB / Pi, IP/RTSP, Tapo, Kinect, or skip).
  3. Discord webhook (optional).

When complete, sets `setup_complete=true` in the settings table; the
`before_request` gate in app.py then stops redirecting to /setup.
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
# Helpers
# ---------------------------------------------------------------------------

def setup_complete() -> bool:
    """True once the wizard has finished. Cached on first hit."""
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


def _wizard_state() -> str:
    """Which step are we on?"""
    if not _has_users():
        return "admin"
    if not get_setting("camera_preferred_source", ""):
        return "camera"
    if get_setting("setup_skip_discord", "") != "true" and not get_setting("discord_webhook_url", ""):
        return "discord"
    return "done"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@setup_bp.route("/", methods=["GET"])
def wizard():
    if setup_complete():
        return redirect(url_for("admin.dashboard"))
    return render_template("setup/wizard.html", step=_wizard_state())


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
    logger.info("Setup wizard: admin user '%s' created (uid=%d)", username, uid)

    # Sign the user in so the rest of the wizard runs under their session.
    user_row = models.get_user_by_id(db, uid)
    if user_row:
        login_user(user_row)

    audit("SETUP_ADMIN_CREATED", user_id=uid, username=username,
          ip_address=get_client_ip())
    flash(f"Welcome, {username}.", "success")
    return redirect(url_for("setup.wizard"))


@setup_bp.route("/camera", methods=["POST"])
def submit_camera():
    if not _has_users():
        return redirect(url_for("setup.wizard"))

    skip = (request.form.get("skip") or "").strip().lower() == "true"
    if skip:
        # Pick a sensible default so the wizard advances. Users can change
        # this later in Settings.
        set_setting("camera_preferred_source", "webcam",
                    user_id=session.get("user_id"))
        flash("Skipped camera setup. Configure it later in Settings.", "info")
        return redirect(url_for("setup.wizard"))

    cam_type = (request.form.get("cam_type") or "webcam").strip().lower()
    if cam_type not in {"webcam", "ip", "kinect", "tapo"}:
        cam_type = "webcam"

    uid = session.get("user_id")
    set_setting("camera_preferred_source", cam_type, user_id=uid)

    if cam_type == "ip":
        url = (request.form.get("ip_camera_url") or "").strip()
        if not url:
            flash("IP camera URL is required.", "danger")
            return redirect(url_for("setup.wizard"))
        set_setting("ip_camera_url", url, user_id=uid)
        transport = (request.form.get("ip_camera_rtsp_transport") or "tcp").strip().lower()
        if transport not in {"tcp", "udp", "udp_multicast", "http"}:
            transport = "tcp"
        set_setting("ip_camera_rtsp_transport", transport, user_id=uid)
    elif cam_type == "tapo":
        host = (request.form.get("tapo_host") or "").strip()
        username = (request.form.get("tapo_username") or "admin").strip() or "admin"
        password = request.form.get("tapo_password") or ""
        stream = (request.form.get("tapo_stream") or "stream1").strip().lower()
        if stream not in {"stream1", "stream2"}:
            stream = "stream1"
        if not host or not password:
            flash("Tapo host and password are required.", "danger")
            return redirect(url_for("setup.wizard"))
        set_setting("tapo_host", host, user_id=uid)
        set_setting("tapo_username", username, user_id=uid)
        set_setting("tapo_password", password, user_id=uid)
        set_setting("tapo_stream", stream, user_id=uid)

    audit("SETUP_CAMERA_CONFIGURED", user_id=uid,
          username=session.get("username", ""),
          detail=f"type={cam_type}", ip_address=get_client_ip())
    return redirect(url_for("setup.wizard"))


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
        if webhook:
            set_setting("discord_webhook_url", webhook, user_id=uid)

    set_setting("setup_complete", "true", user_id=uid)
    audit("SETUP_COMPLETE", user_id=uid,
          username=session.get("username", ""),
          ip_address=get_client_ip())
    flash("Setup complete. Welcome to CacheSec.", "success")

    # Kick the camera loop now that settings exist.
    try:
        from camera import get_camera_loop
        get_camera_loop().start()
    except Exception as exc:
        logger.warning("Camera loop start after setup failed: %s", exc)

    return redirect(url_for("admin.dashboard"))


@setup_bp.route("/test-camera", methods=["POST"])
def test_camera():
    """Probe the camera config without saving — returns ok/error JSON."""
    if not _has_users():
        return jsonify(ok=False, error="Create the admin account first."), 403

    cam_type = (request.form.get("cam_type") or "webcam").strip().lower()

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
        return jsonify(ok=False, error=f"Unknown camera type: {cam_type}")
    except Exception as exc:
        logger.exception("Camera probe failed")
        return jsonify(ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _probe_webcam() -> "tuple":
    import cv2
    idx = config.CAMERA_INDEX
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return jsonify(ok=False, error=f"Cannot open /dev/video{idx}")
        ok, _ = cap.read()
        if not ok:
            return jsonify(ok=False, error="Camera opened but returned no frames.")
        return jsonify(ok=True, message=f"Captured a frame from /dev/video{idx}.")
    finally:
        cap.release()


def _probe_url(url: str) -> "tuple":
    import cv2
    if not url:
        return jsonify(ok=False, error="URL is empty.")
    # Force ffmpeg backend with a short timeout — RTSP discovery can hang
    # for the full default timeout otherwise.
    import os
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
        return jsonify(ok=False, error="Stream opened but no frames received in time.")
    finally:
        cap.release()
        if prev is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev


def _probe_kinect() -> "tuple":
    try:
        from kinect import kinect_available
    except Exception as exc:
        return jsonify(ok=False, error=f"Kinect module unavailable: {exc}")
    if not kinect_available():
        return jsonify(ok=False, error="Kinect device not detected on USB.")
    return jsonify(ok=True, message="Kinect detected.")
