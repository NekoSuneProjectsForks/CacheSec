"""
camera.py — Camera capture and main detection loop.

Responsibilities:
  1. Open the camera — Kinect v1 (preferred) or USB webcam (fallback).
  2. Continuously read frames.
  3. Run face detection every FRAME_SKIP frames.
  4. Match detected faces against enrolled gallery.
  5. Route results to recorder.py and sound.py.
  6. Save events and snapshots to the database.
  7. Expose the latest JPEG frame for the live web stream (MJPEG).
  8. Handle camera disconnects and attempt reconnection.
  9. Apply night-vision filter when frame is dark.

Kinect vs webcam night-vision:
  - Kinect: switches to the native IR stream (hardware IR projector illuminates
    the room in complete darkness; no software processing needed beyond the
    green tint). Face detection runs on the IR frame directly.
  - Webcam fallback: software gamma lift + CLAHE + green tint on the RGB frame.
    Quality is limited by how much light the sensor can collect.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

import config
from onvif_control import OnvifNightVisionController, OnvifNightVisionSettings
from recognition import get_recognizer, DetectedFace
from recorder import get_recorder
from utils import timestamped_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state for live MJPEG stream
# ---------------------------------------------------------------------------
_latest_frame_lock  = threading.Lock()
_latest_jpeg: bytes = b""
_camera_status      = {
    "running":      False,
    "error":        "",
    "night_vision": False,
    "source":       "webcam",   # "webcam", "ip", or "kinect"
    "sls_enabled":  config.SLS_ENABLED,
    "sls_active":   False,
    "depth":        False,
}

# ---------------------------------------------------------------------------
# Night-vision parameters
# ---------------------------------------------------------------------------
# Brightness threshold below which night-vision filter activates (0-255).
# Hysteresis band prevents rapid switching: activate below threshold,
# deactivate above threshold + NIGHT_VISION_HYSTERESIS.
NIGHT_VISION_THRESHOLD   = 100  # activate if mean brightness < 100/255
NIGHT_VISION_HYSTERESIS  = 25   # deactivate only when > 125/255
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def get_latest_jpeg() -> bytes:
    with _latest_frame_lock:
        return _latest_jpeg


def _set_latest_jpeg(frame: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _latest_frame_lock:
            global _latest_jpeg
            _latest_jpeg = buf.tobytes()


def get_camera_status() -> dict:
    return dict(_camera_status)


def get_live_sources() -> list[dict]:
    preferred = _live_camera_preferred_source()
    sources = [{
        "id": "primary",
        "label": "Primary Detection Feed",
        "kind": "primary",
        "active": True,
        "detail": _camera_status.get("source", preferred),
    }]

    # If detection is running from IP/Kinect, the local USB camera can still be
    # opened as a separate live-only feed.
    if preferred != "webcam":
        sources.append({
            "id": "webcam",
            "label": f"USB / Pi Camera {config.CAMERA_INDEX}",
            "kind": "webcam",
            "active": False,
            "detail": f"index {config.CAMERA_INDEX}",
        })

    # Expose Kinect as an explicit live-only source so operators can view
    # Kinect and webcam side-by-side regardless of which source detection uses.
    if config.KINECT_ENABLED:
        try:
            from kinect import kinect_available
            has_kinect = kinect_available()
        except Exception:
            has_kinect = False
        if has_kinect:
            sources.append({
                "id": "kinect",
                "label": "Kinect v1",
                "kind": "kinect",
                "active": False,
                "detail": "RGB/IR (hardware)",
            })

    for idx, item in enumerate(_configured_ip_sources(), start=1):
        sources.append({
            "id": f"ip{idx}",
            "label": item["label"],
            "kind": "ip",
            "active": False,
            "detail": _display_source_url(item["url"]),
        })

    # Tapo as a separate switchable feed (so PTZ controls render even when
    # detection is using a different primary source).
    try:
        from tapo_control import tapo_configured, tapo_settings
        if tapo_configured() and preferred != "tapo":
            s = tapo_settings()
            sources.append({
                "id": "tapo",
                "label": "Tapo Camera",
                "kind": "tapo",
                "active": False,
                "detail": f"{s['host']} ({s['stream']})",
            })
    except Exception:
        pass
    return sources


class FFmpegFrameCapture:
    """Minimal VideoCapture-like wrapper for HLS streams via system ffmpeg."""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int = 15,
        http_options: dict[str, str] | None = None,
    ):
        self.url = url
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen[bytes] | None = None

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg is not available for HLS camera support")
            return

        vf = (
            f"fps={self.fps},"
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "warning",
            "-nostdin",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-rw_timeout", "5000000",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "2",
        ]
        cmd += _ffmpeg_http_input_args(url, http_options)
        cmd += [
            "-i", url,
            "-an",
            "-vf", vf,
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_bytes * 2,
            )
        except Exception as exc:
            logger.warning("Failed to start ffmpeg for %s: %s", _redact_url(url), exc)
            self._proc = None

    def isOpened(self) -> bool:
        return bool(self._proc and self._proc.poll() is None and self._proc.stdout)

    def set(self, _prop_id: int, _value: float) -> bool:
        return False

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def _read_exact(self, size: int) -> bytes:
        if not self._proc or not self._proc.stdout:
            return b""
        data = bytearray()
        while len(data) < size:
            chunk = self._proc.stdout.read(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        payload = self._read_exact(self._frame_bytes)
        if len(payload) != self._frame_bytes:
            self.release()
            return False, None
        frame = np.frombuffer(payload, dtype=np.uint8)
        return True, frame.reshape((self.height, self.width, 3))

    def release(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass


CaptureHandle = cv2.VideoCapture | FFmpegFrameCapture


# ---------------------------------------------------------------------------
# Night-vision filter
# ---------------------------------------------------------------------------

def _apply_night_vision(frame: np.ndarray) -> np.ndarray:
    """
    Night-vision filter for genuinely dark frames:
      1. Gamma correction — aggressively brightens dark pixels while
         leaving bright pixels mostly untouched (gamma < 1 = brighten).
      2. CLAHE on the Y channel — stretches local contrast after brightening.
      3. Green phosphor tint.
    Works even when the frame is near-black because gamma lift happens
    before CLAHE, giving the equaliser actual signal to work with.
    """
    # -- Step 1: extreme gamma lift for near-pitch-black frames (gamma=0.12)
    inv_gamma = 1.0 / 0.12
    lut = np.array([
        min(255, int((i / 255.0) ** inv_gamma * 255))
        for i in range(256)
    ], dtype=np.uint8)
    brightened = cv2.LUT(frame, lut)

    # -- Step 2: convert to grayscale and discard colour noise —
    #    in true darkness colour channels are just sensor noise.
    #    Working in grayscale gives a cleaner NV look.
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

    # -- Step 3: aggressive CLAHE (high clipLimit = more local contrast)
    clahe_hard = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    gray_eq = clahe_hard.apply(gray)

    # -- Step 4: light denoise to kill salt-and-pepper sensor noise
    gray_dn = cv2.fastNlMeansDenoising(gray_eq, h=10, templateWindowSize=7, searchWindowSize=21)

    # -- Step 5: return as grayscale BGR (all channels equal)
    return cv2.cvtColor(gray_dn, cv2.COLOR_GRAY2BGR)



_night_vision_active = False


# ---------------------------------------------------------------------------
# Unknown-detection state machine
# ---------------------------------------------------------------------------

class _UnknownTracker:
    """
    Tracks a continuous unknown-person event to prevent duplicate DB rows
    and require a sustained presence before triggering an alert.

    State machine:
      IDLE      → face appears → CONFIRMING (accumulate confirm_secs)
      CONFIRMING → face held for confirm_secs → ACTIVE (create event, start recording)
      CONFIRMING → face disappears → back to IDLE (false positive / glance)
      ACTIVE    → face gone   → COOLDOWN (signal recorder to stop after its gap)
      COOLDOWN  → cooldown_secs elapsed → IDLE (ready for next event)

    confirm_secs prevents a single-frame or brief appearance from triggering
    an alert — the person must be in frame continuously for ~3-4 seconds.
    """

    CONFIRM_SECS = 3.5   # must be seen this long before alert fires

    def __init__(self):
        self.active              = False
        self.confirming          = False      # seen but not yet confirmed
        self.confirm_start       = 0.0        # when we first saw this unknown
        self.event_id: int | None = None
        self.last_seen           = 0.0
        self.last_event_time     = 0.0
        self.cooldown_secs       = config.UNKNOWN_COOLDOWN_SECONDS

    def reset(self):
        self.active        = False
        self.confirming    = False
        self.confirm_start = 0.0
        self.event_id      = None
        self.last_seen     = 0.0
        self.last_event_time = time.monotonic()

    def in_cooldown(self) -> bool:
        return (time.monotonic() - self.last_event_time) < self.cooldown_secs

    def is_confirmed(self) -> bool:
        """True once the unknown has been in frame long enough to trigger."""
        return self.confirming and (time.monotonic() - self.confirm_start) >= self.CONFIRM_SECS

    def is_expired(self) -> bool:
        return (
            self.active
            and (time.monotonic() - self.last_seen) >= self.cooldown_secs
        )


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraLoop:
    def __init__(self):
        self._stop_flag  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CameraLoop"
        )
        self._thread.start()
        logger.info("CameraLoop started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("CameraLoop stopped")

    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture | None:
        idx = config.CAMERA_INDEX
        # Try V4L2 backend first (most reliable on Pi OS)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d", idx)
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal latency

        # Start in auto-exposure (aperture priority) with neutral settings.
        # Night vision will switch to manual max when darkness is detected.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = aperture priority auto
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)      # neutral (default=0)
        cap.set(cv2.CAP_PROP_CONTRAST, 32)       # default
        cap.set(cv2.CAP_PROP_GAIN, 0)            # let auto handle it

        # Warm up: read and discard a few frames so the sensor settles
        for _ in range(5):
            cap.read()

        # Log what we actually got
        exp  = cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = cap.get(cv2.CAP_PROP_GAIN)
        logger.info("Camera opened (index=%d, %dx%d, exposure=%.0f, gain=%.0f)",
                    idx, config.FRAME_WIDTH, config.FRAME_HEIGHT, exp, gain)
        return cap

    def _open_ip_camera(self) -> CaptureHandle | None:
        url = _live_setting("ip_camera_url", config.IP_CAMERA_URL).strip()
        if not url:
            logger.error("IP camera source selected but IP_CAMERA_URL is empty")
            _camera_status["error"] = "IP camera URL is not configured"
            return None

        transport = _live_setting(
            "ip_camera_rtsp_transport", config.IP_CAMERA_RTSP_TRANSPORT
        ).strip().lower()
        if transport not in {"tcp", "udp", "udp_multicast", "http"}:
            transport = "tcp"

        url = _normalize_camera_url(url)
        if not _is_supported_camera_url(url):
            logger.error("Unsupported IP camera URL: %s", _redact_url(url))
            _camera_status["error"] = "IP camera URL must use rtsp://, http://, or https://"
            return None

        cap = _open_stream_capture(url, "IP camera", transport=transport)
        if cap is None:
            logger.error("Cannot open IP camera stream: %s", _redact_url(url))
            _camera_status["error"] = "IP camera unavailable"
            return None

        logger.info("IP camera opened: %s", _redact_url(url))
        return cap

    def _open_tapo_camera(self) -> CaptureHandle | None:
        from tapo_control import tapo_rtsp_url, tapo_settings
        s = tapo_settings()
        if not s["host"] or not s["password"]:
            logger.error("Tapo source selected but host/password are not configured")
            _camera_status["error"] = "Tapo camera credentials not configured"
            return None

        url = tapo_rtsp_url(s)
        cap = _open_stream_capture(url, "Tapo camera", transport="tcp")
        if cap is None:
            logger.error("Cannot open Tapo camera stream at %s", _redact_url(url))
            _camera_status["error"] = "Tapo camera unavailable"
            return None

        logger.info("Tapo camera opened: %s (%s)", s["host"], s["stream"])
        return cap

    def _run(self) -> None:
        global _night_vision_active

        recognizer = get_recognizer()
        recorder   = get_recorder()
        recorder.start_background()

        tracker        = _UnknownTracker()
        frame_count    = 0
        reconnect_wait = 2
        record_all_mode = _live_bool_setting("record_all_mode", False)
        record_all_last_check = time.monotonic()
        record_all_last_signal = 0.0

        _camera_status["running"] = True
        _camera_status["error"]   = ""
        _night_vision_active = False
        _camera_status["night_vision"] = False
        _camera_status["sls_enabled"] = config.SLS_ENABLED
        _camera_status["sls_active"] = False
        _camera_status["depth"] = False

        # ---- Open preferred source, with fallback ----
        from kinect import get_kinect, kinect_available, KinectLED
        kinect = get_kinect()
        use_kinect = False
        cap = None
        preferred_source = _live_camera_preferred_source()
        ip_source_url = _normalize_camera_url(_live_setting("ip_camera_url", config.IP_CAMERA_URL))
        ip_onvif = _build_ip_onvif_controller(ip_source_url) if preferred_source == "ip" else None

        initial_ip_night = ip_onvif.initial_state() if ip_onvif is not None else None
        if initial_ip_night is not None:
            _night_vision_active = initial_ip_night
            _camera_status["night_vision"] = initial_ip_night

        def primary_source_label() -> str:
            if preferred_source == "ip":
                return "ip"
            if preferred_source == "tapo":
                return "tapo"
            return "webcam"

        def try_open_primary_source() -> CaptureHandle | None:
            return self.__try_open_primary(preferred_source)

        def start_kinect_source() -> bool:
            global _night_vision_active
            if not (config.KINECT_ENABLED and kinect_available()):
                return False
            if kinect.start():
                _camera_status["source"] = "kinect"
                if config.KINECT_TILT != 0:
                    kinect.set_tilt(config.KINECT_TILT)
                logger.info("Using Kinect as camera source")
                cap = None

                if config.SLS_ENABLED and config.SLS_MODE == "always":
                    logger.info("SLS always mode enabled — starting Kinect in IR/depth mode")
                    _night_vision_active = True
                    _camera_status["night_vision"] = True
                    _camera_status["sls_active"] = True
                    kinect.set_mode("ir")
                    kinect.set_led(KinectLED.BLINK_GREEN)
                else:
                    # Check brightness of first frame — if already dark, start in IR
                    first = kinect.read_frame()
                    if first is not None:
                        gray_mean = float(np.mean(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)))
                        if gray_mean < NIGHT_VISION_THRESHOLD:
                            logger.info("Room already dark (%.1f) — starting Kinect in IR mode", gray_mean)
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                        else:
                            kinect.set_led(KinectLED.GREEN)
                    else:
                        kinect.set_led(KinectLED.GREEN)
                return True
            else:
                logger.warning("Kinect detected but failed to start: %s — falling back to webcam",
                               kinect.error)
                return False

        def switch_to_kinect_night(gray_mean: float) -> bool:
            global _night_vision_active
            if not config.KINECT_NIGHT_VISION_ENABLED:
                return False
            logger.info("Main camera dark (brightness=%.1f) - trying Kinect IR night vision", gray_mean)
            if not start_kinect_source():
                return False
            if cap is not None:
                cap.release()
            _night_vision_active = True
            _camera_status["night_vision"] = True
            _camera_status["source"] = "kinect"
            kinect.set_mode("ir")
            kinect.set_led(KinectLED.BLINK_GREEN)
            self._kinect_settle = 10
            logger.info("Switched to Kinect IR night vision")
            return True

        def switch_to_webcam_day(gray_mean: float):
            global _night_vision_active
            new_cap = try_open_primary_source()
            if new_cap is None:
                return None
            try:
                kinect.stop()
            except Exception:
                pass
            _night_vision_active = False
            _camera_status["night_vision"] = False
            _camera_status["sls_active"] = False
            _camera_status["depth"] = False
            _camera_status["source"] = primary_source_label()
            logger.info(
                "Kinect bright (brightness=%.1f) - switched back to %s camera",
                gray_mean, primary_source_label(),
            )
            return new_cap

        if preferred_source in {"webcam", "ip", "tapo"}:
            _camera_status["source"] = primary_source_label()
            cap = try_open_primary_source()
            if cap is None:
                logger.warning("Preferred %s camera unavailable; trying Kinect fallback", primary_source_label())
                use_kinect = start_kinect_source()
                _camera_status["error"] = "" if use_kinect else _camera_status["error"]
            else:
                logger.info("Using %s as camera source", primary_source_label())
        else:
            use_kinect = start_kinect_source()
            if not use_kinect:
                _camera_status["source"] = "webcam"
                cap = self.__try_open()
                if cap is not None:
                    logger.info("Using webcam as camera source")

        if cap is None and not use_kinect:
            _camera_status["running"] = False
            recorder.stop_background()
            return

        try:
            while not self._stop_flag.is_set():

                # ---- Read frame from appropriate source ----
                if use_kinect:
                    frame = self._read_kinect_frame(kinect)
                    if frame is None:
                        time.sleep(0.03)
                        continue
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.warning("Webcam read failed — reconnecting in %ds", reconnect_wait)
                        _camera_status["error"] = "Camera disconnected"
                        cap.release()
                        time.sleep(reconnect_wait)
                        cap = try_open_primary_source()
                        if cap is None:
                            time.sleep(reconnect_wait)
                            continue
                        _camera_status["error"] = ""
                        frame_count = 0
                        continue

                frame_count += 1

                # ---- Night-vision ----
                if use_kinect:
                    if config.SLS_ENABLED and config.SLS_MODE == "always":
                        if not _night_vision_active or kinect.get_mode() != "ir":
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            kinect.set_mode("ir")
                            kinect.set_led(KinectLED.BLINK_GREEN)
                    else:
                        # Skip brightness check for a few frames after a mode switch
                        # to let the Kinect flush its old-stream buffer before we
                        # evaluate brightness again.
                        kinect_settle = getattr(self, '_kinect_settle', 0)
                        if kinect_settle > 0:
                            self._kinect_settle = kinect_settle - 1
                        else:
                            gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                            if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                                _night_vision_active = True
                                _camera_status["night_vision"] = True
                                kinect.set_mode("ir")
                                kinect.set_led(KinectLED.BLINK_GREEN)
                                self._kinect_settle = 6   # skip 6 frames after switch
                                logger.info("Kinect → IR mode (brightness=%.1f)", gray_mean)
                            elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                                if (
                                    preferred_source in {"webcam", "ip", "tapo"}
                                    and config.KINECT_NIGHT_VISION_ENABLED
                                ):
                                    new_cap = switch_to_webcam_day(gray_mean)
                                    if new_cap is not None:
                                        cap = new_cap
                                        use_kinect = False
                                        frame_count = 0
                                        continue
                                _night_vision_active = False
                                _camera_status["night_vision"] = False
                                kinect.set_mode("rgb")
                                kinect.set_led(KinectLED.GREEN)
                                self._kinect_settle = 6
                                logger.info("Kinect → RGB mode (brightness=%.1f)", gray_mean)

                    # kinect.read_frame() returns plain BGR in RGB mode and
                    # CLAHE-enhanced grayscale-as-BGR in IR mode — no further processing
                    display_frame = frame

                    # SLS skeleton overlay — uses Kinect depth data.
                    sls_active = bool(
                        config.SLS_ENABLED
                        and (_night_vision_active or config.SLS_MODE == "always")
                    )
                    _camera_status["sls_active"] = sls_active
                    if sls_active:
                        depth_for_skel = kinect.read_depth()
                        _camera_status["depth"] = depth_for_skel is not None
                        if depth_for_skel is not None:
                            from skeleton import overlay_skeletons
                            display_frame = overlay_skeletons(
                                display_frame.copy(),
                                depth_for_skel,
                                max_people=config.SLS_MAX_PEOPLE,
                            )
                    else:
                        _camera_status["depth"] = kinect.read_depth() is not None
                else:
                    _camera_status["sls_active"] = False
                    _camera_status["depth"] = False
                    if preferred_source == "tapo":
                        # Tapo cameras flip their own IR-cut filter at night;
                        # don't apply the software green-tint filter.
                        if _night_vision_active:
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                        display_frame = frame
                    elif preferred_source == "ip":
                        # IP cameras can expose IR-cut control via ONVIF. When
                        # enabled, use the same brightness detection as the
                        # webcam path, but send the switch command to the
                        # camera instead of drawing a fake green NV filter.
                        if ip_onvif is not None:
                            gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                            if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                                if ip_onvif.set_night_vision(True):
                                    _night_vision_active = True
                                    _camera_status["night_vision"] = True
                                    logger.info("IP camera ONVIF night vision ON (brightness=%.1f)", gray_mean)
                            elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                                if ip_onvif.set_night_vision(False):
                                    _night_vision_active = False
                                    _camera_status["night_vision"] = False
                                    logger.info("IP camera ONVIF night vision OFF (brightness=%.1f)", gray_mean)
                        elif _night_vision_active:
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                        display_frame = frame
                    else:
                        # Webcam: software NV filter
                        gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                        if not _night_vision_active and gray_mean < NIGHT_VISION_THRESHOLD:
                            if switch_to_kinect_night(gray_mean):
                                use_kinect = True
                                cap = None
                                frame_count = 0
                                continue
                            _night_vision_active = True
                            _camera_status["night_vision"] = True
                            _set_camera_exposure(cap, night=True)
                            logger.info("Night-vision ON (brightness=%.1f)", gray_mean)
                        elif _night_vision_active and gray_mean > (NIGHT_VISION_THRESHOLD + NIGHT_VISION_HYSTERESIS):
                            _night_vision_active = False
                            _camera_status["night_vision"] = False
                            _set_camera_exposure(cap, night=False)
                            logger.info("Night-vision OFF (brightness=%.1f)", gray_mean)

                        display_frame = _apply_night_vision(frame) if _night_vision_active else frame

                # Record the operator-visible image: normal frames in daylight,
                # night-vision/SLS-enhanced frames when those modes are active.
                recorder.push_frame(display_frame.copy())

                now = time.monotonic()
                # Keep this setting dynamic so it can be toggled at runtime.
                if now - record_all_last_check >= 2.0:
                    new_mode = _live_bool_setting("record_all_mode", False)
                    if record_all_mode and not new_mode:
                        recorder.signal_unknown_gone()
                    record_all_mode = new_mode
                    record_all_last_check = now

                if record_all_mode and (now - record_all_last_signal) >= 1.0:
                    # Continuous recording mode. None means "no specific event id".
                    recorder.signal_unknown_visible(None)
                    record_all_last_signal = now

                # Run detection every FRAME_SKIP frames
                if frame_count % max(1, config.FRAME_SKIP) != 0:
                    _set_latest_jpeg(display_frame)
                    continue

                threshold = _live_threshold()

                # Detect faces — on raw frame for accuracy
                # For Kinect IR, the IR frame works well for SCRFD detection
                faces: list[DetectedFace] = recognizer.detect(frame)

                if not faces:
                    if tracker.active and not record_all_mode:
                        recorder.signal_unknown_gone()
                        tracker.reset()
                    _set_latest_jpeg(display_frame)
                    continue

                unknown_in_frame = False
                annotated = display_frame.copy()

                # Grab Kinect depth frame for spoof check (best-effort)
                depth_raw = None
                if use_kinect:
                    depth_raw = kinect.read_depth()

                fh, fw = frame.shape[:2]
                for face in faces:
                    if face.embedding is None:
                        continue

                    # Record every detection in the heatmap
                    from heatmap import record_detection
                    record_detection(face.bbox, frame_w=fw, frame_h=fh)

                    # --- Spoof check ---
                    x1, y1, x2, y2 = face.bbox
                    face_crop = frame[y1:y2, x1:x2]
                    from spoof import is_live
                    live, spoof_reason = is_live(face_crop, face.bbox, depth_raw)
                    if not live:
                        logger.info("Spoof detected (bbox=%s): %s", face.bbox, spoof_reason)
                        _draw_face(annotated, face, "SPOOF", 0.0, color=(0, 165, 255))
                        continue

                    # --- Mask check ---
                    masked, mask_reason = _check_mask(face_crop)
                    if masked:
                        logger.info("Mask detected (bbox=%s): %s", face.bbox, mask_reason)
                        _draw_face(annotated, face, "MASKED", 0.0, color=(255, 165, 0))
                        unknown_in_frame = True
                        continue

                    match = recognizer.match(face.embedding, threshold=threshold)

                    if match:
                        # Check access schedule — treat out-of-hours as unknown
                        from database import raw_db_ctx
                        import models as m
                        with raw_db_ctx() as db:
                            allowed = m.is_person_allowed_now(db, match.person_id)
                        if allowed:
                            _draw_face(annotated, face, match.person_name,
                                       match.score, color=(0, 255, 0))
                            _log_recognized(face, match)
                            # Cancel any pending unknown tracker — this is a known person
                            if tracker.active or tracker.confirming:
                                logger.info("Known person recognised — cancelling unknown tracker")
                                if tracker.active and not record_all_mode:
                                    recorder.signal_unknown_gone()
                                tracker.reset()
                        else:
                            unknown_in_frame = True
                            _draw_face(annotated, face,
                                       f"{match.person_name} (NO ACCESS)",
                                       match.score, color=(0, 165, 255))
                    else:
                        unknown_in_frame = True
                        _draw_face(annotated, face, "UNKNOWN", 0.0, color=(0, 0, 220))

                if unknown_in_frame:
                    tracker.last_seen = time.monotonic()
                    if tracker.active:
                        # Already confirmed and recording — keep signalling
                        recorder.signal_unknown_visible(tracker.event_id)
                    elif not tracker.in_cooldown():
                        if not tracker.confirming:
                            # First sighting — start the confirmation timer
                            tracker.confirming    = True
                            tracker.confirm_start = time.monotonic()
                            logger.debug("Unknown face seen — confirming (need %.1fs)", tracker.CONFIRM_SECS)
                        elif tracker.is_confirmed():
                            # Held long enough — fire the alert
                            event_id = _create_unknown_event(frame)
                            tracker.active          = True
                            tracker.confirming      = False
                            tracker.event_id        = event_id
                            tracker.last_event_time = time.monotonic()
                            recorder.signal_unknown_visible(event_id)
                            _alert_unknown(event_id, frame)
                        # else: still accumulating confirm time — show "VERIFYING" label
                        if tracker.confirming and not tracker.active:
                            elapsed = time.monotonic() - tracker.confirm_start
                            remaining = max(0, tracker.CONFIRM_SECS - elapsed)
                            # Overlay a "verifying" countdown on the annotated frame
                            cv2.putText(annotated,
                                        f"Verifying... {remaining:.1f}s",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 200, 255), 2)
                else:
                    if tracker.active and not record_all_mode:
                        recorder.signal_unknown_gone()
                        tracker.reset()
                    elif tracker.confirming:
                        # Disappeared before confirmation — reset silently
                        tracker.confirming    = False
                        tracker.confirm_start = 0.0
                        logger.debug("Unknown face gone before confirmation — ignoring")

                _set_latest_jpeg(annotated)

        finally:
            if use_kinect:
                try:
                    kinect.set_led(KinectLED.BLINK_RED)
                    kinect.stop()
                except Exception:
                    pass
            elif cap is not None:
                cap.release()
            recorder.stop_background()
            _camera_status["running"] = False

    def _read_kinect_frame(self, kinect) -> np.ndarray | None:
        """Read a frame from Kinect in whichever mode is currently active."""
        return kinect.read_frame()

    def __try_open(self) -> cv2.VideoCapture | None:
        for attempt in range(5):
            if self._stop_flag.is_set():
                return None
            cap = self._open_camera()
            if cap:
                return cap
            logger.info("Retrying camera open (attempt %d/5)", attempt + 1)
            time.sleep(2)
        _camera_status["error"] = "Camera unavailable"
        return None

    def __try_open_primary(self, preferred_source: str) -> CaptureHandle | None:
        if preferred_source == "ip":
            for attempt in range(5):
                if self._stop_flag.is_set():
                    return None
                cap = self._open_ip_camera()
                if cap:
                    return cap
                logger.info("Retrying IP camera open (attempt %d/5)", attempt + 1)
                time.sleep(2)
            _camera_status["error"] = "IP camera unavailable"
            return None
        if preferred_source == "tapo":
            for attempt in range(5):
                if self._stop_flag.is_set():
                    return None
                cap = self._open_tapo_camera()
                if cap:
                    return cap
                logger.info("Retrying Tapo camera open (attempt %d/5)", attempt + 1)
                time.sleep(2)
            _camera_status["error"] = "Tapo camera unavailable"
            return None
        return self.__try_open()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _live_setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting
        value = get_setting(key, default)
    except Exception:
        return default
    return value if value is not None else default


def _live_int_setting(key: str, default: int = 0) -> int:
    raw = _live_setting(key, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _live_bool_setting(key: str, default: bool = False) -> bool:
    raw = _live_setting(key, "true" if default else "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _live_camera_preferred_source() -> str:
    source = _live_setting("camera_preferred_source", config.CAMERA_PREFERRED_SOURCE)
    source = source.strip().lower()
    return source if source in {"webcam", "ip", "kinect", "tapo"} else "webcam"


def _build_ip_onvif_controller(stream_url: str) -> OnvifNightVisionController | None:
    mode = _live_setting("ip_camera_onvif_night_mode", config.IP_CAMERA_ONVIF_NIGHT_MODE)
    mode = mode.strip().lower()
    if mode not in {"disabled", "detect"}:
        mode = "disabled"
    if mode == "disabled":
        return None

    settings = OnvifNightVisionSettings(
        mode=mode,
        host=_live_setting("ip_camera_onvif_host", config.IP_CAMERA_ONVIF_HOST).strip(),
        port=_live_int_setting("ip_camera_onvif_port", config.IP_CAMERA_ONVIF_PORT),
        username=_live_setting("ip_camera_onvif_username", config.IP_CAMERA_ONVIF_USERNAME),
        password=_live_setting("ip_camera_onvif_password", config.IP_CAMERA_ONVIF_PASSWORD),
        wsdl_dir=_live_setting("ip_camera_onvif_wsdl_dir", config.IP_CAMERA_ONVIF_WSDL_DIR).strip(),
        stream_url=stream_url,
        force_persistence=False,
    )
    return OnvifNightVisionController(settings)


def _configured_ip_sources(include_primary: bool | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if include_primary is None:
        include_primary = _live_camera_preferred_source() != "ip"

    primary = _normalize_camera_url(_live_setting("ip_camera_url", config.IP_CAMERA_URL))
    if include_primary and _is_supported_camera_url(primary):
        entries.append({"label": "Primary IP Camera", "url": primary, "options": {}})

    raw = _live_setting("ip_camera_urls", config.IP_CAMERA_URLS)
    for chunk in raw.replace(",", "\n").splitlines():
        item = chunk.strip()
        if not item or item.startswith("#"):
            continue
        parsed = _parse_camera_source_entry(item, len(entries) + 1)
        if parsed:
            entries.append(parsed)
    return entries


def _parse_camera_source_entry(item: str, index: int) -> dict[str, object] | None:
    parts = [part.strip() for part in item.split("|") if part.strip()]
    if not parts:
        return None

    options_start = len(parts)
    for idx, part in enumerate(parts):
        if "=" in part:
            options_start = idx
            break
    main_parts = parts[:options_start]
    option_parts = parts[options_start:]

    label = f"IP Camera {index}"
    url = ""
    if main_parts:
        first = _normalize_camera_url(main_parts[0])
        if _is_supported_camera_url(first):
            url = first
        else:
            label = main_parts[0]
            if len(main_parts) >= 2:
                url = _normalize_camera_url(main_parts[1])

    if not _is_supported_camera_url(url):
        return None

    options: dict[str, str] = {}
    for opt in option_parts:
        key, _, value = opt.partition("=")
        key = key.strip().lower().replace("-", "_")
        value = value.strip()
        if not value:
            continue
        if key in {"referer", "origin", "user_agent"}:
            options[key] = value

    return {"label": label, "url": url, "options": options}


def _normalize_camera_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url or "://" in url:
        return url
    host = url.split("/", 1)[0]
    if "." in host or ":" in host or host.lower() == "localhost":
        return f"http://{url}"
    return url


def _is_supported_camera_url(url: str) -> bool:
    scheme = urlsplit(url).scheme.lower()
    return scheme in {"rtsp", "rtsps", "http", "https"}


def _set_ffmpeg_capture_options(url: str, transport: str | None = None) -> None:
    if not url.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        return
    if not transport:
        transport = _live_setting(
            "ip_camera_rtsp_transport", config.IP_CAMERA_RTSP_TRANSPORT
        ).strip().lower()
    if transport not in {"tcp", "udp", "udp_multicast", "http"}:
        transport = "tcp"
    # OpenCV forwards these options to its ffmpeg backend. TCP is generally
    # more reliable for wireless RTSP cameras; nobuffer keeps latency down.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|max_delay;500000|stimeout;5000000"
    )


def _looks_like_hls_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return path.endswith((".m3u8", ".m3u"))


def _default_http_camera_user_agent() -> str:
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )


def _default_http_camera_options(url: str) -> dict[str, str]:
    options = {"user_agent": _default_http_camera_user_agent()}
    host = urlsplit(url).netloc.lower()
    if "surfline" in host:
        options["referer"] = "https://www.surfline.com/"
        options["origin"] = "https://www.surfline.com"
    return options


def _http_camera_options(
    url: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    options = _default_http_camera_options(url)
    if overrides:
        for key, value in overrides.items():
            if value:
                options[key] = value
    return options


def _ffmpeg_http_input_args(
    url: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    if not url.lower().startswith(("http://", "https://")):
        return []
    options = _http_camera_options(url, overrides)
    args: list[str] = []
    user_agent = options.get("user_agent", "").strip()
    if user_agent:
        args += ["-user_agent", user_agent]
    header_lines: list[str] = []
    referer = options.get("referer", "").strip()
    origin = options.get("origin", "").strip()
    if referer:
        header_lines.append(f"Referer: {referer}")
    if origin:
        header_lines.append(f"Origin: {origin}")
    if header_lines:
        args += ["-headers", "\r\n".join(header_lines) + "\r\n"]
    return args


def _prime_capture(cap: CaptureHandle, attempts: int = 10) -> bool:
    for _ in range(max(1, attempts)):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
        time.sleep(0.1)
    return False


def _open_hls_capture(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
) -> FFmpegFrameCapture | None:
    cap = FFmpegFrameCapture(
        url,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        fps=15,
        http_options=http_options,
    )
    if not cap.isOpened():
        cap.release()
        logger.warning("ffmpeg HLS open failed for %s: %s", label, _redact_url(url))
        return None
    if not _prime_capture(cap, attempts=5):
        cap.release()
        logger.warning(
            "ffmpeg HLS produced no frames for %s: %s. "
            "If this is a web player feed, try referer/origin headers in IP_CAMERA_URLS.",
            label,
            _redact_url(url),
        )
        return None
    logger.info("Opened HLS stream via ffmpeg for %s: %s", label, _redact_url(url))
    return cap


def _open_stream_capture(
    url: str,
    label: str,
    transport: str | None = None,
    http_options: dict[str, str] | None = None,
) -> CaptureHandle | None:
    url = _normalize_camera_url(url)
    if not _is_supported_camera_url(url):
        return None

    # Prefer system ffmpeg for HLS playlists because OpenCV network support
    # varies a lot between builds, while ffmpeg handles .m3u8 reliably.
    if _looks_like_hls_url(url):
        return _open_hls_capture(url, label, http_options=http_options)

    _set_ffmpeg_capture_options(url, transport)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not _prime_capture(cap, attempts=10):
        cap.release()
        return None
    return cap


def _looks_like_snapshot_url(url: str) -> bool:
    path = urlsplit(url).path.lower()
    if path.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return True
    lowered = url.lower()
    return any(token in lowered for token in ("snapshot", "image", "still", "jpg"))


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
    except Exception:
        return "<invalid-url>"


def _display_source_url(url: str) -> str:
    try:
        parts = urlsplit(_redact_url(url))
        suffix = "?..." if parts.query else ""
        text = urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) + suffix
        return text if len(text) <= 120 else text[:117] + "..."
    except Exception:
        text = _redact_url(url)
        return text if len(text) <= 120 else text[:117] + "..."


def _check_mask(face_crop: np.ndarray) -> tuple[bool, str]:
    """
    Detect if the lower face is covered (mask, scarf, balaclava).

    Splits the face crop into upper (eyes/forehead) and lower (nose/mouth)
    halves and compares skin-tone pixel density. A real unmasked face has
    skin tone in both halves. A masked face has skin tone concentrated only
    in the upper half.

    Returns (masked: bool, reason: str).
    """
    if face_crop is None or face_crop.size == 0:
        return False, "no_crop"

    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return False, "too_small"

    # Convert to YCrCb — skin tone range is well-defined in this space
    ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    # Standard skin-tone range in YCrCb
    lower = np.array([0,   133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    mid = h // 2
    upper_half = skin_mask[:mid, :]
    lower_half = skin_mask[mid:, :]

    upper_density = float(np.sum(upper_half > 0)) / max(upper_half.size, 1)
    lower_density = float(np.sum(lower_half > 0)) / max(lower_half.size, 1)

    # Masked if upper half has decent skin signal but lower half is mostly absent
    if upper_density > 0.15 and lower_density < 0.08:
        return True, f"lower_skin={lower_density:.2f}"

    return False, "ok"


def _set_camera_exposure(cap: cv2.VideoCapture, night: bool) -> None:
    """
    Switch camera between night (max exposure/gain) and day (auto) modes.
    V4L2 auto_exposure: 1 = manual, 3 = aperture-priority auto.
    """
    try:
        import subprocess, shutil
        if night:
            # Manual mode, max exposure, max gain
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, 5000)
            cap.set(cv2.CAP_PROP_GAIN, 100)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 64)
            # Also set via v4l2-ctl — some cameras ignore OpenCV props
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                dev = f"/dev/video{config.CAMERA_INDEX}"
                subprocess.run(
                    [v4l2, "-d", dev,
                     "--set-ctrl=auto_exposure=1",
                     "--set-ctrl=exposure_time_absolute=5000",
                     "--set-ctrl=gain=100",
                     "--set-ctrl=brightness=64"],
                    capture_output=True, timeout=2
                )
            logger.info("Camera exposure: NIGHT (manual max)")
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_GAIN, 0)
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                dev = f"/dev/video{config.CAMERA_INDEX}"
                subprocess.run(
                    [v4l2, "-d", dev, "--set-ctrl=auto_exposure=3"],
                    capture_output=True, timeout=2
                )
            logger.info("Camera exposure: DAY (auto)")
    except Exception as exc:
        logger.debug("Exposure switch failed (non-fatal): %s", exc)


def _live_threshold() -> float:
    from database import get_setting
    try:
        return float(get_setting("recognition_threshold",
                                  str(config.RECOGNITION_THRESHOLD)))
    except (ValueError, TypeError):
        return config.RECOGNITION_THRESHOLD


def _draw_face(
    frame: np.ndarray,
    face: DetectedFace,
    label: str,
    score: float,
    color: tuple,
) -> None:
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}" + (f" {score:.2f}" if score > 0 else "")
    cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)



def _log_recognized(face: DetectedFace, match) -> None:
    """Log a recognized-person event (throttled to avoid DB spam)."""
    # Simple in-memory throttle: one DB write per person per 30s
    now = time.monotonic()
    key = match.person_id
    last = _recognized_throttle.get(key, 0)
    if now - last < 60:
        return
    _recognized_throttle[key] = now

    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            m.create_event(
                db,
                event_type="recognized",
                person_id=match.person_id,
                person_name=match.person_name,
                confidence=round(match.score, 4),
            )
    except Exception as exc:
        logger.warning("Failed to log recognized event: %s", exc)


_recognized_throttle: dict[int, float] = {}


def _save_snapshot(frame: np.ndarray) -> str:
    """Save a JPEG snapshot and return just the filename (not the full path)."""
    Path(config.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    fname = timestamped_filename("unknown", "jpg")
    fpath = str(Path(config.SNAPSHOTS_DIR) / fname)
    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return fname  # only the filename — route reconstructs the full path


def _create_unknown_event(frame: np.ndarray) -> int:
    """Write an unknown event to the DB and return the new event_id."""
    snapshot_path = ""
    try:
        snapshot_path = _save_snapshot(frame)
    except Exception as exc:
        logger.warning("Snapshot save failed: %s", exc)

    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            event_id = m.create_event(
                db,
                event_type="unknown",
                snapshot_path=snapshot_path,
                occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        return event_id
    except Exception as exc:
        logger.error("Failed to create unknown event: %s", exc)
        return -1


def _alert_unknown(event_id: int, frame: np.ndarray) -> None:
    """Fire sound and Discord notification for a new unknown event."""
    from sound import play_access_denied
    play_access_denied()

    # snapshot_path in DB is just the filename; build the full path for Discord
    snapshot_full_path = ""
    try:
        from database import raw_db_ctx
        import models as m
        with raw_db_ctx() as db:
            ev = m.get_event_by_id(db, event_id)
            if ev and ev["snapshot_path"]:
                snapshot_full_path = str(
                    Path(config.SNAPSHOTS_DIR) / Path(ev["snapshot_path"]).name
                )
    except Exception:
        pass

    from discord_notify import notify_unknown
    notify_unknown(
        event_id=event_id,
        occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        recording_started=True,
        snapshot_path=snapshot_full_path,
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_camera_loop: CameraLoop | None = None


def get_camera_loop() -> CameraLoop:
    global _camera_loop
    if _camera_loop is None:
        _camera_loop = CameraLoop()
    return _camera_loop


def generate_mjpeg(source_id: str = "primary"):
    """Generator for Flask MJPEG streaming endpoint."""
    if source_id == "primary":
        while True:
            frame = get_latest_jpeg()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.05)  # ~20 FPS max to browser
        return

    if source_id == "webcam":
        yield from _generate_capture_mjpeg(config.CAMERA_INDEX, label="USB webcam")
        return

    if source_id == "kinect":
        yield from _generate_kinect_mjpeg()
        return

    if source_id == "tapo":
        try:
            from tapo_control import tapo_rtsp_url, tapo_configured
        except Exception:
            yield from _generate_error_mjpeg()
            return
        if not tapo_configured():
            yield from _generate_error_mjpeg()
            return
        url = tapo_rtsp_url()
        if not url:
            yield from _generate_error_mjpeg()
            return
        yield from _generate_capture_mjpeg(url, label="Tapo camera")
        return

    if source_id.startswith("ip"):
        try:
            idx = int(source_id[2:]) - 1
        except ValueError:
            idx = -1
        sources = _configured_ip_sources()
        if 0 <= idx < len(sources):
            source = sources[idx]
            yield from _generate_ip_mjpeg(
                source["url"],
                source["label"],
                http_options=source.get("options"),
            )
            return

    yield from _generate_error_mjpeg()


def _generate_kinect_mjpeg():
    try:
        from kinect import get_kinect, kinect_available
    except Exception:
        logger.warning("Kinect module unavailable for live stream")
        yield from _generate_error_mjpeg()
        return

    if not kinect_available():
        logger.warning("Kinect live feed requested but Kinect is not available")
        yield from _generate_error_mjpeg()
        return

    kinect = get_kinect()
    if not kinect.available and not kinect.start():
        logger.warning("Kinect live feed could not start: %s", kinect.error)
        yield from _generate_error_mjpeg()
        return

    while True:
        frame = kinect.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.03)


def _generate_ip_mjpeg(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
):
    if _looks_like_snapshot_url(url):
        yield from _generate_snapshot_mjpeg(url, label, http_options=http_options)
        return
    yield from _generate_capture_mjpeg(url, label=label, http_options=http_options)


def _generate_capture_mjpeg(
    source,
    label: str = "camera",
    http_options: dict[str, str] | None = None,
):
    if isinstance(source, str):
        source = _normalize_camera_url(source)
        if not _is_supported_camera_url(source):
            logger.warning("Live feed has unsupported URL for %s: %s", label, _redact_url(source))
            yield from _generate_error_mjpeg()
            return
        cap = _open_stream_capture(source, label, http_options=http_options)
    else:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        logger.warning("Live feed could not open %s", label)
        yield from _generate_error_mjpeg()
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.03)
    finally:
        cap.release()


def _generate_snapshot_mjpeg(
    url: str,
    label: str,
    http_options: dict[str, str] | None = None,
):
    while True:
        try:
            options = _http_camera_options(url, http_options)
            headers = {"User-Agent": options.get("user_agent", "CacheSec/1.0")}
            if options.get("referer"):
                headers["Referer"] = options["referer"]
            if options.get("origin"):
                headers["Origin"] = options["origin"]
            req = Request(url, headers=headers)
            with urlopen(req, timeout=5) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("snapshot did not decode as an image")
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        except Exception as exc:
            logger.debug("Snapshot live feed failed for %s: %s", label, exc)
            yield from _generate_error_mjpeg(single=True)
        time.sleep(1.0)


def _generate_error_mjpeg(single: bool = False):
    img = np.zeros((240, 426, 3), dtype=np.uint8)
    cv2.putText(img, "No feed", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    payload = buf.tobytes() if ok else b""
    while True:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        if single:
            return
        time.sleep(1.0)
