"""
tapo_control.py — Thin wrapper around pytapo for Tapo C-series cameras.

Responsibilities:
  - Build the RTSP URL for the camera (pytapo provides credentials, the URL
    follows the documented rtsp://user:pass@host:554/streamN format).
  - PTZ controls: continuous move (up/down/left/right + stop) and presets.
  - Privacy mode toggle.

Why a wrapper: pytapo is a synchronous HTTP client. Calls can block for
several seconds when the camera is slow to respond, so callers should run
these from a worker thread or accept the latency on a request thread.
The Tapo class is also re-instantiated on auth errors because session
tokens expire silently.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from urllib.parse import quote

import config

logger = logging.getLogger(__name__)


_VALID_DIRECTIONS = {"up", "down", "left", "right"}


def _live(key: str, default: str) -> str:
    try:
        from database import get_setting
        value = get_setting(key, default)
    except Exception:
        return default
    return (value if value is not None else default) or default


def tapo_settings() -> dict[str, str]:
    return {
        "host": _live("tapo_host", config.TAPO_HOST).strip(),
        "username": _live("tapo_username", config.TAPO_USERNAME).strip() or "admin",
        "password": _live("tapo_password", config.TAPO_PASSWORD),
        "cloud_password": _live("tapo_cloud_password", config.TAPO_CLOUD_PASSWORD),
        "stream": (_live("tapo_stream", config.TAPO_STREAM).strip().lower() or "stream1"),
    }


def tapo_rtsp_url(settings: dict[str, str] | None = None) -> str:
    s = settings or tapo_settings()
    if not s["host"] or not s["password"]:
        return ""
    user = quote(s["username"], safe="")
    pw = quote(s["password"], safe="")
    stream = s["stream"] if s["stream"] in {"stream1", "stream2"} else "stream1"
    return f"rtsp://{user}:{pw}@{s['host']}:554/{stream}"


class TapoController:
    """Lazy-built singleton wrapper. Rebuilds on auth failure."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: Any | None = None
        self._signature: tuple[str, str, str] = ("", "", "")
        self._last_init_error: str = ""

    def _signature_for(self, s: dict[str, str]) -> tuple[str, str, str]:
        return (s["host"], s["username"], s["cloud_password"] or s["password"])

    def _build(self, s: dict[str, str]) -> Any | None:
        if not s["host"] or not (s["password"] or s["cloud_password"]):
            self._last_init_error = "host or password not set"
            return None
        try:
            from pytapo import Tapo
        except ImportError:
            self._last_init_error = "pytapo not installed"
            logger.warning("pytapo is not installed; Tapo controls disabled")
            return None
        # pytapo accepts the camera-account password as `password` and the
        # cloud password (used for some firmware versions) as
        # `cloudPassword`. Pass both when available.
        try:
            client = Tapo(
                s["host"],
                s["username"] or "admin",
                s["password"],
                cloudPassword=s["cloud_password"] or s["password"],
            )
            self._last_init_error = ""
            return client
        except Exception as exc:
            self._last_init_error = str(exc)
            logger.warning("Tapo client init failed: %s", exc)
            return None

    def _client_for(self, s: dict[str, str]) -> Any | None:
        sig = self._signature_for(s)
        with self._lock:
            if self._client is None or sig != self._signature:
                self._client = self._build(s)
                self._signature = sig
                self._last_init_error = getattr(self, "_last_init_error", "")
            return self._client

    def _invalidate(self) -> None:
        with self._lock:
            self._client = None

    def _call(self, fn_name: str, *args, **kwargs) -> tuple[bool, str]:
        s = tapo_settings()
        client = self._client_for(s)
        if client is None:
            err = self._last_init_error or "Tapo camera not configured"
            return False, f"Tapo connection failed: {err}"
        try:
            getattr(client, fn_name)(*args, **kwargs)
            return True, ""
        except Exception as exc:
            self._invalidate()
            logger.warning("Tapo %s failed: %s", fn_name, exc)
            return False, str(exc)

    # ---- public API ----

    def move(self, direction: str, speed: int = 50) -> tuple[bool, str]:
        direction = (direction or "").strip().lower()
        if direction == "stop":
            return self._call("motorMoveStop")
        if direction not in _VALID_DIRECTIONS:
            return False, f"invalid direction '{direction}'"
        speed = max(1, min(100, int(speed)))
        return self._call("moveMotor" + direction.capitalize(), speed)

    def list_presets(self) -> list[dict[str, str]]:
        s = tapo_settings()
        client = self._client_for(s)
        if client is None:
            return []
        try:
            raw = client.getPresets() or {}
        except Exception as exc:
            self._invalidate()
            logger.debug("Tapo getPresets failed: %s", exc)
            return []
        presets: list[dict[str, str]] = []
        # pytapo returns {preset_id: name} or list-of-dicts depending on
        # firmware. Normalise both shapes.
        if isinstance(raw, dict):
            for pid, name in raw.items():
                presets.append({"id": str(pid), "name": str(name)})
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    pid = item.get("id") or item.get("preset_id") or ""
                    name = item.get("name") or item.get("preset_name") or pid
                    if pid:
                        presets.append({"id": str(pid), "name": str(name)})
        return presets

    def go_to_preset(self, preset_id: str) -> tuple[bool, str]:
        if not preset_id:
            return False, "preset id required"
        return self._call("setPreset", str(preset_id))

    def set_privacy_mode(self, enabled: bool) -> tuple[bool, str]:
        return self._call("setPrivacyMode", bool(enabled))

    def get_privacy_mode(self) -> bool | None:
        s = tapo_settings()
        client = self._client_for(s)
        if client is None:
            return None
        try:
            value = client.getPrivacyMode()
        except Exception as exc:
            self._invalidate()
            logger.debug("Tapo getPrivacyMode failed: %s", exc)
            return None
        if isinstance(value, dict):
            return str(value.get("enabled", "")).lower() in {"on", "true", "1"}
        return bool(value)


_controller: TapoController | None = None


def get_tapo_controller() -> TapoController:
    global _controller
    if _controller is None:
        _controller = TapoController()
    return _controller


def tapo_configured() -> bool:
    s = tapo_settings()
    return bool(s["host"] and s["password"])
