"""
tapo_control.py — Helpers for using TP-Link Tapo cameras as a video source.

Tapo C-series cameras expose a standard RTSP feed at
  rtsp://<user>:<pass>@<host>:554/<stream1|stream2>
authenticated with the "Camera Account" credentials set in the Tapo app
under Advanced Settings → Camera Account.

This module just builds that URL from settings; no pytapo dependency.
PTZ / presets / privacy mode controls were removed because Tapo's
local HTTP API requires the cloud account password and is sensitive to
auth-failure lockouts that are hard to recover from over the wire.
"""

from __future__ import annotations

from urllib.parse import quote

import config


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


def tapo_configured() -> bool:
    s = tapo_settings()
    return bool(s["host"] and s["password"])
