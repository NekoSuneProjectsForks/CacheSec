"""
onvif_control.py - Optional ONVIF helpers for IP-camera night-vision control.

This module is intentionally isolated from the main camera loop so SOAP/WSDL
issues do not leak into the rest of the capture pipeline.
"""

from __future__ import annotations

import logging
import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

_REQUIRED_WSDL_FILES = (
    Path("ver10/device/wsdl/devicemgmt.wsdl"),
    Path("ver20/imaging/wsdl/imaging.wsdl"),
)


def _looks_like_wsdl_dir(path: Path) -> bool:
    return path.is_dir() and all((path / rel).is_file() for rel in _REQUIRED_WSDL_FILES)


def _candidate_wsdl_dirs(explicit: str = "") -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: str | Path | None) -> None:
        if not path:
            return
        candidate = Path(path).expanduser()
        try:
            key = str(candidate.resolve(strict=False))
        except OSError:
            key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    add(explicit)

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    add(Path(sys.prefix) / "Lib" / "site-packages" / "wsdl")
    add(Path(sys.prefix) / "lib" / pyver / "site-packages" / "wsdl")
    add(Path(sys.prefix) / "local" / "lib" / pyver / "site-packages" / "wsdl")

    try:
        for base in site.getsitepackages():
            add(Path(base) / "wsdl")
    except Exception:
        pass

    try:
        add(Path(site.getusersitepackages()) / "wsdl")
    except Exception:
        pass

    try:
        import onvif as onvif_pkg

        pkg_file = Path(onvif_pkg.__file__).resolve()
        add(pkg_file.parent / "wsdl")
        add(pkg_file.parent.parent / "wsdl")
        add(pkg_file.parent.parent.parent / "wsdl")
    except Exception:
        pass

    return candidates


def _resolve_wsdl_dir(explicit: str = "") -> Path | None:
    for candidate in _candidate_wsdl_dirs(explicit):
        if _looks_like_wsdl_dir(candidate):
            return candidate
    return None


def _stream_connection_details(url: str) -> tuple[str, str, str, int | None, str]:
    raw = (url or "").strip()
    if not raw:
        return "", "", "", None, ""

    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    return host, username, password, parsed.port, parsed.scheme.lower()


@dataclass(slots=True)
class OnvifNightVisionSettings:
    mode: str = "disabled"
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    wsdl_dir: str = ""
    stream_url: str = ""
    force_persistence: bool = False


class OnvifNightVisionController:
    """Best-effort ONVIF Imaging client focused on IrCutFilter control."""

    CONNECT_RETRY_SECS = 30.0

    def __init__(self, settings: OnvifNightVisionSettings):
        self.settings = settings
        self._client = None
        self._media_service = None
        self._imaging_service = None
        self._video_source_token = ""
        self._supported_modes: set[str] = set()
        self._cooldown_until = 0.0
        self._last_error = ""
        self._active = False
        self._state_known = False
        self._ready_logged = False

    def enabled(self) -> bool:
        return self.settings.mode in {"detect", "force_off"}

    def detects_darkness(self) -> bool:
        return self.settings.mode == "detect"

    def initial_state(self) -> bool | None:
        if not self.enabled():
            return None
        if self.settings.mode == "force_off":
            self.set_night_vision(False)
            return False
        return self._read_current_state()

    def set_night_vision(self, active: bool) -> bool:
        if not self.enabled():
            return False
        if self.settings.mode == "force_off" and active:
            return False

        desired_mode = "OFF" if active else "ON"
        if not self._ensure_ready():
            return False
        if self._supported_modes and desired_mode not in self._supported_modes:
            self._warn_once(
                "ONVIF camera does not expose IrCutFilter mode %s (supported=%s)",
                desired_mode,
                ",".join(sorted(self._supported_modes)) or "unknown",
            )
            return False

        try:
            current = self._read_imaging_settings()
            request = self._imaging_service.create_type("SetImagingSettings")
            request.VideoSourceToken = self._video_source_token
            request.ForcePersistence = bool(self.settings.force_persistence)

            if current is not None:
                try:
                    current.IrCutFilter = desired_mode
                    request.ImagingSettings = current
                except Exception:
                    request.ImagingSettings = {"IrCutFilter": desired_mode}
            else:
                request.ImagingSettings = {"IrCutFilter": desired_mode}

            self._imaging_service.SetImagingSettings(request)
        except Exception as exc:
            self._mark_failure(f"ONVIF SetImagingSettings failed: {exc}")
            return False

        self._active = active
        self._state_known = True
        self._cooldown_until = 0.0
        self._last_error = ""
        return True

    def _read_current_state(self) -> bool | None:
        current = self._read_imaging_settings()
        if current is None:
            return None

        mode = str(getattr(current, "IrCutFilter", "") or "").upper()
        if mode == "OFF":
            self._active = True
            self._state_known = True
            return True
        if mode == "ON":
            self._active = False
            self._state_known = True
            return False

        self._state_known = False
        return None

    def _read_imaging_settings(self):
        if not self._ensure_ready():
            return None
        try:
            return self._imaging_service.GetImagingSettings({
                "VideoSourceToken": self._video_source_token,
            })
        except Exception as exc:
            self._mark_failure(f"ONVIF GetImagingSettings failed: {exc}")
            return None

    def _ensure_ready(self) -> bool:
        if self._imaging_service is not None and self._video_source_token:
            return True

        now = time.monotonic()
        if now < self._cooldown_until:
            return False

        host, port, username, password = self._resolve_connection()
        if not host:
            self._warn_once("ONVIF night vision is enabled but no IP camera host is configured")
            self._cooldown_until = now + self.CONNECT_RETRY_SECS
            return False

        wsdl_dir = _resolve_wsdl_dir(self.settings.wsdl_dir)
        if wsdl_dir is None:
            self._warn_once(
                "ONVIF WSDL files were not found; install onvif-zeep or set IP_CAMERA_ONVIF_WSDL_DIR"
            )
            self._cooldown_until = now + self.CONNECT_RETRY_SECS
            return False

        try:
            from onvif import ONVIFCamera
        except Exception as exc:
            self._warn_once("ONVIF support is unavailable (%s); install onvif-zeep", exc)
            self._cooldown_until = now + self.CONNECT_RETRY_SECS
            return False

        try:
            client = ONVIFCamera(host, port, username, password, str(wsdl_dir))
            media_service = client.create_media_service()
            imaging_service = client.create_imaging_service()
            video_source_token = self._get_video_source_token(media_service)

            supported_modes: set[str] = set()
            try:
                options = imaging_service.GetOptions({"VideoSourceToken": video_source_token})
                supported_modes = {
                    str(mode).upper()
                    for mode in (getattr(options, "IrCutFilterModes", []) or [])
                    if str(mode).strip()
                }
            except Exception:
                pass

            self._client = client
            self._media_service = media_service
            self._imaging_service = imaging_service
            self._video_source_token = video_source_token
            self._supported_modes = supported_modes
            self._cooldown_until = 0.0
            self._last_error = ""

            if not self._ready_logged:
                logger.info(
                    "ONVIF night vision ready for %s:%s (video source token=%s)",
                    host, port, video_source_token,
                )
                self._ready_logged = True
            return True
        except Exception as exc:
            self._mark_failure(f"ONVIF setup failed for {host}:{port}: {exc}")
            return False

    def _resolve_connection(self) -> tuple[str, int, str, str]:
        stream_host, stream_user, stream_password, stream_port, stream_scheme = _stream_connection_details(
            self.settings.stream_url
        )

        host = self.settings.host.strip() or stream_host
        username = self.settings.username or stream_user
        password = self.settings.password or stream_password

        port = self.settings.port
        if port <= 0:
            if stream_scheme in {"http", "https"} and stream_port:
                port = stream_port
            elif stream_scheme == "https":
                port = 443
            else:
                port = 80

        return host, port, username, password

    def _get_video_source_token(self, media_service) -> str:
        try:
            sources = media_service.GetVideoSources() or []
        except Exception:
            sources = []

        for source in sources:
            token = str(getattr(source, "token", "") or getattr(source, "_token", "") or "").strip()
            if token:
                return token

        try:
            profiles = media_service.GetProfiles() or []
        except Exception:
            profiles = []

        for profile in profiles:
            config = getattr(profile, "VideoSourceConfiguration", None)
            token = str(getattr(config, "SourceToken", "") or "").strip()
            if token:
                return token

        raise RuntimeError("camera returned no ONVIF video source token")

    def _mark_failure(self, message: str) -> None:
        if message != self._last_error:
            logger.warning("%s", message)
            self._last_error = message
        self._client = None
        self._media_service = None
        self._imaging_service = None
        self._video_source_token = ""
        self._supported_modes.clear()
        self._cooldown_until = time.monotonic() + self.CONNECT_RETRY_SECS

    def _warn_once(self, message: str, *args) -> None:
        rendered = message % args if args else message
        if rendered != self._last_error:
            logger.warning("%s", rendered)
            self._last_error = rendered
