"""
config.py — Centralised configuration loaded from environment / .env file.

All other modules import from here. Never hardcode secrets elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")


def _bool(key: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = _int("PORT", 5000)
DEBUG: bool = FLASK_ENV == "development"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(_BASE_DIR / "cachesec.db"))

# ---------------------------------------------------------------------------
# Face recognition
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD: float = _float("RECOGNITION_THRESHOLD", 0.4)
FRAME_SKIP: int = _int("FRAME_SKIP", 3)
CAMERA_INDEX: int = _int("CAMERA_INDEX", 0)
FRAME_WIDTH: int = _int("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = _int("FRAME_HEIGHT", 480)
UNKNOWN_COOLDOWN_SECONDS: int = _int("UNKNOWN_COOLDOWN_SECONDS", 10)
CAMERA_PREFERRED_SOURCE: str = os.getenv("CAMERA_PREFERRED_SOURCE", "webcam").strip().lower()
if CAMERA_PREFERRED_SOURCE not in {"webcam", "kinect"}:
    CAMERA_PREFERRED_SOURCE = "webcam"

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
RECORDINGS_DIR: str = os.getenv("RECORDINGS_DIR", str(_BASE_DIR / "recordings"))
SNAPSHOTS_DIR: str = os.getenv("SNAPSHOTS_DIR", str(_BASE_DIR / "snapshots"))
MIN_RECORDING_SECONDS: int = _int("MIN_RECORDING_SECONDS", 15)
MAX_RECORDING_SECONDS: int = _int("MAX_RECORDING_SECONDS", 5400)  # 1h30m

# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
DISCORD_COOLDOWN_SECONDS: int = _int("DISCORD_COOLDOWN_SECONDS", 60)

# ---------------------------------------------------------------------------
# Kinect
# ---------------------------------------------------------------------------
KINECT_ENABLED: bool = _bool("KINECT_ENABLED", True)   # auto-detect on startup
KINECT_TILT:    int  = _int("KINECT_TILT", 0)          # motor tilt degrees (-27 to +27)
# Keep this off for SLS: a separate motor/LED handle can block libfreenect's
# sync video/depth stream on some hosts.
KINECT_MOTOR_ENABLED: bool = _bool("KINECT_MOTOR_ENABLED", False)

# ---------------------------------------------------------------------------
# SLS / skeleton overlay
# ---------------------------------------------------------------------------
SLS_ENABLED: bool = _bool("SLS_ENABLED", True)
SLS_MODE: str = os.getenv("SLS_MODE", "night").strip().lower()
if SLS_MODE not in {"night", "always"}:
    SLS_MODE = "night"
SLS_MAX_PEOPLE: int = _int("SLS_MAX_PEOPLE", 4)

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
SOUND_ENABLED: bool = _bool("SOUND_ENABLED", True)
SOUND_GPIO_PIN: int = _int("SOUND_GPIO_PIN", 18)

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", str(_BASE_DIR / "uploads" / "faces"))
MAX_CONTENT_LENGTH: int = _int("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)  # 5 MB
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ---------------------------------------------------------------------------
# Session / Security
# ---------------------------------------------------------------------------
SESSION_LIFETIME_MINUTES: int = _int("SESSION_LIFETIME_MINUTES", 60)
SESSION_COOKIE_SECURE: bool = _bool("SESSION_COOKIE_SECURE", True)
LOGIN_MAX_ATTEMPTS: int = _int("LOGIN_MAX_ATTEMPTS", 5)
LOGIN_LOCKOUT_SECONDS: int = _int("LOGIN_LOCKOUT_SECONDS", 300)

# ---------------------------------------------------------------------------
# Proxy / Cloudflare
# ---------------------------------------------------------------------------
PROXY_COUNT: int = _int("PROXY_COUNT", 1)
_raw_hosts = os.getenv("ALLOWED_HOSTS", "")
ALLOWED_HOSTS: list[str] = [h.strip() for h in _raw_hosts.split(",") if h.strip()]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: str = os.getenv("LOG_FILE", str(_BASE_DIR / "logs" / "cachesec.log"))

# ---------------------------------------------------------------------------
# Ensure required directories exist at import time
# ---------------------------------------------------------------------------
for _d in (RECORDINGS_DIR, SNAPSHOTS_DIR, UPLOAD_FOLDER, str(_BASE_DIR / "logs")):
    Path(_d).mkdir(parents=True, exist_ok=True)
