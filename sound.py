"""
sound.py — GPIO PWM buzzer driver for CacheSec on Raspberry Pi 5.

Uses lgpio directly (gpiozero's auto-detection of pin factories is broken on
Pi 5 because the old /dev/gpiomem no longer exists — it's now /dev/gpiochip0).

lgpio opens /dev/gpiochip0 which has group=gpio and an explicit ACL for the
service user, so no root privileges are needed.

Runs all sounds in daemon threads so detection never blocks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# lgpio initialisation
# ---------------------------------------------------------------------------

_handle = None   # lgpio chip handle
_pin    = None   # GPIO pin number
_lock   = threading.Lock()
_gpio_ok = False


def _init_speaker() -> bool:
    global _handle, _pin, _gpio_ok
    if not config.SOUND_ENABLED:
        return False
    try:
        import lgpio
        h = lgpio.gpiochip_open(0)   # /dev/gpiochip0 — the main GPIO bank on Pi 5
        lgpio.gpio_claim_output(h, config.SOUND_GPIO_PIN)
        _handle  = h
        _pin     = config.SOUND_GPIO_PIN
        _gpio_ok = True
        logger.info("GPIO speaker initialised (lgpio, chip=0, pin=%d)", config.SOUND_GPIO_PIN)
        return True
    except Exception as exc:
        logger.warning(
            "GPIO speaker not available (pin %d): %s — sounds disabled",
            config.SOUND_GPIO_PIN, exc,
        )
        _gpio_ok = False
        return False


# ---------------------------------------------------------------------------
# Low-level tone primitive
# ---------------------------------------------------------------------------

def _tone(freq: float, duration: float, duty: int = 50, gap: float = 0.02) -> None:
    """Play a single PWM tone. Runs synchronously — call from sound thread only."""
    if not _gpio_ok or _handle is None or _pin is None:
        return
    try:
        import lgpio
        with _lock:
            lgpio.tx_pwm(_handle, _pin, freq, duty)
        time.sleep(duration)
        with _lock:
            lgpio.tx_pwm(_handle, _pin, freq, 0)   # duty=0 = silence
        time.sleep(gap)
    except Exception as exc:
        logger.warning("Tone error: %s", exc)


# ---------------------------------------------------------------------------
# Sound sequences
# ---------------------------------------------------------------------------

def _play_access_denied_sync() -> None:
    _tone(400, 0.15)
    _tone(300, 0.15)
    _tone(200, 0.25)
    time.sleep(0.05)
    for _ in range(3):
        _tone(250, 0.05)
        time.sleep(0.03)


def _play_access_granted_sync() -> None:
    _tone(1000, 0.05)
    _tone(1400, 0.05)
    _tone(1800, 0.08)
    time.sleep(0.05)
    _tone(2200, 0.12)


def _play_alert_sync() -> None:
    for _ in range(2):
        _tone(800, 0.1)
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Public API — non-blocking
# ---------------------------------------------------------------------------

def _fire_and_forget(fn: Callable) -> None:
    t = threading.Thread(target=fn, daemon=True)
    t.start()


def play_access_denied() -> None:
    if not config.SOUND_ENABLED or not _gpio_ok:
        return
    logger.debug("Playing access_denied sound")
    _fire_and_forget(_play_access_denied_sync)


def play_access_granted() -> None:
    if not config.SOUND_ENABLED or not _gpio_ok:
        return
    logger.debug("Playing access_granted sound")
    _fire_and_forget(_play_access_granted_sync)


def play_alert() -> None:
    if not config.SOUND_ENABLED or not _gpio_ok:
        return
    _fire_and_forget(_play_alert_sync)


def shutdown() -> None:
    global _handle, _gpio_ok
    if _handle is not None:
        try:
            import lgpio
            lgpio.gpio_write(_handle, _pin, 0)
            lgpio.gpio_free(_handle, _pin)
            lgpio.gpiochip_close(_handle)
            logger.info("GPIO speaker closed")
        except Exception as exc:
            logger.warning("Error closing GPIO speaker: %s", exc)
        finally:
            _handle  = None
            _gpio_ok = False


# ---------------------------------------------------------------------------
# Initialise at import time
# ---------------------------------------------------------------------------
_init_speaker()
