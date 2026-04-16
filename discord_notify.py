"""
discord_notify.py — Discord webhook notifications for unknown detections.

Features:
  - @everyone mention with event details
  - Snapshot image attachment when available
  - Per-event cooldown to prevent spam
  - Graceful failure with error logging
  - Background thread so it never blocks the detection loop
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cooldown tracking (in-memory, keyed by event_id)
# ---------------------------------------------------------------------------
_last_sent: dict[int, float] = {}   # event_id → timestamp
_cooldown_lock = threading.Lock()


def _is_on_cooldown(event_id: int) -> bool:
    with _cooldown_lock:
        last = _last_sent.get(event_id, 0)
        return (time.monotonic() - last) < config.DISCORD_COOLDOWN_SECONDS


def _mark_sent(event_id: int) -> None:
    with _cooldown_lock:
        _last_sent[event_id] = time.monotonic()


def clear_event_cooldown(event_id: int) -> None:
    """Call when an event ends so the next event can notify immediately."""
    with _cooldown_lock:
        _last_sent.pop(event_id, None)


# ---------------------------------------------------------------------------
# Webhook sender
# ---------------------------------------------------------------------------

def _build_payload(
    event_id: int,
    occurred_at: str,
    recording_started: bool,
    recording_filename: str = "",
) -> dict[str, Any]:
    """Build the Discord message embed payload."""
    ts = occurred_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Discord timestamp format
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        discord_ts = f"<t:{int(dt.timestamp())}:F>"
    except ValueError:
        discord_ts = ts

    rec_line = (
        f"Recording started: `{recording_filename}`"
        if recording_started and recording_filename
        else ("Recording started" if recording_started else "Recording NOT started")
    )

    embed = {
        "title": "⚠️  Unknown person detected",
        "color": 0xFF0000,
        "fields": [
            {"name": "Event ID",   "value": str(event_id),  "inline": True},
            {"name": "Time",       "value": discord_ts,      "inline": True},
            {"name": "Recording",  "value": rec_line,        "inline": False},
        ],
        "footer": {"text": "CacheSec · Face Recognition Security"},
        "timestamp": ts,
    }

    return {
        "content": "@everyone",
        "embeds": [embed],
    }


def _send(
    event_id: int,
    occurred_at: str,
    recording_started: bool,
    recording_filename: str = "",
    snapshot_path: str = "",
) -> bool:
    """
    Perform the actual HTTP request. Returns True on success.
    Must be run inside a daemon thread.
    """
    webhook_url = config.DISCORD_WEBHOOK_URL
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL is not set — skipping notification")
        return False

    payload = _build_payload(event_id, occurred_at, recording_started, recording_filename)

    files  = None
    file_h = None
    try:
        snap = Path(snapshot_path) if snapshot_path else None
        if snap and snap.is_file():
            file_h = open(snap, "rb")
            files  = {"file": (snap.name, file_h, "image/jpeg")}

        resp = requests.post(
            webhook_url,
            json=payload if files is None else None,
            data=payload if files is not None else None,
            files=files,
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Discord webhook sent for event %d (status %d)", event_id, resp.status_code)
        return True

    except requests.RequestException as exc:
        logger.error("Discord webhook failed for event %d: %s", event_id, exc)
        # Persist the error to the DB
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.update_event(db, event_id, webhook_error=str(exc)[:500])
        except Exception:
            pass
        return False
    finally:
        if file_h:
            file_h.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify_unknown(
    event_id: int,
    occurred_at: str,
    recording_started: bool,
    recording_filename: str = "",
    snapshot_path: str = "",
) -> None:
    """
    Send a Discord alert for an unknown detection event.

    Non-blocking: spawns a daemon thread.
    Respects per-event cooldown to avoid webhook spam.
    """
    if _is_on_cooldown(event_id):
        logger.debug("Discord cooldown active for event %d — skipping", event_id)
        return

    _mark_sent(event_id)

    def _task():
        success = _send(
            event_id=event_id,
            occurred_at=occurred_at,
            recording_started=recording_started,
            recording_filename=recording_filename,
            snapshot_path=snapshot_path,
        )
        # Update webhook_sent flag in DB
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.update_event(db, event_id, webhook_sent=1 if success else 0)
        except Exception as exc:
            logger.warning("Could not update webhook_sent: %s", exc)

    threading.Thread(target=_task, daemon=True).start()
