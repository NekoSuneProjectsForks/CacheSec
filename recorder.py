"""
recorder.py — Video recording logic for unknown detection events.

Recording contract:
  - Start when an unknown face is detected.
  - Keep recording while the unknown person remains visible.
  - If they disappear before MIN_RECORDING_SECONDS, hold open until min is met.
  - If they reappear while recording is active, extend (same clip, same event).
  - Stop when they have been absent for UNKNOWN_COOLDOWN_SECONDS AND min duration met.
  - Never exceed MAX_RECORDING_SECONDS; force-stop and save if hit.
  - Timestamped filenames: unknown_YYYYMMDD_HHMMSS.mp4
  - Links the recording to the event row in the DB.

Codec strategy:
  OpenCV's bundled ffmpeg does not include libx264, so writing H.264 directly
  via VideoWriter fails on Pi OS. Instead:
    1. Record to a temporary .avi file using MJPEG (always works with OpenCV).
    2. After the recording ends, spawn a background thread that calls the
       system ffmpeg (which has libx264) to re-encode to H.264 .mp4.
    3. The .avi is deleted after successful re-encode.
  This gives browser-playable H.264 mp4 without blocking the recording loop.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class RecordingState(NamedTuple):
    is_recording: bool
    event_id: int | None
    filename: str           # final .mp4 filename (may still be encoding)
    duration_seconds: float


class Recorder:
    """
    Thread-safe video recorder.

    Usage:
        recorder = Recorder()
        recorder.start_background()

        recorder.push_frame(frame)
        recorder.signal_unknown_visible(event_id)
        recorder.signal_unknown_gone()

        recorder.stop_background()
    """

    def __init__(self):
        self._frame_q:   queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._cmd_q:     queue.Queue[tuple]              = queue.Queue()
        self._state_lock = threading.Lock()

        self._is_recording      = False
        self._event_id: int | None = None
        self._writer: cv2.VideoWriter | None = None
        self._avi_path          = ""   # temp .avi being written
        self._mp4_filename      = ""   # final .mp4 name (set at start)
        self._start_time        = 0.0
        self._last_visible_time = 0.0
        self._min_duration      = config.MIN_RECORDING_SECONDS
        self._max_duration      = config.MAX_RECORDING_SECONDS
        self._save_locally      = config.SAVE_RECORDINGS_LOCALLY

        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signal_unknown_visible(self, event_id: int) -> None:
        self._cmd_q.put(("visible", event_id))

    def signal_unknown_gone(self) -> None:
        self._cmd_q.put(("gone", None))

    def push_frame(self, frame: np.ndarray) -> None:
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            pass

    def get_state(self) -> RecordingState:
        with self._state_lock:
            dur = (time.monotonic() - self._start_time) if self._is_recording else 0.0
            return RecordingState(
                is_recording=self._is_recording,
                event_id=self._event_id,
                filename=self._mp4_filename,
                duration_seconds=round(dur, 1),
            )

    def start_background(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="Recorder")
        self._thread.start()
        logger.info("Recorder background thread started")

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        self._cmd_q.put(("stop", None))
        if self._thread:
            self._thread.join(timeout=timeout)
        self._finalise_recording()
        logger.info("Recorder stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        unknown_visible = False

        while not self._stop_flag.is_set():
            # Drain command queue
            while True:
                try:
                    cmd, arg = self._cmd_q.get_nowait()
                except queue.Empty:
                    break

                if cmd == "visible":
                    unknown_visible         = True
                    self._last_visible_time = time.monotonic()
                    if not self._is_recording:
                        self._start_recording(arg)
                    elif self._event_id != arg:
                        with self._state_lock:
                            self._event_id = arg

                elif cmd == "gone":
                    unknown_visible = False

                elif cmd == "stop":
                    return

            if self._is_recording:
                try:
                    frame = self._frame_q.get(timeout=0.05)
                    if self._writer and frame is not None:
                        self._writer.write(frame)
                except queue.Empty:
                    pass

                elapsed = time.monotonic() - self._start_time
                absent  = time.monotonic() - self._last_visible_time

                max_duration = max(1, self._max_duration)
                min_duration = max(1, self._min_duration)

                if elapsed >= max_duration:
                    logger.warning("Max recording duration reached — stopping")
                    self._finalise_recording()

                elif (
                    not unknown_visible
                    and elapsed >= min_duration
                    and absent  >= 3.0   # 3-second settling gap after camera signals gone
                ):
                    logger.info("Unknown person absent %.1fs — stopping recording", absent)
                    self._finalise_recording()

            else:
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    time.sleep(0.05)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _start_recording(self, event_id: int) -> None:
        Path(config.RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)
        ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mp4_fname   = f"unknown_{ts}.mp4"
        avi_path    = str(Path(config.RECORDINGS_DIR) / f"unknown_{ts}.avi")
        min_duration = _int_setting("min_recording_seconds", config.MIN_RECORDING_SECONDS)
        max_duration = _int_setting("max_recording_seconds", config.MAX_RECORDING_SECONDS)
        save_locally = _bool_setting("save_recordings_locally", config.SAVE_RECORDINGS_LOCALLY)

        # MJPEG into AVI — always works with OpenCV's bundled ffmpeg
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps    = 15.0
        size   = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

        writer = cv2.VideoWriter(avi_path, fourcc, fps, size)
        if not writer.isOpened():
            logger.error("VideoWriter failed to open: %s", avi_path)
            writer.release()
            return

        with self._state_lock:
            self._is_recording      = True
            self._event_id          = event_id
            self._writer            = writer
            self._avi_path          = avi_path
            self._mp4_filename      = mp4_fname
            self._start_time        = time.monotonic()
            self._last_visible_time = time.monotonic()
            self._min_duration      = min_duration
            self._max_duration      = max_duration
            self._save_locally      = save_locally

        logger.info(
            "Recording started (MJPEG/AVI): %s  event=%d  save_locally=%s",
            avi_path, event_id, save_locally,
        )

        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        mp4_path   = str(Path(config.RECORDINGS_DIR) / mp4_fname)
        fields = {"recording_start": started_at}
        if save_locally:
            fields["recording_path"] = mp4_path
        else:
            fields["notes"] = "Recording will be uploaded to Discord only."
        self._update_event_recording(event_id, **fields)

    def _finalise_recording(self) -> None:
        with self._state_lock:
            if not self._is_recording:
                return
            writer      = self._writer
            event_id    = self._event_id
            avi_path    = self._avi_path
            mp4_fname   = self._mp4_filename
            start       = self._start_time
            save_locally = self._save_locally
            self._is_recording  = False
            self._writer        = None
            self._event_id      = None
            self._avi_path      = ""
            self._mp4_filename  = ""

        if writer:
            writer.release()

        duration = round(time.monotonic() - start, 1)
        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info("Raw recording closed: %s  duration=%.1fs", avi_path, duration)

        if event_id is not None:
            self._update_event_recording(event_id, recording_end=ended_at, ended_at=ended_at)

        # Re-encode in a daemon thread so we don't block anything
        mp4_path = str(Path(config.RECORDINGS_DIR) / mp4_fname)
        t = threading.Thread(
            target=self._reencode,
            args=(avi_path, mp4_path, mp4_fname, event_id, duration, ended_at, save_locally),
            daemon=True,
            name="Reencoder",
        )
        t.start()

        from discord_notify import clear_event_cooldown
        if event_id:
            clear_event_cooldown(event_id)

    def _reencode(
        self,
        avi_path: str,
        mp4_path: str,
        mp4_fname: str,
        event_id: int | None,
        duration: float,
        ended_at: str,
        save_locally: bool,
    ) -> None:
        """
        Re-encode the temp AVI to H.264 MP4 using system ffmpeg.
        Runs in a background thread after recording ends.
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # ffmpeg not found — just rename the AVI as a fallback
            logger.warning("ffmpeg not found; keeping .avi file as-is")
            avi_fname = Path(avi_path).name
            if save_locally:
                self._save_recording_row(
                    event_id=event_id,
                    filename=avi_fname,
                    file_size_bytes=_file_size(avi_path),
                    duration_seconds=duration,
                    started_at=ended_at,
                    ended_at=ended_at,
                )
                self._update_event_recording(event_id, recording_path=avi_path)
            else:
                self._upload_and_discard(avi_path, avi_fname, event_id, duration, ended_at)
            return

        logger.info("Re-encoding %s → %s", Path(avi_path).name, Path(mp4_path).name)
        cmd = [
            ffmpeg, "-y",
            "-i",       avi_path,
            "-c:v",     "libx264",
            "-preset",  "veryfast",   # fast enough for Pi; change to 'faster' for quality
            "-crf",     "26",         # quality (18=lossless, 28=medium, 23=default)
            "-movflags", "+faststart", # moov atom at front — essential for browser streaming
            "-an",                    # no audio track
            mp4_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                mp4_size = _file_size(mp4_path)
                logger.info(
                    "Re-encode complete: %s (%.1f MB)", mp4_fname, mp4_size / 1e6
                )
                # Remove the temp AVI
                try:
                    os.unlink(avi_path)
                except OSError:
                    pass
                if save_locally:
                    self._save_recording_row(
                        event_id=event_id,
                        filename=mp4_fname,
                        file_size_bytes=mp4_size,
                        duration_seconds=duration,
                        started_at=ended_at,
                        ended_at=ended_at,
                    )
                    self._update_event_recording(event_id, recording_path=mp4_path)
                else:
                    self._upload_and_discard(mp4_path, mp4_fname, event_id, duration, ended_at)
            else:
                logger.error("ffmpeg re-encode failed:\n%s", result.stderr[-2000:])
                # Fall back: keep the AVI, update DB to point at it
                avi_fname = Path(avi_path).name
                if save_locally:
                    self._save_recording_row(
                        event_id=event_id,
                        filename=avi_fname,
                        file_size_bytes=_file_size(avi_path),
                        duration_seconds=duration,
                        started_at=ended_at,
                        ended_at=ended_at,
                    )
                    self._update_event_recording(event_id, recording_path=avi_path)
                else:
                    _delete_quietly(mp4_path)
                    self._upload_and_discard(avi_path, avi_fname, event_id, duration, ended_at)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out re-encoding %s", avi_path)
            if not save_locally:
                _delete_quietly(avi_path)
                _delete_quietly(mp4_path)
                self._update_event_recording(
                    event_id,
                    webhook_error="Recording upload skipped: ffmpeg timed out.",
                    notes="Recording was not saved locally and encoding timed out.",
                )
        except Exception as exc:
            logger.error("Re-encode error: %s", exc)
            if not save_locally:
                _delete_quietly(avi_path)
                _delete_quietly(mp4_path)
                self._update_event_recording(
                    event_id,
                    webhook_error=f"Recording upload skipped: {exc}"[:500],
                    notes="Recording was not saved locally after encoding error.",
                )

    def _upload_and_discard(
        self,
        path: str,
        filename: str,
        event_id: int | None,
        duration: float,
        ended_at: str,
    ) -> None:
        """Upload the finished clip to Discord, then remove the local file."""
        try:
            from discord_notify import upload_recording
            success = upload_recording(
                event_id=event_id,
                recording_path=path,
                recording_filename=filename,
                duration_seconds=duration,
                ended_at=ended_at,
            )
            note = (
                f"Recording uploaded to Discord only: {filename}"
                if success else
                f"Recording was not saved locally; Discord upload failed: {filename}"
            )
            fields = {"notes": note}
            if success:
                fields["webhook_sent"] = 1
                fields["webhook_error"] = ""
            else:
                fields["webhook_sent"] = 0
            self._update_event_recording(event_id, **fields)
        finally:
            _delete_quietly(path)

    def _update_event_recording(self, event_id: int | None, **fields) -> None:
        if event_id is None:
            return
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.update_event(db, event_id, **fields)
        except Exception as exc:
            logger.warning("Could not update event recording fields: %s", exc)

    def _save_recording_row(self, **fields) -> None:
        try:
            from database import raw_db_ctx
            import models as m
            with raw_db_ctx() as db:
                m.create_recording(db, **fields)
        except Exception as exc:
            logger.warning("Could not save recording row: %s", exc)


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _delete_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _bool_setting(key: str, default: bool) -> bool:
    try:
        from database import get_setting
        value = get_setting(key, "true" if default else "false")
    except Exception:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_setting(key: str, default: int) -> int:
    try:
        from database import get_setting
        value = int(get_setting(key, str(default)))
    except Exception:
        return default
    return value if value > 0 else default


_recorder: Recorder | None = None


def get_recorder() -> Recorder:
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder
