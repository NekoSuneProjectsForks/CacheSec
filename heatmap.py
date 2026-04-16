"""
heatmap.py — Detection heatmap for CacheSec.

Accumulates face detection bounding box centres into a 2D accumulator grid
and renders a colour heatmap image on demand. Thread-safe, in-memory only
(resets on restart). The admin UI fetches this as a PNG overlay.
"""

from __future__ import annotations

import threading
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Accumulator resolution — independent of frame size, scaled on render
_GRID_W = 64
_GRID_H = 48

_accum = np.zeros((_GRID_H, _GRID_W), dtype=np.float32)
_lock  = threading.Lock()
_total_hits = 0


def record_detection(bbox: tuple[int, int, int, int],
                     frame_w: int = 640, frame_h: int = 480) -> None:
    """
    Record a face detection centre into the accumulator.
    Call this every time a face is detected (unknown or recognized).
    """
    global _total_hits
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Map to grid coordinates
    gx = int(cx / frame_w * _GRID_W)
    gy = int(cy / frame_h * _GRID_H)
    gx = max(0, min(_GRID_W - 1, gx))
    gy = max(0, min(_GRID_H - 1, gy))

    with _lock:
        _accum[gy, gx] += 1.0
        _total_hits += 1


def render_heatmap(width: int = 640, height: int = 480) -> bytes:
    """
    Render the accumulator as a semi-transparent colour heatmap PNG.
    Returns raw PNG bytes ready to serve directly.
    """
    with _lock:
        data = _accum.copy()

    if data.max() < 1:
        # No data yet — return a transparent PNG
        blank = np.zeros((height, width, 4), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", blank)
        return buf.tobytes() if ok else b""

    # Upscale grid to display size
    upscaled = cv2.resize(data, (width, height), interpolation=cv2.INTER_LINEAR)

    # Gaussian blur for smooth blobs
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=width // 20)

    # Normalise to 0-255
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply COLORMAP_JET (blue=cold → red=hot)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Make low-value areas transparent (alpha proportional to intensity)
    alpha = norm.astype(np.float32) / 255.0
    alpha = np.power(alpha, 0.5)   # sqrt for more visible low counts
    alpha_u8 = (alpha * 180).astype(np.uint8)   # max 180/255 opacity

    bgra = cv2.cvtColor(coloured, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha_u8

    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes() if ok else b""


def get_stats() -> dict:
    """Return summary stats for the heatmap."""
    with _lock:
        data = _accum.copy()
        total = _total_hits

    if data.max() < 1:
        return {"total": 0, "hotspot": None}

    idx = np.unravel_index(np.argmax(data), data.shape)
    gy, gx = idx
    # Convert back to approximate frame pixel coords (640×480)
    hx = int(gx / _GRID_W * 640)
    hy = int(gy / _GRID_H * 480)
    return {
        "total":   total,
        "hotspot": {"x": hx, "y": hy, "count": int(data[gy, gx])},
    }


def reset() -> None:
    """Clear the accumulator (e.g. on a new day)."""
    global _total_hits
    with _lock:
        _accum[:] = 0
        _total_hits = 0
