"""
skeleton.py — SLS-style stick figure overlay for CacheSec night vision.

Approach: uses the Kinect depth frame to find human-shaped blobs and fits
a stick figure to the silhouette using contour analysis + depth segmentation.
No external pose model required — works entirely with OpenCV and the depth
data we already have from the Kinect.

Pipeline per frame:
  1. Depth threshold: isolate pixels within a reasonable person range (0.5–4m)
  2. Morphological clean-up to remove noise
  3. Find contours — filter by area to find person-sized blobs
  4. For each blob: fit a bounding box, estimate joint positions geometrically
  5. Draw the stick figure skeleton over the IR display frame

The joint positions are estimated from the bounding box proportions using
average human body proportions (head 1/8 height, shoulders at 1/4, etc.).
This gives a convincing SLS-like result without a neural network.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Depth range for person detection (raw DEPTH_11BIT units)
# Approx: 500 = ~0.5m, 2000 = ~4m  (values ≥ 2047 = invalid)
# ---------------------------------------------------------------------------
DEPTH_NEAR = 200
DEPTH_FAR  = 1800

# Minimum blob area in pixels to count as a person (at 640×480)
MIN_PERSON_AREA = 3000
MAX_PERSON_AREA = 150000

# Skeleton colours — bright cyan for SLS look
JOINT_COLOUR  = (255, 255, 0)    # cyan (BGR)
BONE_COLOUR   = (180, 255, 0)    # slightly dimmer cyan
JOINT_RADIUS  = 6
BONE_THICKNESS = 2
HEAD_COLOUR   = (255, 255, 255)  # white head circle

# ---------------------------------------------------------------------------
# Joint estimation from bounding box
# ---------------------------------------------------------------------------

def _estimate_joints(x: int, y: int, w: int, h: int) -> dict[str, tuple[int, int]]:
    """
    Estimate 10 joint positions from a bounding box using body proportions.

    Proportions (fraction of bounding box height from top):
      head_centre  : 0.07
      neck         : 0.15
      l_shoulder   : 0.22   r_shoulder : 0.22
      l_elbow      : 0.38   r_elbow    : 0.38
      l_wrist      : 0.52   r_wrist    : 0.52
      l_hip        : 0.55   r_hip      : 0.55
      l_knee       : 0.73   r_knee     : 0.73
      l_ankle      : 0.90   r_ankle    : 0.90
    """
    cx = x + w // 2
    # Shoulder width ~40% of bounding box width
    sw = int(w * 0.20)
    # Hip width ~25%
    hw = int(w * 0.13)

    def pt(fx, fy) -> tuple[int, int]:
        return (int(x + w * fx), int(y + h * fy))

    return {
        "head":      (cx, int(y + h * 0.07)),
        "neck":      (cx, int(y + h * 0.15)),
        "l_shoulder": (cx - sw, int(y + h * 0.22)),
        "r_shoulder": (cx + sw, int(y + h * 0.22)),
        "l_elbow":   (cx - int(w * 0.30), int(y + h * 0.38)),
        "r_elbow":   (cx + int(w * 0.30), int(y + h * 0.38)),
        "l_wrist":   (cx - int(w * 0.33), int(y + h * 0.52)),
        "r_wrist":   (cx + int(w * 0.33), int(y + h * 0.52)),
        "l_hip":     (cx - hw, int(y + h * 0.55)),
        "r_hip":     (cx + hw, int(y + h * 0.55)),
        "l_knee":    (cx - int(w * 0.14), int(y + h * 0.73)),
        "r_knee":    (cx + int(w * 0.14), int(y + h * 0.73)),
        "l_ankle":   (cx - int(w * 0.13), int(y + h * 0.90)),
        "r_ankle":   (cx + int(w * 0.13), int(y + h * 0.90)),
    }


# Skeleton bone connections
_BONES = [
    ("neck",      "head"),
    ("neck",      "l_shoulder"),
    ("neck",      "r_shoulder"),
    ("l_shoulder","l_elbow"),
    ("l_elbow",   "l_wrist"),
    ("r_shoulder","r_elbow"),
    ("r_elbow",   "r_wrist"),
    ("l_shoulder","l_hip"),
    ("r_shoulder","r_hip"),
    ("l_hip",     "r_hip"),
    ("l_hip",     "l_knee"),
    ("l_knee",    "l_ankle"),
    ("r_hip",     "r_knee"),
    ("r_knee",    "r_ankle"),
]


def _draw_skeleton(frame: np.ndarray, joints: dict[str, tuple[int, int]]) -> None:
    """Draw bones and joints onto frame in-place."""
    # Draw bones first (underneath joints)
    for a, b in _BONES:
        if a in joints and b in joints:
            cv2.line(frame, joints[a], joints[b], BONE_COLOUR, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for name, pt in joints.items():
        if name == "head":
            cv2.circle(frame, pt, JOINT_RADIUS + 4, HEAD_COLOUR, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, JOINT_RADIUS, JOINT_COLOUR, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overlay_skeletons(
    display_frame: np.ndarray,
    depth_raw: np.ndarray | None,
    max_people: int = 4,
) -> np.ndarray:
    """
    Given an IR display frame (BGR) and a raw depth frame (uint16 DEPTH_11BIT),
    detect human silhouettes in the depth data and overlay stick figures.

    Returns the annotated frame (modifies in-place and returns it).
    If depth_raw is None, returns the frame unchanged.
    """
    if depth_raw is None:
        return display_frame

    try:
        # 1. Threshold depth to isolate person-range pixels
        mask = ((depth_raw > DEPTH_NEAR) & (depth_raw < DEPTH_FAR)).astype(np.uint8) * 255

        # 2. Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return display_frame

        # 4. Filter by area and aspect ratio (person-shaped)
        people = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_PERSON_AREA <= area <= MAX_PERSON_AREA):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)
            # Person aspect ratio roughly 1.5–4.0 (taller than wide)
            if aspect < 1.0:
                continue
            people.append((area, x, y, w, h))

        # Sort by area descending, take top N
        people.sort(reverse=True)
        people = people[:max_people]

        # 5. Draw skeleton for each detected person
        for _, x, y, w, h in people:
            joints = _estimate_joints(x, y, w, h)
            _draw_skeleton(display_frame, joints)

    except Exception as exc:
        logger.debug("Skeleton overlay error: %s", exc)

    return display_frame
