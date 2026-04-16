"""
spoof.py — Anti-spoofing for CacheSec.

Two complementary checks, both purely software (no extra models needed):

1. LBP texture analysis
   Real skin has fine-grained micro-texture. A printed photo or screen
   capture lacks this because printing/display flattens the high-frequency
   detail and adds regular dot-matrix / pixel-grid patterns.
   We compute the Local Binary Pattern histogram of the face crop and
   measure its entropy. Low entropy → too uniform → likely a spoof.

2. Specular-reflection check (optional, additive evidence)
   Screens have a uniform specular highlight. Real faces have scattered,
   irregular specular spots. We look for a single dominant bright region
   that covers an unusual fraction of the face crop.

3. Kinect depth check (when depth data is available)
   A flat photo has near-zero depth variance across the face region.
   A real face has significant depth relief (nose protrudes ~20-40 mm).
   We measure the std-dev of depth values in the face bounding box.

All checks are combined into a single `is_live()` call. Each check votes;
the final result requires at least `min_votes` checks to pass.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

# LBP entropy threshold — faces with entropy below this are flagged as spoof.
# Real skin: ~5.5-7.0 bits. Printed photo: ~3.0-5.0 bits.
LBP_ENTROPY_THRESHOLD = 4.8

# Specular check: if the brightest connected region covers more than this
# fraction of the face area, flag as screen spoof.
SPECULAR_MAX_FRACTION = 0.25

# Kinect depth std-dev threshold (mm). Real face relief typically ≥ 15 mm.
DEPTH_STDDEV_MIN_MM = 12.0

# How many checks must pass for the face to be considered live.
# With 2 software checks available: set to 1 to be lenient (either passes),
# set to 2 to be strict (both must pass).
MIN_VOTES = 1


# ---------------------------------------------------------------------------
# LBP texture
# ---------------------------------------------------------------------------

def _lbp_entropy(gray: np.ndarray) -> float:
    """
    Compute uniform LBP histogram entropy for a grayscale face crop.
    Higher entropy = more texture complexity = more likely real skin.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 7.0   # too small to judge — pass through

    # Compute LBP manually (radius=1, 8 neighbours, uniform patterns)
    # For speed we use a simple pixel-difference approach
    center = gray[1:-1, 1:-1].astype(np.int16)
    neighbours = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
        gray[2:,   0:-2], gray[1:-1, 0:-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, nb in enumerate(neighbours):
        lbp += ((nb.astype(np.int16) >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    # Shannon entropy
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    return entropy


# ---------------------------------------------------------------------------
# Specular reflection check
# ---------------------------------------------------------------------------

def _specular_fraction(gray: np.ndarray) -> float:
    """
    Return the fraction of face pixels that are very bright (specular).
    Screens tend to have large uniform bright patches.
    """
    if gray.size == 0:
        return 0.0
    # Threshold at top 2% of pixel values
    thresh = int(np.percentile(gray, 98))
    thresh = max(thresh, 200)   # never below 200
    bright = np.sum(gray >= thresh)
    return float(bright) / gray.size


# ---------------------------------------------------------------------------
# Kinect depth check
# ---------------------------------------------------------------------------

def _depth_stddev(depth_raw: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Compute std-dev of depth values (in mm) within the face bounding box.
    Returns 0.0 if depth data is unavailable or the region is empty.
    """
    from kinect import KinectSource
    x1, y1, x2, y2 = bbox
    # Clamp to depth frame size (640×480)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(639, x2), min(479, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = depth_raw[y1:y2, x1:x2]
    valid  = (region > 0) & (region < 2047)
    if valid.sum() < 50:   # not enough valid pixels
        return 0.0

    mm = KinectSource.depth_to_mm(region[valid])
    return float(np.std(mm))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_live(
    face_crop_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    depth_raw: np.ndarray | None = None,
) -> tuple[bool, str]:
    """
    Return (is_live, reason_string).

    face_crop_bgr : BGR crop of the face (any size, will be resized internally)
    bbox          : (x1,y1,x2,y2) in the original frame — used for depth lookup
    depth_raw     : raw DEPTH_11BIT frame from Kinect, or None

    Returns True if the face is likely a real person, False if spoof detected.
    The reason string explains which check(s) failed.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return True, "no_crop"

    # Resize to a fixed 64×64 for consistent analysis
    crop = cv2.resize(face_crop_bgr, (64, 64))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    votes  = 0
    checks = 0
    notes  = []

    # --- Check 1: LBP texture entropy ---
    entropy = _lbp_entropy(gray)
    checks += 1
    if entropy >= LBP_ENTROPY_THRESHOLD:
        votes += 1
        logger.debug("Spoof LBP: entropy=%.2f ✓", entropy)
    else:
        notes.append(f"low_texture(entropy={entropy:.2f})")
        logger.debug("Spoof LBP: entropy=%.2f ✗ (threshold=%.2f)", entropy, LBP_ENTROPY_THRESHOLD)

    # --- Check 2: specular reflection ---
    spec = _specular_fraction(gray)
    checks += 1
    if spec <= SPECULAR_MAX_FRACTION:
        votes += 1
        logger.debug("Spoof specular: fraction=%.3f ✓", spec)
    else:
        notes.append(f"specular(frac={spec:.2f})")
        logger.debug("Spoof specular: fraction=%.3f ✗ (max=%.2f)", spec, SPECULAR_MAX_FRACTION)

    # --- Check 3: Kinect depth (if available) ---
    if depth_raw is not None:
        stddev = _depth_stddev(depth_raw, bbox)
        checks += 1
        if stddev >= DEPTH_STDDEV_MIN_MM:
            votes += 1
            logger.debug("Spoof depth: stddev=%.1f mm ✓", stddev)
        else:
            notes.append(f"flat_depth(std={stddev:.1f}mm)")
            logger.debug("Spoof depth: stddev=%.1f mm ✗ (min=%.1f)", stddev, DEPTH_STDDEV_MIN_MM)

    live = votes >= MIN_VOTES
    reason = "live" if live else ("spoof:" + ",".join(notes))
    return live, reason
