"""
recognition.py — Face detection, embedding, and matching.

Backend: direct ONNX Runtime (no insightface, no onnx package required)
------------------------------------------------------------------------
Models used (downloaded automatically on first run, ~30 MB total):
  - SCRFD-500MF  : face detector  (~2 MB, fast on Pi 5)
  - ArcFace-MFN  : face embedder  (~25 MB, MobileFaceNet backbone)

Why this approach over insightface:
  - insightface depends on the `onnx` package which is broken on Python 3.13
    (ml_dtypes.float4_e2m1fn AttributeError).
  - onnxruntime itself works fine on Python 3.13 / aarch64.
  - Running the ONNX models directly via onnxruntime avoids all of that.
  - Same models, same accuracy, same speed — insightface was doing exactly
    this internally.

Architecture:
  FaceRecognizer
    .detect(frame)           → list[DetectedFace]
    .embed_face(frame, bbox) → np.ndarray (512-d unit vector)
    .match(embedding)        → MatchResult | None
    .reload_gallery()        → reload enrolled embeddings from DB
    .embed_image_bytes(data) → list[np.ndarray]  (for enrollment)

Tuning:
  - Swap ARCFACE_MODEL_URL to a larger ResNet100 model for higher accuracy.
  - Lower RECOGNITION_THRESHOLD (e.g. 0.35) for stricter matching.
  - Reduce det_input_size to (320,320) for faster detection on slow hardware.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model URLs and local cache paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path.home() / ".cachesec" / "models"

# Official insightface buffalo_sc release — contains both ONNX models in one zip.
# det_500m.onnx  : SCRFD-500MF face detector  (~2 MB)
# w600k_mbf.onnx : MobileFaceNet ArcFace embedder (~25 MB)
_BUFFALO_SC_URL  = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "v0.7/buffalo_sc.zip"
)

_DETECTOR_PATH   = _MODEL_DIR / "det_500m.onnx"
_RECOGNIZER_PATH = _MODEL_DIR / "w600k_mbf.onnx"


def _download_models() -> bool:
    """
    Download the buffalo_sc zip and extract the two ONNX model files.
    Returns True if both models are available afterwards.
    """
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if _DETECTOR_PATH.exists() and _RECOGNIZER_PATH.exists():
        return True

    import zipfile, tempfile
    zip_tmp = Path(tempfile.mktemp(suffix=".zip"))
    try:
        logger.info("Downloading buffalo_sc models (~27 MB) from insightface releases …")
        req = urllib.request.Request(
            _BUFFALO_SC_URL, headers={"User-Agent": "cachesec/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_tmp, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

        with zipfile.ZipFile(zip_tmp) as zf:
            for member in zf.namelist():
                fname = Path(member).name
                dest  = _MODEL_DIR / fname
                if not dest.exists():
                    dest.write_bytes(zf.read(member))
                    logger.info("Extracted %s (%.1f MB)", fname, dest.stat().st_size / 1e6)

        return _DETECTOR_PATH.exists() and _RECOGNIZER_PATH.exists()

    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        return False
    finally:
        zip_tmp.unlink(missing_ok=True)


def _ensure_models() -> bool:
    return _download_models()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    bbox:      tuple[int, int, int, int]   # x1, y1, x2, y2
    det_score: float
    kps:       np.ndarray | None = field(default=None, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)


class MatchResult(NamedTuple):
    person_id:   int
    person_name: str
    score:       float   # cosine similarity (higher = more similar)


# ---------------------------------------------------------------------------
# SCRFD detector wrapper
# ---------------------------------------------------------------------------

class _SCRFDDetector:
    """
    SCRFD face detector wrapper around onnxruntime.

    The buffalo_sc det_500m.onnx is the raw strided model with 9 output heads:
      [score_8, score_16, score_32,
       bbox_8,  bbox_16,  bbox_32,
       kps_8,   kps_16,   kps_32]
    Each stride s produces a (H/s * W/s * 2) anchor grid.
    """

    # SCRFD uses 2 anchors per location
    _NUM_ANCHORS = 2
    _STRIDES     = [8, 16, 32]

    def __init__(self, model_path: str, input_size: tuple[int, int] = (640, 640)):
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._input_size = input_size  # (width, height)
        logger.info("SCRFD detector loaded: %s", Path(model_path).name)

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.55) -> list[DetectedFace]:
        """Detect faces in a BGR frame. Returns list of DetectedFace."""
        ih, iw = frame.shape[:2]
        tw, th = self._input_size

        # Letterbox resize
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(frame, (nw, nh))
        padded  = np.zeros((th, tw, 3), dtype=np.uint8)
        padded[:nh, :nw] = resized

        # BGR → RGB, HWC → CHW, normalise to [-1, 1]
        blob = (padded[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = self._sess.run(None, {self._input_name: blob})
        # outputs has 9 tensors: scores×3, bboxes×3, kps×3
        num_strides = len(self._STRIDES)
        score_list = outputs[0:num_strides]
        bbox_list  = outputs[num_strides:num_strides*2]
        kps_list   = outputs[num_strides*2:num_strides*3] if len(outputs) >= num_strides*3 else None

        faces = self._decode(
            score_list, bbox_list, kps_list,
            (tw, th), scale, conf_thresh, iw, ih,
        )
        return self._nms(faces)

    def _decode(
        self,
        score_list, bbox_list, kps_list,
        input_size, scale, conf_thresh, orig_w, orig_h,
    ) -> list[DetectedFace]:
        tw, th = input_size
        results: list[DetectedFace] = []

        for idx, stride in enumerate(self._STRIDES):
            scores = score_list[idx].flatten()          # (H*W*num_anchors,)
            bboxes = bbox_list[idx]                     # (H*W*num_anchors, 4)
            kps    = kps_list[idx] if kps_list else None

            fh, fw = th // stride, tw // stride

            # Build anchor centre grid  (fh*fw*num_anchors, 2)
            # Each spatial location has num_anchors anchors, interleaved.
            cy, cx = np.mgrid[0:fh, 0:fw]
            base = np.stack([cx, cy], axis=-1).reshape(fh * fw, 2)  # (fh*fw, 2)
            centres = np.repeat(base, self._NUM_ANCHORS, axis=0)     # (fh*fw*na, 2)
            centres = (centres + 0.5) * stride   # pixel coords in input space

            keep = np.where(scores >= conf_thresh)[0]
            for i in keep:
                score = float(scores[i])
                cx_, cy_ = centres[i]
                # SCRFD bbox offsets are in stride units → multiply by stride
                # centres are already in pixel space (post ×stride above)
                l, t, r, b = bboxes[i] * stride
                x1 = int(np.clip((cx_ - l) / scale, 0, orig_w))
                y1 = int(np.clip((cy_ - t) / scale, 0, orig_h))
                x2 = int(np.clip((cx_ + r) / scale, 0, orig_w))
                y2 = int(np.clip((cy_ + b) / scale, 0, orig_h))

                pts = None
                if kps is not None:
                    pts_raw = kps[i].reshape(5, 2)   # (5, 2) as (dx, dy) offsets
                    pts = np.stack([
                        np.clip((cx_ + pts_raw[:, 0] * stride) / scale, 0, orig_w),
                        np.clip((cy_ + pts_raw[:, 1] * stride) / scale, 0, orig_h),
                    ], axis=-1).astype(np.float32)

                if x2 > x1 and y2 > y1:
                    results.append(DetectedFace(
                        bbox=(x1, y1, x2, y2),
                        det_score=score,
                        kps=pts,
                    ))
        return results

    @staticmethod
    def _nms(faces: list[DetectedFace], iou_thresh: float = 0.4) -> list[DetectedFace]:
        """Simple IoU-based NMS."""
        if not faces:
            return []
        faces.sort(key=lambda f: f.det_score, reverse=True)
        kept = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            dominated = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k.bbox
                ix = max(0, min(x2, kx2) - max(x1, kx1))
                iy = max(0, min(y2, ky2) - max(y1, ky1))
                inter = ix * iy
                union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
                if union > 0 and inter / union > iou_thresh:
                    dominated = True
                    break
            if not dominated:
                kept.append(face)
        return kept[:20]


# ---------------------------------------------------------------------------
# ArcFace embedder wrapper
# ---------------------------------------------------------------------------

class _ArcFaceEmbedder:
    """
    MobileFaceNet / ArcFace embedder.
    Input: aligned 112×112 (or 64×64 depending on model) face crop.
    Output: 512-d unit embedding vector.
    """

    def __init__(self, model_path: str):
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        # Determine expected input size from model metadata
        shape = inp.shape  # e.g. [1, 3, 112, 112] or [1, 3, 64, 64]
        self._h = int(shape[2]) if shape[2] else 112
        self._w = int(shape[3]) if shape[3] else 112
        logger.info(
            "ArcFace embedder loaded: %s (input %dx%d)",
            Path(model_path).name, self._w, self._h,
        )

    def embed(self, face_img: np.ndarray) -> np.ndarray:
        """
        Given a BGR face crop (any size), return a 512-d normalised embedding.
        """
        resized = cv2.resize(face_img, (self._w, self._h))
        rgb     = resized[:, :, ::-1].astype(np.float32)
        # Normalise to [-1, 1]
        rgb     = (rgb - 127.5) / 128.0
        blob    = rgb.transpose(2, 0, 1)[np.newaxis]
        out     = self._sess.run(None, {self._input_name: blob})[0]
        vec     = out.flatten().astype(np.float32)
        norm    = np.linalg.norm(vec)
        return vec / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Frontality check — reject profile / side-on faces
# ---------------------------------------------------------------------------

def is_frontal(kps: np.ndarray | None, bbox: tuple,
               min_eye_ratio: float = 0.30,
               max_asym: float = 0.28) -> bool:
    """
    Return True only if the face is roughly front-facing.

    Two checks using the 5 facial landmarks
    (order: left-eye, right-eye, nose, left-mouth, right-mouth):

    1. Eye span ratio — the horizontal distance between the eyes must be at
       least `min_eye_ratio` × face width. A profile face has its far eye
       hidden, collapsing the inter-eye distance.

    2. Nose-to-midline asymmetry — the nose x-coordinate should sit near the
       horizontal midpoint of the two eyes. A profile face shifts the nose far
       to one side. Asymmetry = |nose_x - eye_mid_x| / eye_span.
       Reject if > max_asym.

    Falls back to True (allow through) when keypoints are absent so the
    behaviour is unchanged for detectors that don't provide landmarks.
    """
    if kps is None or kps.shape != (5, 2):
        return False  # no landmarks → not a confident frontal detection, reject

    left_eye  = kps[0]
    right_eye = kps[1]
    nose      = kps[2]

    eye_span = float(abs(right_eye[0] - left_eye[0]))
    x1, y1, x2, y2 = bbox
    face_w   = max(float(x2 - x1), 1.0)

    # Check 1: inter-eye distance relative to face width
    if eye_span / face_w < min_eye_ratio:
        return False

    # Check 2: nose horizontal symmetry relative to eye midpoint
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    asym      = abs(float(nose[0]) - eye_mid_x) / max(eye_span, 1.0)
    if asym > max_asym:
        return False

    return True


# ---------------------------------------------------------------------------
# Face alignment helper (5-point similarity transform)
# ---------------------------------------------------------------------------

# Standard 112×112 ArcFace reference landmarks
_ARCFACE_DST = np.array([
    [38.29, 51.70],
    [73.53, 51.50],
    [56.02, 71.74],
    [41.55, 92.37],
    [70.73, 92.20],
], dtype=np.float32)


def _align_face(frame: np.ndarray, kps: np.ndarray | None,
                bbox: tuple, size: int = 112) -> np.ndarray:
    """
    Align face using 5 keypoints (eyes, nose, mouth corners).
    Falls back to a simple crop+resize if keypoints are unavailable.
    """
    if kps is not None and kps.shape == (5, 2):
        try:
            src = kps.astype(np.float32)
            dst = _ARCFACE_DST * (size / 112.0)
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
            if M is not None:
                return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
        except Exception:
            pass

    # Fallback: crop bbox with small margin and resize
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size))


# ---------------------------------------------------------------------------
# Main FaceRecognizer
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """
    Thread-safe face recogniser using SCRFD + ArcFace via onnxruntime.

    Call reload_gallery() after enrolling or deleting faces.
    """

    def __init__(self, threshold: float | None = None):
        self._threshold = threshold if threshold is not None else config.RECOGNITION_THRESHOLD
        self._detector:  _SCRFDDetector | None  = None
        self._embedder:  _ArcFaceEmbedder | None = None
        self._gallery:   list[tuple[int, str, np.ndarray]] = []
        self._lock       = threading.RLock()
        self._ready      = False
        self._init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self) -> None:
        if not _ensure_models():
            logger.error(
                "One or more face recognition models could not be downloaded. "
                "Detection will be disabled until models are available."
            )
            return
        try:
            self._detector = _SCRFDDetector(
                str(_DETECTOR_PATH), input_size=(640, 640)
            )
            self._embedder = _ArcFaceEmbedder(str(_RECOGNIZER_PATH))
            self._ready = True
            logger.info("FaceRecognizer ready (SCRFD + ArcFace via onnxruntime)")
            self.reload_gallery()
        except Exception as exc:
            logger.error("FaceRecognizer init failed: %s", exc)

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def reload_gallery(self) -> int:
        from database import get_raw_db
        import models as m

        db = get_raw_db()
        try:
            rows = m.get_all_embeddings(db)
        finally:
            db.close()

        gallery = []
        for row in rows:
            try:
                emb = _deserialise_embedding(row["embedding"])
                gallery.append((row["person_id"], row["person_name"], emb))
            except Exception as exc:
                logger.warning("Skipping corrupt embedding id=%d: %s", row["id"], exc)

        with self._lock:
            self._gallery = gallery

        logger.info("Gallery reloaded: %d embedding(s)", len(gallery))
        return len(gallery)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Minimum face size as a fraction of the shorter frame dimension.
    # At 640×480 this means the face bbox must be at least 64px wide/tall.
    # Tiny distant faces or partial-head silhouettes won't meet this.
    _MIN_FACE_RATIO = 0.10

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect front-facing faces in a BGR frame. Each DetectedFace includes embedding."""
        if not self._ready or self._detector is None or self._embedder is None:
            return []
        try:
            faces = self._detector.detect(frame)
        except Exception as exc:
            logger.warning("Detection error: %s", exc)
            return []

        ih, iw = frame.shape[:2]
        min_face_px = min(ih, iw) * self._MIN_FACE_RATIO

        frontal = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            face_w = x2 - x1
            face_h = y2 - y1

            # Drop tiny/distant faces — too small to be reliably front-facing
            if face_w < min_face_px or face_h < min_face_px:
                logger.debug(
                    "Skipping small face bbox=%s (%dx%d < min %.0fpx)",
                    face.bbox, face_w, face_h, min_face_px,
                )
                continue

            # Drop profile/side-on faces using landmark geometry
            if not is_frontal(face.kps, face.bbox):
                logger.debug(
                    "Skipping non-frontal face bbox=%s det_score=%.2f",
                    face.bbox, face.det_score,
                )
                continue

            try:
                aligned = _align_face(frame, face.kps, face.bbox,
                                      size=self._embedder._h)
                face.embedding = self._embedder.embed(aligned)
            except Exception as exc:
                logger.warning("Embedding error: %s", exc)
            frontal.append(face)

        return frontal

    def match(self, embedding: np.ndarray, threshold: float | None = None) -> MatchResult | None:
        """Compare embedding against gallery. Returns best match above threshold.

        threshold is a cosine *distance* (0–1). Lower = stricter.
        Required similarity = 1.0 - threshold.
        Default 0.4 → must be ≥ 0.6 similar to match.
        """
        with self._lock:
            gallery = list(self._gallery)

        if not gallery:
            return None

        t = threshold if threshold is not None else self._threshold

        best_score  = -1.0
        best_person = None

        for person_id, name, gal_emb in gallery:
            score = float(np.dot(embedding, gal_emb))
            if score > best_score:
                best_score  = score
                best_person = (person_id, name)

        # Cosine similarity: 1.0 = identical, 0.0 = orthogonal
        # threshold=0.4 → require similarity ≥ 0.6 to recognise
        if best_score >= (1.0 - t) and best_person:
            pid, pname = best_person
            return MatchResult(person_id=pid, person_name=pname, score=best_score)
        return None

    def is_ready(self) -> bool:
        return self._ready

    def embed_image_bytes(self, image_bytes: bytes) -> list[np.ndarray]:
        """
        Given raw image bytes (PNG/JPEG), return normalised embeddings for
        every detected face. Used during enrollment.
        """
        if not self._ready:
            return []
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        faces = self.detect(frame)
        return [f.embedding for f in faces if f.embedding is not None]


# ---------------------------------------------------------------------------
# Serialisation (numpy array → SQLite BLOB)
# ---------------------------------------------------------------------------

def serialise_embedding(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _deserialise_embedding(blob: bytes) -> np.ndarray:
    buf = io.BytesIO(blob)
    arr = np.load(buf).astype(np.float32)
    norm = np.linalg.norm(arr)
    return arr / (norm + 1e-8)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_recognizer: FaceRecognizer | None = None
_recognizer_lock = threading.Lock()


def get_recognizer() -> FaceRecognizer:
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            _recognizer = FaceRecognizer()
    return _recognizer


def reload_gallery() -> int:
    return get_recognizer().reload_gallery()
