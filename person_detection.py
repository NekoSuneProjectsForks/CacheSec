"""
person_detection.py - optional object and motion detection helpers.

Detectron2 is intentionally optional. CacheSec's face recognition remains the
main identity pipeline; this module can add COCO object detection so cameras
can still trigger on bodies, vehicles, bags, animals, and other objects when a
usable face is not visible.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

OBJECT_DETECTION_MODES = {"person", "people_pets", "all"}
PEOPLE_PET_CLASS_IDS = {
    0,   # person
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}


@dataclass(slots=True)
class ObjectDetection:
    bbox: tuple[int, int, int, int]
    score: float
    label: str
    class_id: int = -1
    source: str = "detectron2"


# Compatibility alias for code/extensions that imported the first version.
PersonDetection = ObjectDetection


class _BaseObjectDetector:
    backend = "disabled"

    def is_enabled(self) -> bool:
        return False

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        return []


class _Detectron2ObjectDetector(_BaseObjectDetector):
    backend = "detectron2"

    def __init__(self, threshold: float, device: str, mode: str):
        self._threshold = max(0.05, min(0.99, float(threshold)))
        self._device = _resolve_device(device)
        self._mode = mode if mode in OBJECT_DETECTION_MODES else "people_pets"
        self._predictor = None
        self._ready = False
        self._init()

    def _init(self) -> None:
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except Exception as exc:
            logger.warning(
                "Detectron2 object detection selected but detectron2 is unavailable: %s",
                exc,
            )
            return

        try:
            cfg = get_cfg()
            model_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            cfg.merge_from_file(model_zoo.get_config_file(model_name))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._threshold
            cfg.MODEL.DEVICE = self._device
            self._predictor = DefaultPredictor(cfg)
            self._ready = True
            logger.info(
                "Detectron2 object detector ready (%s, mode=%s, threshold=%.2f, device=%s)",
                model_name,
                self._mode,
                self._threshold,
                self._device,
            )
        except Exception as exc:
            logger.warning("Detectron2 object detector failed to initialise: %s", exc)

    def is_enabled(self) -> bool:
        return self._ready and self._predictor is not None

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        if not self.is_enabled():
            return []

        try:
            outputs = self._predictor(frame)
            instances = outputs.get("instances")
            if instances is None:
                return []
            instances = instances.to("cpu")
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
        except Exception as exc:
            logger.warning("Detectron2 detection error: %s", exc)
            return []

        h, w = frame.shape[:2]
        detections: list[ObjectDetection] = []
        for cls, score, box in zip(classes, scores, boxes):
            class_id = int(cls)
            if self._mode == "person" and class_id != 0:
                continue
            if self._mode == "people_pets" and class_id not in PEOPLE_PET_CLASS_IDS:
                continue
            if float(score) < self._threshold:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class {class_id}"
            detections.append(ObjectDetection((x1, y1, x2, y2), float(score), label, class_id))
        return detections[:30]


class MotionDetector:
    """Small per-camera frame-difference motion detector."""

    def __init__(self):
        self._previous: np.ndarray | None = None

    def detect(
        self,
        frame: np.ndarray,
        min_area: int,
        threshold: int,
    ) -> list[ObjectDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._previous is None:
            self._previous = gray
            return []

        delta = cv2.absdiff(self._previous, gray)
        self._previous = gray
        mask = cv2.threshold(delta, max(1, int(threshold)), 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[ObjectDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(1, int(min_area)):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(ObjectDetection(
                bbox=(x, y, x + w, y + h),
                score=min(1.0, area / max(frame.shape[0] * frame.shape[1], 1)),
                label="motion",
                class_id=-1,
                source="motion",
            ))
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[:20]


def _resolve_device(value: str) -> str:
    requested = (value or "auto").strip().lower()
    if requested == "cuda":
        return "cuda"
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _runtime_setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting

        value = get_setting(key, default)
    except Exception:
        return default
    return value if value is not None else default


def _runtime_float_setting(key: str, default: float) -> float:
    try:
        return float(_runtime_setting(key, str(default)))
    except (TypeError, ValueError):
        return default


_detector_lock = threading.Lock()
_detector: _BaseObjectDetector | None = None
_detector_key: tuple[str, float, str, str] | None = None


def get_object_detector() -> _BaseObjectDetector:
    global _detector, _detector_key

    backend = _runtime_setting(
        "object_detection_backend",
        _runtime_setting("person_detection_backend", config.OBJECT_DETECTION_BACKEND),
    ).strip().lower()
    if backend not in {"disabled", "detectron2"}:
        backend = "disabled"
    threshold = _runtime_float_setting(
        "object_detection_threshold",
        _runtime_float_setting("person_detection_threshold", config.OBJECT_DETECTION_THRESHOLD),
    )
    device = _runtime_setting(
        "object_detection_device",
        _runtime_setting("person_detection_device", config.OBJECT_DETECTION_DEVICE),
    ).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        device = "auto"
    mode = _runtime_setting("object_detection_mode", config.OBJECT_DETECTION_MODE).strip().lower()
    if mode not in OBJECT_DETECTION_MODES:
        mode = "people_pets"

    key = (backend, round(threshold, 4), device, mode)
    with _detector_lock:
        if _detector is not None and _detector_key == key:
            return _detector

        if backend == "detectron2":
            _detector = _Detectron2ObjectDetector(threshold, device, mode)
        else:
            _detector = _BaseObjectDetector()
        _detector_key = key
        return _detector


def get_person_detector() -> _BaseObjectDetector:
    return get_object_detector()


def draw_object_detection(frame: np.ndarray, detection: ObjectDetection) -> None:
    x1, y1, x2, y2 = detection.bbox
    color = (255, 180, 0) if detection.source == "motion" else (0, 80, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = detection.label.upper()
    if detection.score > 0:
        text += f" {detection.score:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_person_detection(frame: np.ndarray, detection: ObjectDetection, label: str = "") -> None:
    if label:
        detection = ObjectDetection(
            bbox=detection.bbox,
            score=detection.score,
            label=label,
            class_id=detection.class_id,
            source=detection.source,
        )
    draw_object_detection(frame, detection)
