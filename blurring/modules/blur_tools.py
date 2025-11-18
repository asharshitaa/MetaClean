import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_WEIGHTS = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Minimum confidence for face detection (0-1 range)
FACE_CONFIDENCE_THRESHOLD = 0.3
FACE_DNN_SCALE_FACTORS = (1.0, 1.3, 1.6)
FACE_HAAR_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Callback signature used for optional progress updates
ProgressCallback = Optional[Callable[[str, dict], None]]

_face_net: Optional[cv2.dnn_Net] = None
_plate_cascade: Optional[cv2.CascadeClassifier] = None


def _non_max_suppression(boxes: List[Tuple[int, int, int, int]], scores: Optional[List[float]] = None,
                         overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    if scores is None:
        scores = [1.0] * len(boxes)
    boxes_np = np.array(
        [[x, y, x + w, y + h, score] for (x, y, w, h), score in zip(boxes, scores)],
        dtype=np.float32
    )
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    s = boxes_np[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(s)[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return [(int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])) for i in keep]


def _get_plate_cascade() -> Optional[cv2.CascadeClassifier]:
    global _plate_cascade
    if _plate_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        _plate_cascade = cascade if not cascade.empty() else None
    return _plate_cascade


def _notify(cb: ProgressCallback, event: str, **payload) -> None:
    if cb:
        cb(event, payload)


def _ensure_face_net() -> cv2.dnn_Net:
    global _face_net
    if _face_net is None:
        if not FACE_PROTO.exists() or not FACE_WEIGHTS.exists():
            raise FileNotFoundError(
                "Face detection model files not found. Expected "
                f"{FACE_PROTO.name} and {FACE_WEIGHTS.name} inside {MODEL_DIR}."
            )
        _face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_WEIGHTS))
    return _face_net


def _tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except (EnvironmentError, pytesseract.TesseractNotFoundError):
        cmd = os.environ.get("TESSERACT_CMD")
        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd
            try:
                pytesseract.get_tesseract_version()
                return True
            except (EnvironmentError, pytesseract.TesseractNotFoundError):
                return False
        if shutil.which("tesseract"):
            pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
            try:
                pytesseract.get_tesseract_version()
                return True
            except (EnvironmentError, pytesseract.TesseractNotFoundError):
                return False
        return False


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces in an image and returns bounding boxes (x, y, w, h).
    """
    net = _ensure_face_net()
    (orig_h, orig_w) = image.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []

    for scale in FACE_DNN_SCALE_FACTORS:
        if scale == 1.0:
            scaled = image
        else:
            scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        (h, w) = scaled.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(scaled, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < FACE_CONFIDENCE_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            # Map back to original image coordinates
            start_x = int(start_x / scale)
            start_y = int(start_y / scale)
            end_x = int(end_x / scale)
            end_y = int(end_y / scale)
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(orig_w, end_x)
            end_y = min(orig_h, end_y)
            if end_x - start_x <= 0 or end_y - start_y <= 0:
                continue
            boxes.append((start_x, start_y, end_x - start_x, end_y - start_y))
            scores.append(confidence)

    # Fallback cascade when DNN misses small faces
    if FACE_HAAR_CASCADE.empty() is False:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade_boxes = FACE_HAAR_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        for (x, y, w, h) in cascade_boxes:
            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(0.55)

    return _non_max_suppression(boxes, scores, overlap_thresh=0.35)


def detect_plates(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Heuristic license plate detector based on edge and contour analysis.
    Returns bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edged = cv2.Canny(enhanced, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:120]

    plate_boxes: List[Tuple[int, int, int, int]] = []
    plate_scores: List[float] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h else 0
        area = w * h
        if 2.0 <= aspect <= 6.5 and area > 1200 and w > 50 and h > 15:
            plate_boxes.append((x, y, w, h))
            plate_scores.append(min(0.8, 0.4 + (area / (image.shape[0] * image.shape[1] + 1e-5))))

    cascade = _get_plate_cascade()
    if cascade is not None:
        cascade_boxes = cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(60, 20)
        )
        for (x, y, w, h) in cascade_boxes:
            aspect = w / float(h) if h else 0
            if not 2.0 <= aspect <= 6.5:
                continue
            plate_boxes.append((int(x), int(y), int(w), int(h)))
            plate_scores.append(0.9)

    return _non_max_suppression(plate_boxes, plate_scores, overlap_thresh=0.4)


def _blur_regions(image: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], kernel: int = 99) -> None:
    for (x, y, w, h) in boxes:
        roi = image[y:y + h, x:x + w]
        if roi.size <= 0:
            continue
        k = kernel if kernel % 2 else kernel + 1  # kernel must be odd for OpenCV
        image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 30)


def _extract_text_boxes(image: np.ndarray, confidence_threshold: int = 70) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = [gray]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))
    variants.append(cv2.GaussianBlur(gray, (3, 3), 0))
    try:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
        variants.append(thresh)
        variants.append(cv2.bitwise_not(thresh))
    except Exception:
        pass

    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    config = "--oem 3 --psm 6"

    for variant in variants:
        scaled = cv2.resize(variant, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
        data = pytesseract.image_to_data(scaled, output_type=pytesseract.Output.DICT, config=config)
        scale_x = variant.shape[1] / scaled.shape[1]
        scale_y = variant.shape[0] / scaled.shape[0]
        for i in range(len(data["level"])):
            try:
                conf = float(data["conf"][i])
            except ValueError:
                continue
            if conf < confidence_threshold:
                continue
            x = int(data["left"][i] * scale_x)
            y = int(data["top"][i] * scale_y)
            w = int(data["width"][i] * scale_x)
            h = int(data["height"][i] * scale_y)
            if w <= 4 or h <= 4:
                continue
            margin = 2
            boxes.append((max(0, x - margin), max(0, y - margin), w + 2 * margin, h + 2 * margin))
            scores.append(conf / 100.0)

    return _non_max_suppression(boxes, scores, overlap_thresh=0.3)


def _blur_text_regions(image: np.ndarray, confidence_threshold: int = 70) -> int:
    boxes = _extract_text_boxes(image, confidence_threshold=confidence_threshold)
    blurred = 0
    for (x, y, w, h) in boxes:
        roi = image[y:y + h, x:x + w]
        if roi.size <= 0:
            continue
        image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (25, 25), 0)
        blurred += 1
    return blurred


def blur_sensitive_content(
    image_path: Path,
    output_dir: Path,
    *,
    auto: bool = True,
    blur_text: bool = True,
    progress_cb: ProgressCallback = None,
) -> Path:
    """
    Processes a single image:
        * detects faces and license plates
        * blurs detected regions automatically when auto=True
        * optionally blurs text content when blur_text=True (requires Tesseract)

    Returns the output image path. Raises FileNotFoundError for missing input.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"blurred_{image_path.name}"

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    _notify(progress_cb, "load", source=str(image_path))

    faces = detect_faces(image)
    _notify(progress_cb, "faces", count=len(faces), boxes=faces)

    plates = detect_plates(image)
    _notify(progress_cb, "plates", count=len(plates), boxes=plates)

    if auto:
        _blur_regions(image, faces)
        _blur_regions(image, plates)
        _notify(progress_cb, "blur_regions", faces=len(faces), plates=len(plates))

    text_blurred = 0
    if blur_text:
        if _tesseract_available():
            text_blurred = _blur_text_regions(image)
            _notify(progress_cb, "blur_text", count=text_blurred)
        else:
            _notify(progress_cb, "tesseract_missing", count=0)

    cv2.imwrite(str(output_path), image)
    _notify(progress_cb, "saved", output=str(output_path), text_blurred=text_blurred)
    return output_path


def blur_faces_and_text(
    image_paths: Iterable[Path],
    output_dir: Path,
    *,
    auto: bool = True,
    blur_text: bool = True,
    progress_cb: ProgressCallback = None,
) -> List[str]:
    """
    Batch helper that applies `blur_sensitive_content` to all images.
    Returns a list of processed image paths (as strings).
    """
    processed: List[str] = []
    for image_path in image_paths:
        try:
            result = blur_sensitive_content(
                Path(image_path),
                Path(output_dir),
                auto=auto,
                blur_text=blur_text,
                progress_cb=progress_cb,
            )
            processed.append(str(result))
        except Exception as exc:
            _notify(progress_cb, "error", source=str(image_path), error=str(exc))
    return processed


if __name__ == "__main__":
    demo_inputs = [ROOT_DIR / "demo.jpg"]
    out_dir = ROOT_DIR / "blurred_output"
    blur_faces_and_text(demo_inputs, out_dir)

