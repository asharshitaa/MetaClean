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
FACE_CONFIDENCE_THRESHOLD = 0.35

# Callback signature used for optional progress updates
ProgressCallback = Optional[Callable[[str, dict], None]]

_face_net: Optional[cv2.dnn_Net] = None


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
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < FACE_CONFIDENCE_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)
        boxes.append((start_x, start_y, end_x - start_x, end_y - start_y))
    return boxes


def detect_plates(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Heuristic license plate detector based on edge and contour analysis.
    Returns bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edged = cv2.Canny(gray, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:80]

    plate_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) < 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h) if h else 0
        area = w * h
        if 2.0 <= aspect <= 6.5 and area > 1500 and w > 60 and h > 15:
            plate_boxes.append((x, y, w, h))
    return plate_boxes


def _blur_regions(image: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], kernel: int = 99) -> None:
    for (x, y, w, h) in boxes:
        roi = image[y:y + h, x:x + w]
        if roi.size <= 0:
            continue
        k = kernel if kernel % 2 else kernel + 1  # kernel must be odd for OpenCV
        image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 30)


def _blur_text_regions(image: np.ndarray, confidence_threshold: int = 70) -> int:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    blurred = 0
    for i in range(len(data["level"])):
        if int(data["conf"][i]) < confidence_threshold:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
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

