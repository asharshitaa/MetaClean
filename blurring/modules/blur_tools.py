import os
import shutil
import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import cv2
import pytesseract
from pathlib import Path

ROOT_DIR= Path(__file__).resolve().parent.parent
MODEL_DIR= ROOT_DIR / "models"
FACE_PROTO= MODEL_DIR / "deploy.prototxt"
FACE_WEIGHTS= MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

#min confidence 0-1
FACE_CONFIDENCE_THRESHOLD= 0.3
FACE_DNN_SCALE_FACTORS= (1.0, 1.3, 1.6)
FACE_HAAR_CASCADE= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#progress updates if needed
ProgressCallback= Optional[Callable[[str, dict], None]]

facenet: Optional[cv2.dnn_Net]= None
platecascade: Optional[cv2.CascadeClassifier]= None

def suppress(rects:List[Tuple[int, int, int, int]], conf:Optional[List[float]] = None,
                         thresh:float = 0.3) -> List[Tuple[int, int, int, int]]:
    if not rects:
        return []
    if conf is None:
        conf= [1.0]*len(rects)
    arr= np.array(
        [[x, y, x + w, y + h, score] for (x, y, w, h), score in zip(rects, conf)],
        dtype=np.float32
    )
    x1= arr[:, 0]
    y1= arr[:, 1]
    x2= arr[:, 2]
    y2= arr[:, 3]
    sc= arr[:, 4]

    area= (x2-x1+1)*(y2-y1+1)
    ord= np.argsort(sc)[::-1]
    tokeep: List[int]= []

    while ord.size>0:
        i= ord[0]
        tokeep.append(i)
        xx1= np.maximum(x1[i], x1[ord[1:]])
        yy1= np.maximum(y1[i], y1[ord[1:]])
        xx2= np.minimum(x2[i], x2[ord[1:]])
        yy2= np.minimum(y2[i], y2[ord[1:]])

        w= np.maximum(0.0, xx2-xx1+1)
        h= np.maximum(0.0, yy2-yy1+1)
        inte= w * h
        ovr= inte/(area[i]+area[ord[1:]]-inte)

        valid= np.where(ovr<=thresh)[0]
        ord= ord[valid + 1]

    return [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in tokeep]

def plate_get()-> Optional[cv2.CascadeClassifier]:
    global platecascade
    if platecascade is None:
        cpath= cv2.data.haarcascades+"haarcascade_russian_plate_number.xml"
        cas= cv2.CascadeClassifier(cpath)
        platecascade= cas if not cas.empty() else None
    return platecascade

def progress(cb:ProgressCallback, event:str, **payload) -> None:
    if cb:
        cb(event, payload)


def face_ensure() -> cv2.dnn_Net:
    global facenet
    if facenet is None:
        if not FACE_PROTO.exists() or not FACE_WEIGHTS.exists():
            raise FileNotFoundError(
                "Face detection model files not found."
                f"{FACE_PROTO.name} and {FACE_WEIGHTS.name} inside {MODEL_DIR}."
            )
        facenet= cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_WEIGHTS))
    return facenet


def tesserect_ready()-> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        cmd= os.environ.get("TESSERACT_CMD")
        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd= cmd
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                return False
        which_cmd= shutil.which("tesseract")
        if which_cmd:
            pytesseract.pytesseract.tesseract_cmd= which_cmd
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                return False
        return False


def face_detect(frame: np.ndarray)-> List[Tuple[int, int, int, int]]:
    net= face_ensure()
    (oh, ow)= frame.shape[:2]
    rects: List[Tuple[int, int, int, int]]= []
    conf: List[float]= []
    si= 0
    while si<len(FACE_DNN_SCALE_FACTORS):
        scale= FACE_DNN_SCALE_FACTORS[si]
        si+=1
        if scale==1.0:
            scaled_frame= frame
        else:
            scaled_frame= cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        (h, w)= scaled_frame.shape[:2]
        blob= cv2.dnn.blobFromImage(
            cv2.resize(scaled_frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        detections= net.forward()

        i = 0
        while i<detections.shape[2]:
            confidence= float(detections[0, 0, i, 2])
            i+=1
            if confidence<FACE_CONFIDENCE_THRESHOLD:
                continue
            box= detections[0, 0, i-1, 3:7]*np.array([w, h, w, h])
            (sx, sy, ex, ey)= box.astype("int")

            sx= int(sx/scale)
            sy= int(sy/scale)
            ex= int(ex/scale)
            ey= int(ey/scale)
            sx= max(0, sx)
            sy= max(0, sy)
            ex= min(ow, ex)
            ey= min(oh, ey)
            if ex-sx<= 0 or ey-sy<= 0:
                continue
            rects.append((sx, sy, ex-sx, ey-sy))
            conf.append(confidence)

    #if DNN misses small faces this is fallback
    if not FACE_HAAR_CASCADE.empty():
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray= cv2.equalizeHist(gray)
        hits= FACE_HAAR_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        j= 0
        while j<len(hits):
            (x, y, w, h) = hits[j]
            rects.append((int(x), int(y), int(w), int(h)))
            conf.append(0.55)
            j+= 1
    return suppress(rects, conf, thresh=0.35)


def plates_detect(frame: np.ndarray)-> List[Tuple[int, int, int, int]]:
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray= cv2.bilateralFilter(gray, 9, 75, 75)
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh= clahe.apply(gray)
    edge= cv2.Canny(enh, 50, 200)
    ker= cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed= cv2.morphologyEx(edge, cv2.MORPH_CLOSE, ker, iterations=2)

    contours, _= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key=cv2.contourArea, reverse=True)[:120]

    pboxes: List[Tuple[int, int, int, int]]= []
    scores: List[float]= []
    for c in contours:
        per= cv2.arcLength(c, True)
        appr= cv2.approxPolyDP(c, 0.02 * per, True)
        if len(appr)<4:
            continue
        x, y, w, h= cv2.boundingRect(appr)
        asp= w/float(h) if h else 0
        area= w*h
        if 2.0 <= asp <= 6.5 and area > 1200 and w > 50 and h > 15:
            pboxes.append((x, y, w, h))
            scores.append(min(0.8, 0.4 + (area / (frame.shape[0] * frame.shape[1] + 1e-5))))

    cas= plate_get()
    if cas is not None:
        hits= cas.detectMultiScale(
            enh,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(60, 20)
        )
        k= 0
        while k<len(hits):
            (x, y, w, h)= hits[k]
            k+= 1
            asp = w / float(h) if h else 0
            if not 2.0 <= asp <= 6.5:
                continue
            pboxes.append((int(x), int(y), int(w), int(h)))
            scores.append(0.9)
    return suppress(pboxes, scores, thresh=0.4)


def apply_blur(frame: np.ndarray, rects: Sequence[Tuple[int, int, int, int]], ker: int = 99) -> None:
    for (x, y, w, h) in rects:
        roi= frame[y:y + h, x:x + w]
        if roi.size<= 0:
            continue
        k= ker if ker % 2 else ker + 1  # ker must be odd for OpenCV
        frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 30)


def get_textbox(frame: np.ndarray, confidence_threshold: int = 70) -> List[Tuple[int, int, int, int]]:
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var= [gray]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    var.append(clahe.apply(gray))
    var.append(cv2.GaussianBlur(gray, (3, 3), 0))
    try:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 9)
        var.append(thresh)
        var.append(cv2.bitwise_not(thresh))
    except Exception:
        pass

    rects: List[Tuple[int, int, int, int]]= []
    conf: List[float]= []
    config = "--oem 3 --psm 6"

    for v in var:
        scale= cv2.resize(v, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
        data= pytesseract.image_to_data(scale, output_type=pytesseract.Output.DICT, config=config)
        sx= v.shape[1] / scale.shape[1]
        sy= v.shape[0] / scale.shape[0]
        for i in range(len(data["level"])):
            try:
                conf= float(data["conf"][i])
            except ValueError:
                continue
            if conf< confidence_threshold:
                continue
            x= int(data["left"][i] * sx)
            y= int(data["top"][i] * sy)
            w= int(data["width"][i] * sx)
            h= int(data["height"][i] * sy)
            if w<=4 or h<=4:
                continue
            margin= 2
            rects.append((max(0, x - margin), max(0, y - margin), w + 2 * margin, h + 2 * margin))
            conf.append(conf / 100.0)
    return suppress(rects, conf, thresh=0.3)


def text_blur(frame: np.ndarray, confidence_threshold: int = 70) -> int:
    rects= get_textbox(frame, confidence_threshold=confidence_threshold)
    blurred= 0
    i = 0
    while i<len(rects):
        x, y, w, h= rects[i]
        roi= frame[y:y + h, x:x + w]
        if roi.size>0:
            frame[y:y + h, x:x + w]= cv2.GaussianBlur(roi, (25, 25), 0)
            blurred+= 1
        i+=1
    return blurred


def sensitive_blur(
    ipath: Path,
    output_dir: Path,
    *,
    auto: bool= True,
    blur_text: bool= True,
    progress_cb: ProgressCallback= None,
) -> Path:
    #detect n blur
    ipath= Path(ipath)
    if not ipath.exists():
        raise FileNotFoundError(f"Input frame not found: {ipath}")

    output_dir= Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path= output_dir / f"blurred_{ipath.name}"

    frame= cv2.imread(str(ipath))
    if frame is None:
        raise RuntimeError(f"Unable to read frame: {ipath}")

    progress(progress_cb, "load", source=str(ipath))

    faces= face_detect(frame)
    progress(progress_cb, "faces", count=len(faces), rects=faces)

    plates= plates_detect(frame)
    progress(progress_cb, "plates", count=len(plates), rects=plates)

    if auto:
        apply_blur(frame, faces)
        apply_blur(frame, plates)
        progress(progress_cb, "blur_regions", faces=len(faces), plates=len(plates))

    text_blurred= 0
    if blur_text:
        if tesserect_ready():
            text_blurred = text_blur(frame)
            progress(progress_cb, "blur_text", count=text_blurred)
        else:
            progress(progress_cb, "tesseract_missing", count=0)

    cv2.imwrite(str(output_path), frame)
    progress(progress_cb, "saved", output=str(output_path), text_blurred=text_blurred)
    return output_path


def batch_blur(
    ipaths: Iterable[Path],
    output_dir: Path,
    *,
    auto: bool= True,
    blur_text: bool= True,
    progress_cb: ProgressCallback= None,
) -> List[str]:
    processed: List[str]= []
    for ipath in ipaths:
        try:
            result= sensitive_blur(
                Path(ipath),
                Path(output_dir),
                auto=auto,
                blur_text=blur_text,
                progress_cb=progress_cb,
            )
            processed.append(str(result))
        except Exception as exc:
            progress(progress_cb, "error", source=str(ipath), error=str(exc))
    return processed


if __name__ == "__main__":
    demo_inputs = [ROOT_DIR / "demo.jpg"]
    out_dir = ROOT_DIR / "blurred_output"
    batch_blur(demo_inputs, out_dir)

