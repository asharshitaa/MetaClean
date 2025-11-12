# privacy_score.py
import re
import io
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageOps
import piexif
import numpy as np
import cv2
import pytesseract

# Utility: read EXIF
def get_exif_dict(pil_image):
    try:
        exif_bytes = pil_image.info.get('exif')
        if exif_bytes:
            return piexif.load(exif_bytes)
    except Exception:
        pass
    # attempt to read via piexif.load from file bytes
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG')
        buf.seek(0)
        return piexif.load(buf.getvalue())
    except Exception:
        return {}

# --- EXIF helpers -----------------------------------------------------------

def has_gps(exif_dict):
    gps = exif_dict.get('GPS', {})
    return bool(gps)

def has_datetime(exif_dict):
    exif_sub = exif_dict.get('Exif', {})
    dt_keys = [piexif.ExifIFD.DateTimeOriginal, piexif.ExifIFD.DateTimeDigitized]
    for k in dt_keys:
        if k in exif_sub:
            return True
    # also check 0th for DateTime
    zeroth = exif_dict.get('0th', {})
    if piexif.ImageIFD.DateTime in zeroth:
        return True
    return False

def has_serial_or_unique(exif_dict):
    exif_0th = exif_dict.get('0th', {})
    exif_sub = exif_dict.get('Exif', {})
    keys = [piexif.ImageIFD.SerialNumber, piexif.ImageIFD.ImageUniqueID] if hasattr(piexif.ImageIFD, 'SerialNumber') else []
    # fallback: maker note
    if piexif.ExifIFD.MakerNote in exif_sub:
        return True
    # check for ImageUniqueID
    try:
        if piexif.ImageIFD.ImageUniqueID in exif_0th:
            return True
    except Exception:
        pass
    return False

def exif_summary(exif_dict):
    out = {}
    if not exif_dict:
        return out
    # 0th and Exif and GPS
    for ifd in ('0th', 'Exif', 'GPS', '1st'):
        sub = exif_dict.get(ifd, {})
        for k, v in sub.items():
            try:
                tagname = piexif.TAGS[ifd][k]['name']
            except Exception:
                tagname = str(k)
            out[f"{ifd}:{tagname}"] = v
    return out

def detect_faces_cv2(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_list = []
    for (x, y, w, h) in faces:
        faces_list.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
    return faces_list

def variance_of_laplacian(roi_gray):
    return cv2.Laplacian(roi_gray, cv2.CV_64F).var()


def extract_candidate_plate_rois(np_img, max_rois: int = 6, additional_regions: Optional[List[Dict[str, Any]]] = None):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    image_area = gray.shape[0] * gray.shape[1]
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1500:
            continue
        area_ratio = area / float(image_area)
        if area_ratio > 0.045 or area_ratio < 0.001:
            continue
        aspect_ratio = w / float(h) if h else 0
        if aspect_ratio < 1.5 or aspect_ratio > 6.5:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        margin = int(max(2, 0.12 * min(w, h)))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(gray.shape[1], x + w + margin)
        y1 = min(gray.shape[0], y + h + margin)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        rois.append(roi)
        if len(rois) >= max_rois:
            break
    if additional_regions:
        for reg in additional_regions:
            x = int(reg.get('x', 0))
            y = int(reg.get('y', 0))
            w = int(reg.get('w', 0))
            h = int(reg.get('h', 0))
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(gray.shape[1], x + w)
            y1 = min(gray.shape[0], y + h)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            rois.append(roi)
            if len(rois) >= max_rois:
                break
    return rois

LICENSE_REGEX = re.compile(
    r'\b(?=[A-Z0-9\- ]*[A-Z])(?=[A-Z0-9\- ]*\d)(?:[A-Z]{2,3}\s*\d{1,4}\s*[A-Z]{1,3}\s*\d{1,4}|[A-Z0-9]{2,4}-?[A-Z0-9]{2,4}-?[A-Z0-9]{2,4})\b'
)
PHONE_REGEX = re.compile(r'(\+?\d{2,4}[\s\-]?)?(\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{2,4})')
ADDRESS_KEYWORDS = [
    'no', 'no.', 'house', 'h.no', 'door', 'street', 'st', 'road', 'lane', 'sector', 'block', 'avenue'
]
EMAIL_REGEX = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.IGNORECASE)
GSTIN_REGEX = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9][Z][A-Z0-9]\b', re.IGNORECASE)
PAN_REGEX = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
LONG_NUMBER_REGEX = re.compile(r'\b\d{8,}\b')
FINANCIAL_KEYWORDS = [
    'invoice', 'inv', 'gst', 'gstin', 'tax', 'bill', 'receipt', 'amount', 'total', 'balance',
    'account', 'a/c', 'ifsc', 'upi', 'bank', 'branch', 'mrp', 'subtotal', 'due'
]


def _collect_ocr_texts(gray_img, extra_images: Optional[List[np.ndarray]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    roi_outputs: List[Dict[str, Any]] = []

    def _safe_ocr(image, config, collector: Optional[List[str]] = None):
        try:
            output = pytesseract.image_to_string(image, config=config)
            if output and output.strip():
                cleaned = output.strip()
                texts.append(cleaned)
                if collector is not None:
                    collector.append(cleaned)
        except Exception:
            pass

    def _run_variants(image, collector: Optional[List[str]] = None):
        _safe_ocr(image, '--oem 3 --psm 6', collector)
        try:
            scaled = cv2.resize(image, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            scaled = cv2.bilateralFilter(scaled, d=7, sigmaColor=55, sigmaSpace=55)
            _safe_ocr(scaled, '--oem 3 --psm 7', collector)
        except Exception:
            pass
        try:
            _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _safe_ocr(otsu, '--oem 3 --psm 6', collector)
            inv = cv2.bitwise_not(otsu)
            _safe_ocr(inv, '--oem 3 --psm 6', collector)
        except Exception:
            pass
        try:
            adaptive = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )
            _safe_ocr(adaptive, '--oem 3 --psm 6', collector)
        except Exception:
            pass

    _run_variants(gray_img)

    if extra_images:
        for idx, roi in enumerate(extra_images):
            collector: List[str] = []
            _run_variants(roi, collector)
            _safe_ocr(roi, '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', collector)
            try:
                roi_scaled = cv2.resize(roi, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
                _safe_ocr(roi_scaled, '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', collector)
            except Exception:
                pass
            try:
                roi_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
                _run_variants(roi_clahe, collector)
            except Exception:
                pass
            unique_collector = list(dict.fromkeys([c for c in collector if c.strip()]))
            roi_outputs.append({'index': idx, 'texts': unique_collector})

    seen = set()
    deduped: List[str] = []
    for txt in texts:
        if txt not in seen:
            deduped.append(txt)
            seen.add(txt)
    return deduped, roi_outputs


def classify_plate_or_sign(text: str) -> str:
    upper = text.upper()
    if any(word in upper for word in PLATE_STOPWORDS):
        return 'sign'
    cleaned = re.sub(r'[^A-Z0-9]', '', upper)
    if len(cleaned) < 5 or len(cleaned) > 10:
        return 'sign' if any(ch.isalpha() for ch in cleaned) and not any(ch.isdigit() for ch in cleaned) else 'unknown'
    letters = sum(c.isalpha() for c in cleaned)
    digits = sum(c.isdigit() for c in cleaned)
    if digits < 2 or letters < 2:
        return 'sign'
    tokens = [tok for tok in re.split(r'\s+', upper) if tok]
    if tokens and any(len(tok) > 6 and not any(ch.isdigit() for ch in tok) for tok in tokens):
        return 'sign'
    for pattern in PLATE_STRONG_REGEXES:
        if pattern.fullmatch(cleaned):
            return 'plate'
    if 5 <= len(cleaned) <= 9 and 2 <= letters <= 5 and 2 <= digits <= 6 and len(tokens) <= 3:
        return 'plate'
    return 'unknown'


PLATE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'


def detect_license_plate_regions(np_img: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
    if cascade.empty():
        return []
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    image_h, image_w = gray.shape[:2]
    regions: List[Dict[str, Any]] = []
    for (x, y, w, h) in detections:
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if aspect < 2.0 or aspect > 6.5:
            continue
        area_ratio = (w * h) / float(image_w * image_h)
        if area_ratio < 0.002 or area_ratio > 0.05:
            continue
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        edge_map = cv2.Canny(roi_gray, 60, 180)
        edge_density = float(edge_map.mean() / 255.0)
        regions.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'aspect': float(aspect),
            'area_ratio': float(area_ratio),
            'edge_density': edge_density
        })
    return regions


INVOICE_KEYWORDS = {'invoice', 'gst', 'gstin', 'tax', 'amount', 'total', 'subtotal', 'balance', 'receipt', 'bill', 'due'}
BILL_KEYWORDS = {'bill', 'mrp', 'qty', 'rate', 'customer', 'items', 'description'}
ID_CARD_KEYWORDS = {'identity', 'id', 'card', 'license', 'licence', 'passport', 'driver', 'driving', 'aadhaar', 'birth', 'national', 'authority'}
STREET_SIGN_KEYWORDS = {'street', 'road', 'avenue', 'lane', 'boulevard', 'highway', 'drive', 'st.', 'rd.', 'ave'}
CREDIT_CARD_KEYWORDS = {'valid', 'thru', 'expiry', 'exp', 'cvv', 'bank', 'debit', 'credit'}
CREDIT_CARD_REGEX = re.compile(r'(?:\d[ -]?){13,19}')
PLATE_STOPWORDS = {'STREET', 'ROAD', 'LANE', 'AVENUE', 'BLVD', 'HIGHWAY', 'DRIVE', 'EXIT', 'STOP', 'SCHOOL', 'PARK', 'AREA', 'BLOCK', 'CITY', 'CENTRE'}
PLATE_STRONG_REGEXES = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'),          # e.g. KA01AB1234
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]\d{4}$'),               # e.g. DL08C1234
    re.compile(r'^[A-Z]{3}\d{3}$'),                         # e.g. ABC123
    re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,2}$'),           # e.g. M123AB
    re.compile(r'^[A-Z]{2}\d{3,4}$'),                       # e.g. AB1234
]


def classify_document_text(text: str) -> Tuple[str, Dict[str, Any]]:
    text_lower = text.lower()
    hits: Dict[str, Any] = {'keywords': []}
    doc_type = 'document'

    if any(kw in text_lower for kw in ID_CARD_KEYWORDS):
        doc_type = 'id_card'
        hits['keywords'] = [kw for kw in ID_CARD_KEYWORDS if kw in text_lower]
    elif any(kw in text_lower for kw in INVOICE_KEYWORDS):
        doc_type = 'invoice'
        hits['keywords'] = [kw for kw in INVOICE_KEYWORDS if kw in text_lower]
    elif any(kw in text_lower for kw in BILL_KEYWORDS):
        doc_type = 'bill'
        hits['keywords'] = [kw for kw in BILL_KEYWORDS if kw in text_lower]
    elif any(kw in text_lower for kw in STREET_SIGN_KEYWORDS):
        doc_type = 'street_sign'
        hits['keywords'] = [kw for kw in STREET_SIGN_KEYWORDS if kw in text_lower]

    if doc_type in {'document', 'bill', 'invoice'} and CREDIT_CARD_REGEX.search(text):
        doc_type = 'credit_card'
        hits.setdefault('keywords', []).append('credit-card-number')
    elif doc_type in {'document'} and any(char.isdigit() for char in text):
        matches = CREDIT_CARD_REGEX.findall(text)
        if matches:
            doc_type = 'credit_card'
            hits.setdefault('keywords', []).append('credit-card-number')

    if doc_type == 'document' and any(kw in text_lower for kw in STREET_SIGN_KEYWORDS):
        doc_type = 'street_sign'
        hits['keywords'] = [kw for kw in STREET_SIGN_KEYWORDS if kw in text_lower]

    snippet = text.strip().splitlines()
    hits['snippet'] = snippet[:3]
    return doc_type, hits


def detect_document_regions(np_img: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    image_area = h * w
    detections: List[Dict[str, Any]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 0.04 * image_area or area > 0.9 * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / float(bh)
        if aspect < 0.6 or aspect > 2.8:
            continue
        area_ratio = area / float(image_area)
        roi = gray[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue
        doc_texts, _ = _collect_ocr_texts(roi)
        combined_text = '\n'.join(doc_texts)
        doc_type, meta = classify_document_text(combined_text)
        detections.append({
            'x': int(x),
            'y': int(y),
            'w': int(bw),
            'h': int(bh),
            'aspect': float(aspect),
            'area_ratio': float(area_ratio),
            'type': doc_type,
            'meta': meta
        })
        if len(detections) >= 6:
            break
    return detections


def ocr_find_sensitive(np_img, plate_regions: Optional[List[Dict[str, Any]]] = None):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    plate_rois = extract_candidate_plate_rois(np_img, additional_regions=plate_regions)
    collected_texts, roi_outputs = _collect_ocr_texts(gray, plate_rois)
    combined = '\n'.join(collected_texts).strip()

    results = {
        'raw_text': combined,
        'license_like': [],
        'address_like': [],
        'phone_like': [],
        'plate_candidates': len(plate_rois),
        'email_like': [],
        'tax_id_like': [],
        'financial_keywords': [],
        'long_number_like': [],
        'plate_text_hits': 0,
        'plate_roi_texts': roi_outputs,
        'plate_regions_detected': [dict(r) for r in (plate_regions or [])],
        'street_keywords': [],
        'plate_sign_hits': 0,
    }

    upper_text = combined.upper()
    lower_text = combined.lower()

    if combined:
        for match in LICENSE_REGEX.findall(upper_text):
            cleaned = re.sub(r'\s+', ' ', match).strip()
            if cleaned and cleaned not in results['license_like']:
                results['license_like'].append(cleaned)

        for keyword in ADDRESS_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                results['address_like'].append(keyword)

        for match in PHONE_REGEX.findall(combined):
            flat = ''.join(match)
            digits = re.sub(r'\D', '', flat)
            if len(digits) >= 7:
                cleaned = flat.strip()
                if cleaned and cleaned not in results['phone_like']:
                    results['phone_like'].append(cleaned)

        for match in EMAIL_REGEX.findall(combined):
            cleaned = match.strip()
            if cleaned and cleaned not in results['email_like']:
                results['email_like'].append(cleaned)

        for match in GSTIN_REGEX.findall(combined):
            cleaned = match.strip().upper()
            if cleaned and cleaned not in results['tax_id_like']:
                results['tax_id_like'].append(cleaned)

        for match in PAN_REGEX.findall(upper_text):
            cleaned = match.strip().upper()
            if cleaned and cleaned not in results['tax_id_like']:
                results['tax_id_like'].append(cleaned)

        for keyword in FINANCIAL_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                if keyword not in results['financial_keywords']:
                    results['financial_keywords'].append(keyword)

        for keyword in STREET_SIGN_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                if keyword not in results['street_keywords']:
                    results['street_keywords'].append(keyword)

        for match in LONG_NUMBER_REGEX.findall(combined):
            digits = ''.join(re.findall(r'\d', match))
            if len(digits) >= 8:
                cleaned = match.strip()
                if cleaned and cleaned not in results['long_number_like']:
                    results['long_number_like'].append(cleaned)

    plate_text_hits = 0
    plate_sign_hits = 0
    plate_classifications: List[str] = []
    for roi_out in roi_outputs:
        classification = 'unknown'
        for text in roi_out.get('texts', []):
            cls = classify_plate_or_sign(text)
            if cls == 'plate':
                plate_text_hits += 1
                classification = 'plate'
                break
            elif cls == 'sign':
                classification = 'sign'
        plate_classifications.append(classification)
        if classification == 'sign':
            plate_sign_hits += 1

    results['plate_text_hits'] = plate_text_hits
    results['plate_sign_hits'] = plate_sign_hits
    results['plate_classifications'] = plate_classifications
    return results

# --- Scoring helpers --------------------------------------------------------

def classify_risk(score: int) -> str:
    if score >= 75:
        return 'low'
    if score >= 45:
        return 'moderate'
    return 'high'


def metadata_penalties_exceed_threshold(penalty: int) -> bool:
    return penalty >= 40


def build_recommendations(metadata_flags: Dict[str, bool],
                          faces_info: List[Dict[str, object]],
                          ocr_hits: Dict[str, Any],
                          documents: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    recs: List[str] = []
    if metadata_flags.get('gps_present'):
        recs.append('Remove GPS location metadata before sharing (use sanitized download).')
    if metadata_flags.get('datetime_present'):
        recs.append('Strip exact capture timestamps from EXIF if they reveal sensitive timelines.')
    if metadata_flags.get('serial_present'):
        recs.append('Remove maker serial or unique IDs to avoid device fingerprinting.')
    if metadata_flags.get('exif_present') and not any(
            metadata_flags[key] for key in ('gps_present', 'datetime_present', 'serial_present')):
        recs.append('Remove the remaining EXIF fields to minimise metadata leakage.')

    if faces_info:
        if any(not f['blurred'] for f in faces_info):
            recs.append('Blur, crop, or mask visible faces before sharing publicly.')
        else:
            recs.append('Double-check blurred faces for completeness before sharing.')

    if ocr_hits.get('license_like') or ocr_hits.get('plate_text_hits'):
        recs.append('Mask vehicle number plates that appear readable in the image.')
    elif ocr_hits.get('plate_candidates'):
        recs.append('Verify signage or reflective surfaces that resemble a number plate and mask if needed.')
    if ocr_hits.get('email_like'):
        recs.append('Remove or redact email addresses before sharing.')
    if ocr_hits.get('phone_like'):
        recs.append('Obscure phone numbers in the document or image.')
    if ocr_hits.get('address_like'):
        recs.append('Hide house numbers or address signage visible in the scene.')
    if ocr_hits.get('tax_id_like'):
        recs.append('Remove tax or government identification numbers detected in the text.')
    if ocr_hits.get('financial_keywords'):
        recs.append('Avoid sharing invoices or bills publicly; redact sensitive billing details.')
    if ocr_hits.get('long_number_like'):
        recs.append('Mask account or reference numbers found in the image.')
    if ocr_hits.get('street_keywords'):
        recs.append('Consider masking street signage if it reveals your precise location.')
    if ocr_hits.get('signage_flagged'):
        recs.append('Detected street signage; verify that location details are not exposed.')

    if documents:
        for doc in documents:
            dtype = doc.get('type')
            if dtype == 'invoice' or dtype == 'bill':
                recs.append('Redact invoices or bills before sharing; remove totals, tax IDs, and customer details.')
            elif dtype == 'credit_card':
                recs.append('Remove or blur any payment card numbers shown in the image.')
            elif dtype == 'id_card':
                recs.append('Do not share ID cards publicly; cover personal identifiers.')
            elif dtype == 'document':
                recs.append('Review detected documents for personal or financial details before sharing.')
            elif dtype == 'street_sign':
                recs.append('Consider masking street signage if it reveals your precise location.')

    # Preserve order while removing duplicates
    return list(dict.fromkeys(recs))


def analyze_image_bytes(image_bytes, filename_hint='image'):
    pil = Image.open(io.BytesIO(image_bytes))
    pil = ImageOps.exif_transpose(pil).convert('RGB')
    width, height = pil.size
    np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    license_plate_regions = detect_license_plate_regions(np_img)
    document_regions = detect_document_regions(np_img)
    exif = get_exif_dict(pil)
    exif_present = bool(exif and any(exif.values()))
    gps = has_gps(exif)
    dt = has_datetime(exif)
    serial = has_serial_or_unique(exif)

    # faces
    faces = detect_faces_cv2(np_img)
    faces_info = []
    for f in faces:
        x, y, w, h = f['x'], f['y'], f['w'], f['h']
        roi = np_img[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_score = variance_of_laplacian(roi_gray)
        blur_value = float(blur_score)
        is_blurred = bool(blur_value < 100.0)
        faces_info.append({'rect': f, 'blur_var': blur_value, 'blurred': is_blurred})

    # OCR
    ocr = ocr_find_sensitive(np_img, plate_regions=license_plate_regions)

    reasons: List[str] = []

    component_penalties = {
        'metadata': {'penalty': 0, 'issues': []},
        'visual': {'penalty': 0, 'issues': []},
        'textual': {'penalty': 0, 'issues': []},
        'documents': {'penalty': 0, 'issues': []},
        'other': {'penalty': 0, 'issues': []},
    }

    # Metadata penalties
    if gps:
        component_penalties['metadata']['penalty'] += 40
        component_penalties['metadata']['issues'].append('GPS coordinates present in EXIF.')
        reasons.append('GPS coordinates found in EXIF (high risk).')
    if dt:
        component_penalties['metadata']['penalty'] += 10
        component_penalties['metadata']['issues'].append('Capture timestamp present in EXIF.')
        reasons.append('Exact Date/Time found in EXIF (time reveals).')
    if serial:
        component_penalties['metadata']['penalty'] += 10
        component_penalties['metadata']['issues'].append('Serial number or maker note present in EXIF.')
        reasons.append('Serial/unique maker info in EXIF.')
    if exif_present and not (gps or dt or serial):
        component_penalties['metadata']['penalty'] += 5
        component_penalties['metadata']['issues'].append('Other EXIF metadata present.')
        reasons.append('Other EXIF metadata present.')

    # Visual penalties (faces)
    if faces_info:
        for i, finfo in enumerate(faces_info):
            if finfo['blurred']:
                component_penalties['visual']['penalty'] += 3
                component_penalties['visual']['issues'].append(f'Face {i+1} detected but partially blurred.')
                reasons.append(f'Face {i+1} detected but blurred (less risk).')
            else:
                penalty = 12 if i == 0 else 8
                component_penalties['visual']['penalty'] += penalty
                component_penalties['visual']['issues'].append(f'Face {i+1} is clearly visible.')
                reasons.append(f'Face {i+1} detected and clearly visible (privacy risk).')

    # Textual penalties (OCR)
    plate_regions_count = len(license_plate_regions)
    plate_candidates = max(plate_regions_count, ocr.get('plate_candidates', 0))
    plate_text_hits = ocr.get('plate_text_hits', 0)
    street_keywords_hits = ocr.get('street_keywords', [])
    plate_sign_hits = ocr.get('plate_sign_hits', 0)
    signage_only_plate = bool(plate_candidates and plate_text_hits == 0 and (plate_sign_hits or street_keywords_hits))

    if ocr['license_like']:
        component_penalties['textual']['penalty'] += 60
        component_penalties['textual']['issues'].append('License plate-like text detected.')
        reasons.append('License plate-like text detected by OCR.')
    elif plate_text_hits:
        component_penalties['textual']['penalty'] += 50
        component_penalties['textual']['issues'].append('Localized text resembles a license plate.')
        reasons.append('Localized region contains plate-like text (review vehicle details).')
    elif signage_only_plate:
        component_penalties['documents']['penalty'] += 5
        component_penalties['documents']['issues'].append('Street signage detected; consider masking location info.')
        reasons.append('Street signage detected; location may be exposed.')
        plate_candidates = 0
    elif plate_candidates:
        component_penalties['textual']['penalty'] += 15
        component_penalties['textual']['issues'].append('Plate-shaped region detected with unclear text.')
        reasons.append('Possible license plate region detected; verify visibility.')

    ocr['plate_candidates'] = plate_candidates
    ocr['signage_flagged'] = bool(signage_only_plate)
    if ocr['address_like']:
        component_penalties['textual']['penalty'] += 40
        component_penalties['textual']['issues'].append('Address keywords detected in text.')
        reasons.append('Address / house number keywords found by OCR.')
    if ocr['phone_like']:
        component_penalties['textual']['penalty'] += 35
        component_penalties['textual']['issues'].append('Phone number-like text detected.')
        reasons.append('Phone-number-like text found by OCR.')
    if ocr['email_like']:
        component_penalties['textual']['penalty'] += 45
        component_penalties['textual']['issues'].append('Email address detected in text.')
        reasons.append('Email address detected in OCR text (contact detail).')
    if ocr['tax_id_like']:
        component_penalties['textual']['penalty'] += 55
        component_penalties['textual']['issues'].append('Tax or government ID detected in text.')
        reasons.append('Tax/Government ID detected in OCR text (highly sensitive).')
    if ocr['financial_keywords']:
        component_penalties['textual']['penalty'] += 30
        component_penalties['textual']['issues'].append('Financial keywords detected (invoice/billing).')
        reasons.append('Financial or billing-related keywords detected (bill/invoice).')
    if ocr['long_number_like']:
        component_penalties['textual']['penalty'] += 25
        component_penalties['textual']['issues'].append('Long numeric sequences found (possible account numbers).')
        reasons.append('Long numeric sequence detected (possible account or ID number).')

    if component_penalties['textual']['penalty'] >= 60:
        reasons.append('Sensitive text details detected; score capped for safety.')

    # Document penalties
    doc_penalty_map = {
        'invoice': 60,
        'bill': 55,
        'credit_card': 70,
        'id_card': 65,
        'document': 35,
        'street_sign': 0,
    }
    doc_reason_map = {
        'invoice': 'Document region resembles an invoice or bill.',
        'bill': 'Billing document detected (contains purchase details).',
        'credit_card': 'Payment card detected (card numbers may be visible).',
        'id_card': 'Identity card detected (personal identifiers visible).',
        'document': 'Document detected; review for sensitive information.',
        'street_sign': 'Street signage detected (review visibility).',
    }
    high_risk_doc = False
    for doc in document_regions:
        dtype = doc.get('type', 'document')
        penalty = doc_penalty_map.get(dtype, 30)
        if penalty <= 0:
            continue
        component_penalties['documents']['penalty'] += penalty
        label = doc_reason_map.get(dtype, 'Document detected.')
        snippet = doc.get('meta', {}).get('snippet', [])
        snippet_text = ', '.join(snippet[:2]) if snippet else ''
        issue = label if not snippet_text else f"{label} Sample text: {snippet_text}"
        component_penalties['documents']['issues'].append(issue)
        reasons.append(label)
        if dtype in {'invoice', 'bill', 'credit_card', 'id_card'}:
            high_risk_doc = True

    # Other penalties
    megapixels = (width * height) / 1_000_000.0
    if megapixels > 4:
        component_penalties['other']['penalty'] += 5
        component_penalties['other']['issues'].append('High resolution image exposes fine details.')
        reasons.append(f'High resolution image ({megapixels:.1f} MP) - more detail visible.')

    total_penalty = sum(comp['penalty'] for comp in component_penalties.values())
    score = max(0, min(100, int(round(100 - total_penalty))))

    score_cap = 100
    if ocr['license_like']:
        score_cap = min(score_cap, 40)
    elif plate_text_hits:
        score_cap = min(score_cap, 45)
    elif plate_candidates:
        score_cap = min(score_cap, 70)
    if ocr['tax_id_like']:
        score_cap = min(score_cap, 35)
    if ocr['email_like']:
        score_cap = min(score_cap, 45)
    if ocr['address_like']:
        score_cap = min(score_cap, 50)
    if ocr['phone_like']:
        score_cap = min(score_cap, 55)
    if ocr['financial_keywords']:
        score_cap = min(score_cap, 60)
    if ocr['long_number_like']:
        score_cap = min(score_cap, 55)
    for doc in document_regions:
        dtype = doc.get('type')
        if dtype == 'credit_card':
            score_cap = min(score_cap, 30)
        elif dtype == 'id_card':
            score_cap = min(score_cap, 35)
        elif dtype in {'invoice', 'bill'}:
            score_cap = min(score_cap, 45)
    score = min(score, score_cap)

    risk_level = classify_risk(score)
    critical_text_hits = bool(
        ocr['license_like']
        or ocr['address_like']
        or ocr['phone_like']
        or ocr['email_like']
        or ocr['tax_id_like']
        or ocr['financial_keywords']
        or ocr['long_number_like']
        or plate_text_hits
        or high_risk_doc
    )
    safe_to_share = (
        risk_level == 'low'
        and not critical_text_hits
        and not metadata_penalties_exceed_threshold(component_penalties['metadata']['penalty'])
    )

    metadata_flags = {
        'exif_present': exif_present,
        'gps_present': gps,
        'datetime_present': dt,
        'serial_present': serial,
    }
    recommendations = build_recommendations(metadata_flags, faces_info, ocr, document_regions)

    breakdown = {
        'score': score,
        'max_score': 100,
        'risk_level': risk_level,
        'safe_to_share': bool(safe_to_share),
        'width': width,
        'height': height,
        'megapixels': megapixels,
        'exif_present': exif_present,
        'gps_present': gps,
        'datetime_present': dt,
        'serial_present': serial,
        'faces': faces_info,
        'ocr': ocr,
        'license_plate_regions': license_plate_regions,
        'documents': document_regions,
        'component_penalties': component_penalties,
        'reasons': reasons,
        'recommendations': recommendations,
        'exif_summary': exif_summary(exif)
    }
    return breakdown

def remove_exif(image_bytes):
    pil = Image.open(io.BytesIO(image_bytes))
    pil = ImageOps.exif_transpose(pil)
    out_io = io.BytesIO()
    # Save without exif: PIL drop exif if not passed
    pil.save(out_io, format='JPEG', quality=95)
    return out_io.getvalue()

def blur_faces_in_bytes(image_bytes):
    pil = Image.open(io.BytesIO(image_bytes))
    pil = ImageOps.exif_transpose(pil).convert('RGB')
    np_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    faces = detect_faces_cv2(np_img)
    for f in faces:
        x, y, w, h = f['x'], f['y'], f['w'], f['h']
        roi = np_img[y:y+h, x:x+w]
        k = max(15, (w//7)|1)
        roi_blur = cv2.GaussianBlur(roi, (k,k), 0)
        np_img[y:y+h, x:x+w] = roi_blur
    out_pil = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    out_pil.save(buf, format='JPEG', quality=90)
    return buf.getvalue()
