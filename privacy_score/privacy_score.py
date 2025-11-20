# privacy_score.py
import re
import io
from PIL import Image, ImageOps
from typing import Dict, List, Optional, Any, Tuple
import pytesseract
import piexif
import numpy as np
import cv2

#read exif
def read_exif(pimage):
    try:
        ebyte= pimage.info.get('exif')
        if ebyte:
            return piexif.load(ebyte)
    except Exception:
        pass
    #read from load
    try:
        buf= io.BytesIO()
        pimage.save(buf, format='JPEG')
        buf.seek(0)
        return piexif.load(buf.getvalue())
    except Exception:
        return {}

def gps_is(edict):
    return bool(edict.get('GPS', {}))

def datetime_is(edict):
    esub= edict.get('Exif', {})
    dkeys= [piexif.ExifIFD.DateTimeOriginal, piexif.ExifIFD.DateTimeDigitized]
    for x in dkeys:
        if x in esub:
            return True
    zer= edict.get('0th', {})
    if piexif.ImageIFD.DateTime in zer:
        return True
    return False

def serial_is(edict):
    zero= edict.get('0th', {})
    esub= edict.get('Exif', {})
    try:
        if piexif.ExifIFD.MakerNote in esub:
            return True
    except Exception:
        pass

    try:
        if piexif.ImageIFD.ImageUniqueID in zero:
            return True
    except Exception:
        pass
    return False

def esummary(edict):
    out= {}
    if not edict:
        return out
    for ifd in ('0th', 'Exif', 'GPS', '1st'):
        sub= edict.get(ifd, {})
        items= list(sub.items())
        j= 0
        while j<len(items):
            k, v= items[j]; j+= 1
            try:
                tag= piexif.TAGS[ifd][k]['name']
            except Exception:
                tag= str(k)
            out[f"{ifd}:{tag}"] = v
    return out

def cv2_facedetect(np_img):
    gray= cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    cpath= cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    fcas= cv2.CascadeClassifier(cpath)
    faces= fcas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    flist= []
    for (x, y, w, h) in faces:
        flist.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
    return flist

def lapla_var(rgray):
    return cv2.Laplacian(rgray, cv2.CV_64F).var()


def plate_extract(np_img, max_rois: int = 6, addreg: Optional[List[Dict[str, Any]]] = None):
    gray= cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    iarea= gray.shape[0] * gray.shape[1]
    blur= cv2.bilateralFilter(gray, 11, 17, 17)
    edge= cv2.Canny(blur, 30, 200)
    contours, _= cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rois= []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h= cv2.boundingRect(c)
        ar= w * h
        if ar<1500:
            continue
        rarea= ar/float(iarea)
        if rarea>0.045 or rarea<0.001:
            continue
        aspratio=w/float(h) if h else 0
        if aspratio<1.5 or aspratio > 6.5:
            continue
        peri= cv2.arcLength(c, True)
        appr= cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(appr)!=4:
            continue

        mrg= int(max(2, 0.12*min(w, h)))
        x0= max(0, x-mrg)
        y0= max(0, y-mrg)
        x1= min(gray.shape[1], x+w+mrg)
        y1= min(gray.shape[0], y+h+mrg)
        roi= gray[y0:y1, x0:x1]
        if roi.size==0:
            continue
        rois.append(roi)
        if len(rois)>=max_rois:
            break
    if addreg:
        for r in addreg:
            x= int(r.get('x', 0))
            y= int(r.get('y', 0))
            w= int(r.get('w', 0))
            h= int(r.get('h', 0))
            if w<=0 or h<=0:
                continue
            x0= max(0, x)
            y0= max(0, y)
            x1= min(gray.shape[1], x + w)
            y1= min(gray.shape[0], y + h)
            roi= gray[y0:y1, x0:x1]
            if roi.size==0:
                continue
            rois.append(roi)
            if len(rois)>=max_rois:
                break
    return rois

LICENSE_REGEX= re.compile(
    r'\b(?=[A-Z0-9\- ]*[A-Z])(?=[A-Z0-9\- ]*\d)(?:[A-Z]{2,3}\s*\d{1,4}\s*[A-Z]{1,3}\s*\d{1,4}|[A-Z0-9]{2,4}-?[A-Z0-9]{2,4}-?[A-Z0-9]{2,4})\b'
)
PHONE_REGEX= re.compile(r'(\+?\d{2,4}[\s\-]?)?(\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{2,4})')
ADDRESS_KEYWORDS= [
    'no', 'no.', 'house', 'h.no', 'door', 'street', 'st', 'road', 'lane', 'sector', 'block', 'avenue'
]
EMAIL_REGEX= re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.IGNORECASE)
GSTIN_REGEX= re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9][Z][A-Z0-9]\b', re.IGNORECASE)
PAN_REGEX= re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
LONG_NUMBER_REGEX= re.compile(r'\b\d{8,}\b')
FINANCIAL_KEYWORDS= [
    'invoice', 'inv', 'gst', 'gstin', 'tax', 'bill', 'receipt', 'amount', 'total', 'balance',
    'account', 'a/c', 'ifsc', 'upi', 'bank', 'branch', 'mrp', 'subtotal', 'due'
]

def text_collect(gray_img, extra_images: Optional[List[np.ndarray]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str]= []
    rout: List[Dict[str, Any]]= []

    def safe(image, config, collector: Optional[List[str]] = None):
        try:
            out= pytesseract.image_to_string(image, config=config)
            if out and out.strip():
                clean= out.strip()
                texts.append(clean)
                if collector is not None:
                    collector.append(clean)
        except Exception:
            pass

    def variants_see(image, collector: Optional[List[str]] = None):
        safe(image, '--oem 3 --psm 6', collector)
        try:
            scale= cv2.resize(image, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            scale= cv2.bilateralFilter(scale, d=7, sigmaColor=55, sigmaSpace=55)
            safe(scale, '--oem 3 --psm 7', collector)
        except Exception:
            pass
        try:
            _, otsu= cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            safe(otsu, '--oem 3 --psm 6', collector)
            inv= cv2.bitwise_not(otsu)
            safe(inv, '--oem 3 --psm 6', collector)
        except Exception:
            pass
        try:
            adap= cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )
            safe(adap, '--oem 3 --psm 6', collector)
        except Exception:
            pass

    variants_see(gray_img)

    if extra_images:
        for idx, roi in enumerate(extra_images):
            collector: List[str] = []
            variants_see(roi, collector)
            safe(roi, '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', collector)
            try:
                rscaled= cv2.resize(roi, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
                safe(rscaled, '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', collector)
            except Exception:
                pass
            try:
                roi_clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
                variants_see(roi_clahe, collector)
            except Exception:
                pass
            ucoll= list(dict.fromkeys([c for c in collector if c.strip()]))
            rout.append({'index': idx, 'texts': ucoll})

    seen= set()
    deduped: List[str]= []
    for t in texts:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped, rout


def plate_sign(text: str) -> str:
    upp= text.upper()
    if any(word in upp for word in PLATE_STOPWORDS):
        return 'sign'
    clean= re.sub(r'[^A-Z0-9]', '', upp)
    if len(clean)<5 or len(clean)>10:
        return 'sign' if any(ch.isalpha() for ch in clean) and not any(ch.isdigit() for ch in clean) else 'unknown'
    lett= sum(c.isalpha() for c in clean)
    dig= sum(c.isdigit() for c in clean)
    if dig<2 or lett<2:
        return 'sign'
    tok= [t for t in re.split(r'\s+', upp) if t]
    if tok and any(len(t) > 6 and not any(ch.isdigit() for ch in t) for t in tok):
        return 'sign'
    for pattern in PLATE_STRONG_REGEXES:
        if pattern.fullmatch(clean):
            return 'plate'
    if 5<=len(clean)<=9 and 2<=lett<=5 and 2<=dig<= 6 and len(tok)<=3:
        return 'plate'
    return 'unknown'

PLATE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'

def detect_plate(np_img: np.ndarray)-> List[Dict[str, Any]]:
    gray= cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    casc= cv2.CascadeClassifier(PLATE_CASCADE_PATH)
    if casc.empty():
        return []
    det= casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    image_h, image_w= gray.shape[:2]
    regions: List[Dict[str, Any]]= []
    for (x, y, w, h) in det:
        if w<=0 or h<=0:
            continue
        asp= w/float(h)
        if asp<2.0 or asp>6.5:
            continue
        rarea=(w*h)/float(image_w*image_h)
        if rarea<0.002 or rarea>0.05:
            continue
        rgray= gray[y:y+h, x:x+w]
        if rgray.size==0:
            continue
        emap= cv2.Canny(rgray, 60, 180)
        edensity= float(emap.mean()/255.0)
        regions.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'asp': float(asp),
            'rarea': float(rarea),
            'edensity': edensity
        })
    return regions


INVOICE_KEYWORDS= {'invoice', 'gst', 'gstin', 'tax', 'amount', 'total', 'subtotal', 'balance', 'receipt', 'bill', 'due'}
BILL_KEYWORDS= {'bill', 'mrp', 'qty', 'rate', 'customer', 'items', 'description'}
ID_CARD_KEYWORDS= {'identity', 'id', 'card', 'license', 'licence', 'passport', 'driver', 'driving', 'aadhaar', 'birth', 'national', 'authority'}
STREET_SIGN_KEYWORDS= {'street', 'road', 'avenue', 'lane', 'boulevard', 'highway', 'drive', 'st.', 'rd.', 'ave'}
CREDIT_CARD_KEYWORDS= {'valid', 'thru', 'expiry', 'exp', 'cvv', 'bank', 'debit', 'credit'}
CREDIT_CARD_REGEX= re.compile(r'(?:\d[ -]?){13,19}')
PLATE_STOPWORDS= {'STREET', 'ROAD', 'LANE', 'AVENUE', 'BLVD', 'HIGHWAY', 'DRIVE', 'EXIT', 'STOP', 'SCHOOL', 'PARK', 'AREA', 'BLOCK', 'CITY', 'CENTRE'}
PLATE_STRONG_REGEXES= [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'),          # e.g. KA01AB1234
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]\d{4}$'),               # e.g. DL08C1234
    re.compile(r'^[A-Z]{3}\d{3}$'),                         # e.g. ABC123
    re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,2}$'),           # e.g. M123AB
    re.compile(r'^[A-Z]{2}\d{3,4}$'),                       # e.g. AB1234
]


def doc_text(text: str)-> Tuple[str, Dict[str, Any]]:
    ltext= text.lower()
    hits: Dict[str, Any] = {'keywords': []}
    dtype= 'document'

    if any(kw in ltext for kw in ID_CARD_KEYWORDS):
        dtype= 'id_card'
        hits['keywords']= [kw for kw in ID_CARD_KEYWORDS if kw in ltext]
    elif any(kw in ltext for kw in INVOICE_KEYWORDS):
        dtype= 'invoice'
        hits['keywords']= [kw for kw in INVOICE_KEYWORDS if kw in ltext]
    elif any(kw in ltext for kw in BILL_KEYWORDS):
        dtype= 'bill'
        hits['keywords']= [kw for kw in BILL_KEYWORDS if kw in ltext]
    elif any(kw in ltext for kw in STREET_SIGN_KEYWORDS):
        dtype= 'street_sign'
        hits['keywords']= [kw for kw in STREET_SIGN_KEYWORDS if kw in ltext]

    if dtype in {'document', 'bill', 'invoice'} and CREDIT_CARD_REGEX.search(text):
        dtype= 'credit_card'
        hits.setdefault('keywords', []).append('credit-card-number')
    elif dtype in {'document'} and any(char.isdigit() for char in text):
        mat= CREDIT_CARD_REGEX.findall(text)
        if mat:
            dtype= 'credit_card'
            hits.setdefault('keywords', []).append('credit-card-number')

    if dtype=='document' and any(kw in ltext for kw in STREET_SIGN_KEYWORDS):
        dtype='street_sign'
        hits['keywords']= [kw for kw in STREET_SIGN_KEYWORDS if kw in ltext]

    snip= text.strip().splitlines()
    hits['snip']= snip[:3]
    return dtype, hits


def doc_detect(np_img: np.ndarray)-> List[Dict[str, Any]]:
    gray= cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gray, (5, 5), 0)
    edge= cv2.Canny(blur, 50, 150)
    edge= cv2.dilate(edge, None, iterations=1)
    contours, _= cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w= gray.shape[:2]
    iarea= h * w
    det: List[Dict[str, Any]] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri= cv2.arcLength(c, True)
        appr= cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(appr)!=4:
            continue
        ar=cv2.contourArea(appr)
        if ar<0.04*iarea or ar>0.9*iarea:
            continue
        x, y, bw, bh= cv2.boundingRect(appr)
        if bw<=0 or bh<=0:
            continue
        asp= bw/float(bh)
        if asp<0.6 or asp>2.8:
            continue
        rarea= ar/float(iarea)
        roi= gray[y:y+bh, x:x+bw]
        if roi.size==0:
            continue
        doc_texts, _= text_collect(roi)
        combtext= '\n'.join(doc_texts)
        dtype, meta= doc_text(combtext)
        det.append({
            'x': int(x),
            'y': int(y),
            'w': int(bw),
            'h': int(bh),
            'asp': float(asp),
            'rarea': float(rarea),
            'type': dtype,
            'meta': meta
        })
        if len(det)>=6:
            break
    return det


def sens_find(np_img, plate_regions: Optional[List[Dict[str, Any]]] = None):
    gray= cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    prois= plate_extract(np_img, addreg=plate_regions)
    ctext, rout= text_collect(gray, prois)
    combined= '\n'.join(ctext).strip()

    results= {
        'raw_text': combined,
        'license_like': [],
        'address_like': [],
        'phone_like': [],
        'pcand': len(prois),
        'email_like': [],
        'tax_id_like': [],
        'financial_keywords': [],
        'long_number_like': [],
        'pthit': 0,
        'plate_roi_texts': rout,
        'plate_regions_detected': [dict(r) for r in (plate_regions or [])],
        'street_keywords': [],
        'pshit': 0,
    }

    utext= combined.upper()
    ltext= combined.lower()

    if combined:
        for match in LICENSE_REGEX.findall(utext):
            clean= re.sub(r'\s+', ' ', match).strip()
            if clean and clean not in results['license_like']:
                results['license_like'].append(clean)

        for k in ADDRESS_KEYWORDS:
            if re.search(r'\b' + re.escape(k) + r'\b', ltext):
                results['address_like'].append(k)

        for match in PHONE_REGEX.findall(combined):
            flat= ''.join(match)
            dig= re.sub(r'\D', '', flat)
            if len(dig)>=7:
                clean= flat.strip()
                if clean and clean not in results['phone_like']:
                    results['phone_like'].append(clean)

        for match in EMAIL_REGEX.findall(combined):
            clean= match.strip()
            if clean and clean not in results['email_like']:
                results['email_like'].append(clean)

        for match in GSTIN_REGEX.findall(combined):
            clean= match.strip().upper()
            if clean and clean not in results['tax_id_like']:
                results['tax_id_like'].append(clean)

        for match in PAN_REGEX.findall(utext):
            clean= match.strip().upper()
            if clean and clean not in results['tax_id_like']:
                results['tax_id_like'].append(clean)

        for k in FINANCIAL_KEYWORDS:
            if re.search(r'\b'+re.escape(k)+r'\b', ltext):
                if k not in results['financial_keywords']:
                    results['financial_keywords'].append(k)

        for k in STREET_SIGN_KEYWORDS:
            if re.search(r'\b'+re.escape(k)+r'\b', ltext):
                if k not in results['street_keywords']:
                    results['street_keywords'].append(k)

        for match in LONG_NUMBER_REGEX.findall(combined):
            dig= ''.join(re.findall(r'\d', match))
            if len(dig)>=8:
                clean= match.strip()
                if clean and clean not in results['long_number_like']:
                    results['long_number_like'].append(clean)

    pthit= 0
    pshit= 0
    pclass: List[str]= []
    for roi_out in rout:
        classi= 'unknown'
        for text in roi_out.get('texts', []):
            cls= plate_sign(text)
            if cls=='plate':
                pthit+=1
                classi= 'plate'
                break
            elif cls=='sign':
                classi= 'sign'
        pclass.append(classi)
        if classi=='sign':
            pshit+= 1

    results['pthit']= pthit
    results['pshit']= pshit
    results['pclass']= pclass
    return results

def classify_risk(score: int)-> str:
    if score>=75:
        return 'low'
    if score>=45:
        return 'moderate'
    return 'high'


def thresh_exceed(pen: int)-> bool:
    return pen>=40


def recc_build(mflags: Dict[str, bool],
                          finfo: List[Dict[str, object]],
                          ohit: Dict[str, Any],
                          documents: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    recs: List[str]= []
    if mflags.get('gps_present'):
        recs.append('Remove GPS location metadata before sharing (use sanitized download).')
    if mflags.get('datetime_present'):
        recs.append('Strip exact capture timestamps from EXIF if they reveal sensitive timelines.')
    if mflags.get('serial_present'):
        recs.append('Remove maker ser or unique IDs to avoid device fingerprinting.')
    if mflags.get('pexif') and not any(
            mflags[key] for key in ('gps_present', 'datetime_present', 'serial_present')):
        recs.append('Remove the remaining EXIF fields to minimise metadata leakage.')

    if finfo:
        if any(not f['blurred'] for f in finfo):
            recs.append('Blur, crop, or mask visible faces before sharing publicly.')
        else:
            recs.append('Double-check blurred faces for completeness before sharing.')

    if ohit.get('license_like') or ohit.get('pthit'):
        recs.append('Mask vehicle number plates that appear readable in the image.')
    elif ohit.get('pcand'):
        recs.append('Verify signage or reflective surfaces that resemble a number plate and mask if needed.')
    if ohit.get('email_like'):
        recs.append('Remove or redact email addresses before sharing.')
    if ohit.get('phone_like'):
        recs.append('Obscure phone numbers in the document or image.')
    if ohit.get('address_like'):
        recs.append('Hide house numbers or address signage visible in the scene.')
    if ohit.get('tax_id_like'):
        recs.append('Remove tax or government identification numbers detected in the text.')
    if ohit.get('financial_keywords'):
        recs.append('Avoid sharing invoices or bills publicly; redact sensitive billing details.')
    if ohit.get('long_number_like'):
        recs.append('Mask account or reference numbers found in the image.')
    if ohit.get('street_keywords'):
        recs.append('Consider masking street signage if it reveals your precise location.')
    if ohit.get('signage_flagged'):
        recs.append('Detected street signage; verify that location details are not exposed.')

    if documents:
        for d in documents:
            dtype = d.get('type')
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

    return list(dict.fromkeys(recs))


def image_analy(ibyte, filename_hint='image'):
    pil= Image.open(io.BytesIO(ibyte))
    pil= ImageOps.exif_transpose(pil).convert('RGB')
    width, height= pil.size
    np_img= cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    lpreg= detect_plate(np_img)
    docreg= doc_detect(np_img)
    exif= read_exif(pil)
    pexif= bool(exif and any(exif.values()))
    gps= gps_is(exif)
    dt= datetime_is(exif)
    ser= serial_is(exif)

    faces= cv2_facedetect(np_img)
    finfo= []
    for f in faces:
        x, y, w, h= f['x'], f['y'], f['w'], f['h']
        roi= np_img[y:y+h, x:x+w]
        rgray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bscore= lapla_var(rgray)
        bval= float(bscore)
        blur= bool(bval < 100.0)
        finfo.append({'rect': f, 'blur_var': bval, 'blurred': blur})

    ocr= sens_find(np_img, plate_regions=lpreg)
    reasons: List[str]= []

    pencomp= {
        'metadata': {'pen': 0, 'issues': []},
        'visual': {'pen': 0, 'issues': []},
        'textual': {'pen': 0, 'issues': []},
        'documents': {'pen': 0, 'issues': []},
        'other': {'pen': 0, 'issues': []},
    }

    if gps:
        pencomp['metadata']['pen']+=40
        pencomp['metadata']['issues'].append('GPS coordinates present in EXIF.')
        reasons.append('GPS coordinates found in EXIF (high risk).')
    if dt:
        pencomp['metadata']['pen']+=10
        pencomp['metadata']['issues'].append('Capture timestamp present in EXIF.')
        reasons.append('Exact Date/Time found in EXIF (time reveals).')
    if ser:
        pencomp['metadata']['pen']+=10
        pencomp['metadata']['issues'].append('Serial number or maker note present in EXIF.')
        reasons.append('Serial/unique maker info in EXIF.')
    if pexif and not (gps or dt or ser):
        pencomp['metadata']['pen']+=5
        pencomp['metadata']['issues'].append('Other EXIF metadata present.')
        reasons.append('Other EXIF metadata present.')

    if finfo:
        for i, f_info in enumerate(finfo):
            if f_info['blurred']:
                pencomp['visual']['pen']+=3
                pencomp['visual']['issues'].append(f'Face {i+1} detected but partially blurred.')
                reasons.append(f'Face {i+1} detected but blurred (less risk).')
            else:
                pen=12 if i==0 else 8
                pencomp['visual']['pen']+=pen
                pencomp['visual']['issues'].append(f'Face {i+1} is clearly visible.')
                reasons.append(f'Face {i+1} detected and clearly visible (privacy risk).')

    pcount= len(lpreg)
    pcand= max(pcount, ocr.get('pcand', 0))
    pthit= ocr.get('pthit', 0)
    skhit= ocr.get('street_keywords', [])
    pshit= ocr.get('pshit', 0)
    sign= bool(pcand and pthit == 0 and (pshit or skhit))

    if ocr['license_like']:
        pencomp['textual']['pen']+=60
        pencomp['textual']['issues'].append('License plate-like text detected.')
        reasons.append('License plate-like text detected by OCR.')
    elif pthit:
        pencomp['textual']['pen']+=50
        pencomp['textual']['issues'].append('Localized text resembles a license plate.')
        reasons.append('Localized region contains plate-like text (review vehicle details).')
    elif sign:
        pencomp['documents']['pen']+=5
        pencomp['documents']['issues'].append('Street signage detected; consider masking location info.')
        reasons.append('Street signage detected; location may be exposed.')
        pcand= 0
    elif pcand:
        pencomp['textual']['pen']+=15
        pencomp['textual']['issues'].append('Plate-shaped region detected with unclear text.')
        reasons.append('Possible license plate region detected; verify visibility.')

    ocr['pcand']= pcand
    ocr['signage_flagged']= bool(sign)
    if ocr['address_like']:
        pencomp['textual']['pen']+=40
        pencomp['textual']['issues'].append('Address keywords detected in text.')
        reasons.append('Address / house number keywords found by OCR.')
    if ocr['phone_like']:
        pencomp['textual']['pen']+=35
        pencomp['textual']['issues'].append('Phone number-like text detected.')
        reasons.append('Phone-number-like text found by OCR.')
    if ocr['email_like']:
        pencomp['textual']['pen']+=45
        pencomp['textual']['issues'].append('Email address detected in text.')
        reasons.append('Email address detected in OCR text (contact detail).')
    if ocr['tax_id_like']:
        pencomp['textual']['pen']+=55
        pencomp['textual']['issues'].append('Tax or government ID detected in text.')
        reasons.append('Tax/Government ID detected in OCR text (highly sensitive).')
    if ocr['financial_keywords']:
        pencomp['textual']['pen']+=30
        pencomp['textual']['issues'].append('Financial keywords detected (invoice/billing).')
        reasons.append('Financial or billing-related keywords detected (bill/invoice).')
    if ocr['long_number_like']:
        pencomp['textual']['pen']+=25
        pencomp['textual']['issues'].append('Long numeric sequences found (possible account numbers).')
        reasons.append('Long numeric sequence detected (possible account or ID number).')

    if pencomp['textual']['pen']>=60:
        reasons.append('Sensitive text details detected; score capped for safety.')

    docpen= {
        'invoice': 60,
        'bill': 55,
        'credit_card': 70,
        'id_card': 65,
        'document': 35,
        'street_sign': 0,
    }
    dreason= {
        'invoice': 'Document region resembles an invoice or bill.',
        'bill': 'Billing document detected (contains purchase details).',
        'credit_card': 'Payment card detected (card numbers may be visible).',
        'id_card': 'Identity card detected (personal identifiers visible).',
        'document': 'Document detected; review for sensitive information.',
        'street_sign': 'Street signage detected (review visibility).',
    }
    high_risk_doc= False
    for d in docreg:
        dtype= d.get('type', 'document')
        pen= docpen.get(dtype, 30)
        if pen<=0:
            continue
        pencomp['documents']['pen']+=pen
        label= dreason.get(dtype, 'Document detected.')
        snip= d.get('meta', {}).get('snip', [])
        stext= ', '.join(snip[:2]) if snip else ''
        issue= label if not stext else f"{label} Sample text: {stext}"
        pencomp['documents']['issues'].append(issue)
        reasons.append(label)
        if dtype in {'invoice', 'bill', 'credit_card', 'id_card'}:
            high_risk_doc = True

    mpix= (width*height)/1_000_000.0
    if mpix>4:
        pencomp['other']['pen']+=5
        pencomp['other']['issues'].append('High resolution image exposes fine details.')
        reasons.append(f'High resolution image ({mpix:.1f} MP) - more detail visible.')

    tpen= sum(comp['pen'] for comp in pencomp.values())
    score= max(0, min(100, int(round(100 - tpen))))

    scap= 100
    if ocr['license_like']:
        scap= min(scap, 40)
    elif pthit:
        scap= min(scap, 45)
    elif pcand:
        scap= min(scap, 70)
    if ocr['tax_id_like']:
        scap= min(scap, 35)
    if ocr['email_like']:
        scap= min(scap, 45)
    if ocr['address_like']:
        scap= min(scap, 50)
    if ocr['phone_like']:
        scap= min(scap, 55)
    if ocr['financial_keywords']:
        scap= min(scap, 60)
    if ocr['long_number_like']:
        scap= min(scap, 55)
    for d in docreg:
        dtype= d.get('type')
        if dtype=='credit_card':
            scap= min(scap, 30)
        elif dtype=='id_card':
            scap= min(scap, 35)
        elif dtype in {'invoice', 'bill'}:
            scap= min(scap, 45)
    score= min(score, scap)

    risk= classify_risk(score)
    cthit= bool(
        ocr['license_like']
        or ocr['address_like']
        or ocr['phone_like']
        or ocr['email_like']
        or ocr['tax_id_like']
        or ocr['financial_keywords']
        or ocr['long_number_like']
        or pthit
        or high_risk_doc
    )
    sharesafe= (
        risk=='low'
        and not cthit
        and not thresh_exceed(pencomp['metadata']['pen'])
    )

    mflags= {
        'pexif': pexif,
        'gps_present': gps,
        'datetime_present': dt,
        'serial_present': ser,
    }
    recomm= recc_build(mflags, finfo, ocr, docreg)

    breakd= {
        'score': score,
        'max_score': 100,
        'risk': risk,
        'sharesafe': bool(sharesafe),
        'width': width,
        'height': height,
        'mpix': mpix,
        'pexif': pexif,
        'gps_present': gps,
        'datetime_present': dt,
        'serial_present': ser,
        'faces': finfo,
        'ocr': ocr,
        'lpreg': lpreg,
        'documents': docreg,
        'pencomp': pencomp,
        'reasons': reasons,
        'recomm': recomm,
        'esummary': esummary(exif)
    }
    return breakd

def remove_exif(ibyte):
    pil= Image.open(io.BytesIO(ibyte))
    pil= ImageOps.exif_transpose(pil)
    out_io= io.BytesIO()
    pil.save(out_io, format='JPEG', quality=95)
    return out_io.getvalue()

def faceblur_byte(ibyte):
    pil= Image.open(io.BytesIO(ibyte))
    pil= ImageOps.exif_transpose(pil).convert('RGB')
    np_img= cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    faces= cv2_facedetect(np_img)
    for f in faces:
        x, y, w, h= f['x'], f['y'], f['w'], f['h']
        roi= np_img[y:y+h, x:x+w]
        k= max(15, (w//7)|1)
        rblur= cv2.GaussianBlur(roi, (x,x), 0)
        np_img[y:y+h, x:x+w]= rblur
    outpil= Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    buf= io.BytesIO()
    outpil.save(buf, format='JPEG', quality=90)
    return buf.getvalue()
