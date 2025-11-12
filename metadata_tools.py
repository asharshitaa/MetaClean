from PIL import Image, ExifTags

def read_exif(image_path):
    """
    Reads EXIF metadata from an image and returns a dict mapping human-readable tag names to values.
    Returns {} when no EXIF present.
    """
    img = Image.open(image_path)
    try:
        raw = img._getexif()
    except Exception:
        raw = None
    if not raw:
        return {}
    exif = {}
    for tag_id, value in raw.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        exif[tag] = value
    return exif

def remove_exif(input_path, output_path):
    """
    Saves a copy of the image without EXIF metadata.
    Works for JPEG; for other formats adapt as needed.
    """
    img = Image.open(input_path)
    # copy pixel data to a new image and save without exif
    data = list(img.getdata())
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(data)
    new_img.save(output_path, format='JPEG', quality=95)
    return True

def compute_privacy_score(exif_dict):
    """
    Computes a privacy score (0-100). 100 = private (no sensitive metadata).
    Returns: (score:int, details:dict)
    """
    if not exif_dict:
        return 100, {}
    # Weighted sensitive fields (higher weight = more privacy risk)
    sensitive_weights = {
        "GPSInfo": 35,
        "GPSLatitude": 20, "GPSLongitude": 20,
        "DateTime": 15,
        "DateTimeOriginal": 20,
        "Make": 8,
        "Model": 8,
        "LensModel": 6,
        "Artist": 6,
        "Software": 4,
        "UserComment": 10,
        "Copyright": 6
    }
    found = {}
    total_risk = 0
    for tag, weight in sensitive_weights.items():
        if tag in exif_dict and exif_dict[tag]:
            found[tag] = exif_dict[tag]
            total_risk += weight
    if total_risk > 100:
        total_risk = 100
    score = int(round(100 - total_risk))
    if score < 0:
        score = 0
    details = {"found_fields": list(found.keys()), "raw_found": found, "risk_score": total_risk}
    return score, details
