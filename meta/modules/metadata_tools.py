from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from PIL import ExifTags, Image

ProgressCallback = Callable[[str, dict], None] | None


def read_exif(image_path: str) -> Dict[str, str]:
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


def remove_exif(input_path: str, output_path: str) -> bool:
    """
    Saves a copy of the image without EXIF metadata.
    Works for JPEG; for other formats adapt as needed.
    """
    img = Image.open(input_path)
    data = list(img.getdata())
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(data)
    new_img.save(output_path, format="JPEG", quality=95)
    return True


def compute_privacy_score(exif_dict: Dict[str, str]) -> Tuple[int, Dict[str, object]]:
    """
    Computes a privacy score (0-100). 100 = private (no sensitive metadata).
    Returns: (score:int, details:dict)
    """
    if not exif_dict:
        return 100, {}
    sensitive_weights = {
        "GPSInfo": 35,
        "GPSLatitude": 20,
        "GPSLongitude": 20,
        "DateTime": 15,
        "DateTimeOriginal": 20,
        "Make": 8,
        "Model": 8,
        "LensModel": 6,
        "Artist": 6,
        "Software": 4,
        "UserComment": 10,
        "Copyright": 6,
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


def clean_metadata_batch(
    image_paths: Iterable[str],
    output_dir: str | Path,
    *,
    progress_cb: ProgressCallback = None,
) -> List[dict]:
    """
    Removes metadata from multiple images.
    Returns a list of dictionaries with results for each image.
    """
    results: List[dict] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        source = Path(image_path)
        record = {"source": str(source), "output": None, "before_score": None, "after_score": None, "error": None}

        if not source.exists():
            record["error"] = "File not found"
            if progress_cb:
                progress_cb("error", record)
            results.append(record)
            continue

        try:
            if progress_cb:
                progress_cb("start", {"source": str(source)})

            exif_before = read_exif(str(source))
            before_score, before_details = compute_privacy_score(exif_before)

            cleaned_path = output_dir / f"cleaned_{source.name}"
            remove_exif(str(source), str(cleaned_path))

            exif_after = read_exif(str(cleaned_path))
            after_score, after_details = compute_privacy_score(exif_after)

            record.update(
                {
                    "output": str(cleaned_path),
                    "before_score": before_score,
                    "before_details": before_details,
                    "after_score": after_score,
                    "after_details": after_details,
                }
            )

            if progress_cb:
                progress_cb("done", record)
        except Exception as exc:
            record["error"] = str(exc)
            if progress_cb:
                progress_cb("error", record)
        results.append(record)

    return results
