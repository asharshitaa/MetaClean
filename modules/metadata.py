from pathlib import Path
from typing import Iterable, List

from meta.modules.metadata_tools import clean_metadata_batch


def clean_metadata(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    progress_cb=None,
) -> List[str]:
    """
    Runs metadata removal on a group of images and returns paths to cleaned files.
    """
    results = clean_metadata_batch(
        [str(Path(p)) for p in image_paths],
        Path(output_dir),
        progress_cb=progress_cb,
    )
    cleaned = [item["output"] for item in results if item.get("output")]
    return [str(Path(path)) for path in cleaned]

