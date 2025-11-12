from pathlib import Path
from typing import Iterable, List

from rename_zip_compress.modules.compress_tools import (
    compress_images,
    rename_files,
    zip_folder,
)


def compress_and_package(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    rename_prefix: str = "secure",
    progress_cb=None,
):
    """
    Compresses images, renames them safely, and produces a ZIP archive.

    Returns a dict containing:
      - compressed: list of compressed (pre-rename) image paths
      - renamed: list of renamed image paths
      - zip_path: final archive path (or None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compressed = compress_images(
        [str(Path(p)) for p in image_paths],
        output_dir,
        progress_cb=progress_cb,
    )
    renamed = []
    if compressed:
        new_names = [
            f"{rename_prefix}_{idx}{Path(path).suffix or '.jpg'}"
            for idx, path in enumerate(compressed, start=1)
        ]
        renamed = rename_files(compressed, new_names, progress_cb=progress_cb)

    zip_path = zip_folder(output_dir, progress_cb=progress_cb) if renamed else None
    return {
        "compressed": compressed,
        "renamed": renamed,
        "zip_path": zip_path,
    }

