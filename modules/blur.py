from pathlib import Path
from typing import Iterable, List, Optional

from blurring.modules.blur_tools import blur_faces_and_text


def blur_images(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    auto: bool = True,
    blur_text: bool = True,
    progress_cb=None,
) -> List[str]:
    """
    Wrapper around the blurring pipeline that normalises parameters for the GUI.
    """
    path_list = [Path(p) for p in image_paths]
    if not path_list:
        return []
    return blur_faces_and_text(
        path_list,
        Path(output_dir),
        auto=auto,
        blur_text=blur_text,
        progress_cb=progress_cb,
    )

