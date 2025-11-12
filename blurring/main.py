# from modules.blur_tools import blur_sensitive_content

# if __name__ == "__main__":
#     input_path = r"C:\Users\HP\Downloads\uss\text.jpg"  # or your test image path
#     blur_sensitive_content(input_path)

from pathlib import Path
from typing import Iterable, Optional

from modules.blur import blur_images


def run_blurring(
    image_paths: Iterable[Path],
    output_dir: Optional[Path] = None,
    *,
    auto: bool = True,
    blur_text: bool = True,
    progress_cb=None,
):
    """
    Convenience wrapper used by the GUI to blur sensitive content.
    Returns the list of processed image paths.
    """
    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("No image paths supplied for blurring.")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "blurred_output"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return blur_images(
        image_paths,
        output_dir,
        auto=auto,
        blur_text=blur_text,
        progress_cb=progress_cb,
    )


if __name__ == "__main__":
    sample = [Path(r"C:\Users\HP\Downloads\uss\text.jpg")]
    run_blurring(sample)

