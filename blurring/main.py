from pathlib import Path
from typing import Iterable, Optional
from modules.blur import blur_images

def blur_run(
    ipaths: Iterable[Path],
    output_dir: Optional[Path]= None,
    *,
    auto: bool= True,
    blur_text: bool= True,
    progress_cb=None,
):
    ipaths= list(ipaths)
    if not ipaths:
        raise ValueError("No image paths given to blur.")

    if output_dir is None:
        output_dir= Path(__file__).resolve().parent/"blurred_output"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return blur_images(
        ipaths,
        output_dir,
        auto=auto,
        blur_text=blur_text,
        progress_cb=progress_cb,
    )

if __name__ == "__main__":
    sample = [Path(r"C:\Users\HP\Downloads\uss\text.jpg")]
    blur_run(sample)

