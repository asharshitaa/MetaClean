from pathlib import Path
from typing import Iterable, List, Optional

from blurring.modules.blur_tools import batch_blur


def blur_images(
    ipaths: Iterable[str | Path],
    outdir: str | Path,
    *,
    auto: bool= True,
    tblur: bool= True,
    procb=None,
) -> List[str]:
    plist= [Path(p) for p in ipaths]
    if not plist:
        return []
    return batch_blur(
        plist,
        Path(outdir),
        auto=auto,
        tblur=tblur,
        procb=procb,
    )

