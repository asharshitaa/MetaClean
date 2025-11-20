from pathlib import Path
from typing import Iterable, List

from rename_zip_compress.modules.compress_tools import (
    compress_images,
    rename_files,
    zip_folder,
)


def comp_pack(
    ipaths: Iterable[str | Path],
    outdir: str | Path,
    *,
    rpre: str= "secure",
    procb=None,
):
    outdir= Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    comp= compress_images(
        [str(Path(p)) for p in ipaths],
        outdir,
        
    )
    renamed= []
    if comp:
        nnames= [
            f"{rpre}_{idx}{Path(path).suffix or '.jpg'}"
            for idx, path in enumerate(comp, start=1)
        ]
        renamed= rename_files(comp, nnames)

    zpath= zip_folder(outdir) if renamed else None
    return {
        "comp": comp,
        "renamed": renamed,
        "zpath": zpath,
    }

