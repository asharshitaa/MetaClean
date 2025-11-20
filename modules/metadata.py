from pathlib import Path
from typing import Iterable, List

from meta.modules.metadata_tools import clean_metadata_batch

def clean_metadata(
    ipaths: Iterable[str | Path],
    outdir: str | Path,
    *,
    procb=None,
) -> List[str]:
    
    results= clean_metadata_batch(
        [str(Path(p)) for p in ipaths],
        Path(outdir),
        procb=procb,
    )
    clean= [item["output"] for item in results if item.get("output")]
    return [str(Path(path)) for path in clean]

