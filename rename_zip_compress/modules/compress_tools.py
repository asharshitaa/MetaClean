import os
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from PIL import Image

ProgressCallback = Callable[[str, dict], None] | None


def compress_images(
    paths: Iterable[str],
    outfolder: str | Path,
    *,
    quality: int = 80,
    maxsize: tuple[int, int] = (1080, 1080),
    progress_cb: ProgressCallback = None,
) -> List[str]:
    outfolder = Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    compressed: List[str] = []

    for path in paths:
        source = Path(path)
        if not source.exists():
            if progress_cb:
                progress_cb("error", {"stage": "compress", "source": str(source), "error": "File not found"})
            continue
        try:
            if progress_cb:
                progress_cb("compress_start", {"source": str(source)})
            im = Image.open(source)
            im.thumbnail(maxsize)
            target = outfolder / source.name
            im.save(target, optimize=True, quality=quality)
            compressed.append(str(target))
            if progress_cb:
                progress_cb("compress_done", {"source": str(source), "output": str(target)})
        except Exception as exc:
            if progress_cb:
                progress_cb("error", {"stage": "compress", "source": str(source), "error": str(exc)})
    return compressed


def rename_files(
    paths: Sequence[str],
    newnames: Sequence[str],
    *,
    progress_cb: ProgressCallback = None,
) -> List[str]:
    renamed: List[str] = []
    for path, name in zip(paths, newnames):
        source = Path(path)
        if not source.exists():
            if progress_cb:
                progress_cb("error", {"stage": "rename", "source": str(source), "error": "File not found"})
            continue
        try:
            folder = source.parent
            base, ext = os.path.splitext(name)
            if not ext:
                ext = source.suffix
                name = base + ext
            target = folder / name
            count = 1
            while target.exists():
                target = folder / f"{base} ({count}){ext}"
                count += 1
            os.rename(source, target)
            renamed.append(str(target))
            if progress_cb:
                progress_cb("rename_done", {"source": str(source), "output": str(target)})
        except Exception as exc:
            if progress_cb:
                progress_cb("error", {"stage": "rename", "source": str(source), "error": str(exc)})
            renamed.append(str(source))
    return renamed


def zip_folder(
    path: str | Path,
    name: str = "new_zip.zip",
    *,
    progress_cb: ProgressCallback = None,
) -> str | None:
    path = Path(path)
    zpath = path / name
    base, ext = os.path.splitext(name)
    count = 1
    while zpath.exists():
        zpath = path / f"{base} ({count}){ext}"
        count += 1

    try:
        with zipfile.ZipFile(zpath, "w") as zipf:
            for root, _, files in os.walk(path):
                for file_name in files:
                    if file_name.lower().endswith(".zip"):
                        continue
                    file_path = Path(root) / file_name
                    if file_path == zpath:
                        continue
                    zipf.write(file_path, file_path.relative_to(path))

        if progress_cb:
            progress_cb("zip_done", {"output": str(zpath)})
        return str(zpath)
    except Exception as exc:
        if progress_cb:
            progress_cb("error", {"stage": "zip", "path": str(path), "error": str(exc)})
        return None

