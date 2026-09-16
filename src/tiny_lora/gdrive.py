"""Shared Google Drive zip download, used by both the dataset cache and the outputs cache.

`tiny_lora.data.ensure_gdrive_dataset` and `tiny_lora.train_sft.resolve_resume_checkpoint` both
reduce to "download a zip from Drive, extract it into a cache dir" -- this module is just that
step, plus the one bit of tidy-up ("Drive wrapped everything in a folder") that both need.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-\d+$")


def download_and_extract_zip(cache_dir: Path, zip_file_id: str, zip_name: str) -> None:
    """Download `zip_file_id` from Google Drive into `cache_dir` and extract it there."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Reading from Google Drive requires the 'gdown' package. "
            "Install it with `poetry install -E gdrive`."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / zip_name
    gdown.download(id=zip_file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    zip_path.unlink()


def flatten_single_wrapper_dir(cache_dir: Path) -> None:
    """Pull a zip's contents up one level if Drive's folder-zip wrapped them in a named directory.

    Zipping a folder through Drive's own "Download" wraps its contents in a top-level directory
    named after the folder, landing everything one level deeper than expected. Repeats so a chain
    of nested wrapper folders is fully unwound, not just the outermost one.

    Stops short of unwrapping a directory that is itself named `checkpoint-N`: a zip of exactly
    one checkpoint is a legitimate single-entry archive, not a Drive-wrapped folder, and moving
    its contents up would strip the directory the caller is about to look for by that same name.
    """
    while True:
        entries = list(cache_dir.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            return
        wrapper = entries[0]
        if _CHECKPOINT_DIR_RE.match(wrapper.name):
            return
        for child in wrapper.iterdir():
            child.rename(cache_dir / child.name)
        wrapper.rmdir()
