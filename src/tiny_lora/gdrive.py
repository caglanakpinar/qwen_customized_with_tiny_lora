"""Shared Google Drive zip fetch, used by both the dataset cache and the outputs cache.

`tiny_lora.data.ensure_gdrive_dataset` and `tiny_lora.train_sft.resolve_resume_checkpoint` both
reduce to "get a zip from Drive, extract it into a cache dir" -- this module is just that step,
plus the one bit of tidy-up ("Drive wrapped everything in a folder") that both need.

Two ways to get the zip:

- `download_and_extract_zip` -- `gdown`'s anonymous public-link download. Works with just a file
  id, no Colab dependency, but Google's abuse heuristics throttle anonymous requests for large
  ("can't scan for viruses") files hard: we hit "Too many users have viewed or downloaded this
  file recently" on two different, freshly-shared files in a row, neither of which had actually
  been viewed enough times to earn that -- it is a response to the request pattern (scripted,
  cookie-less, large file), not real popularity.
- `extract_from_drive_mount` -- mounts the caller's own Drive (Colab-only) and reads the file
  directly, authenticated as its owner exactly like a browser download. Sidesteps the throttling
  above entirely since Google never serves the anonymous-abuse response to an authenticated
  request. Preferred when available; see `data.gdrive.mount_path` in a config's `data:` block.
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


def extract_from_drive_mount(cache_dir: Path, mount_path: str) -> None:
    """Extract a zip straight from the caller's own mounted Google Drive into `cache_dir`.

    `mount_path` is the file's path relative to "My Drive", e.g. "ds-assistant-grpo-v2.zip" for
    a file uploaded to Drive's root, or "some/folder/ds-assistant-grpo-v2.zip" for one filed in a
    subfolder. Mounting is idempotent -- Colab's `drive.mount` no-ops (after re-prompting auth if
    the session lost it) when `/content/drive` is already mounted, so calling this more than once
    in a run is harmless.
    """
    try:
        from google.colab import drive
    except ImportError as exc:
        raise ImportError(
            "data.gdrive.mount_path only works inside a Google Colab runtime "
            "(needs google.colab.drive). Use data.gdrive.zip_file_id instead outside Colab."
        ) from exc

    drive.mount("/content/drive")
    source = Path("/content/drive/MyDrive") / mount_path
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found on the mounted Drive. data.gdrive.mount_path should be the "
            "file's path relative to 'My Drive' -- check it was uploaded there (not just "
            "shared from someone else's Drive) and that the path/filename match exactly."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(cache_dir)


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
