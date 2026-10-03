"""Shared Google Drive zip fetch, used by both the dataset cache and the outputs cache.

`tiny_lora.data.ensure_gdrive_dataset` and `tiny_lora.train_sft.resolve_resume_checkpoint` both
reduce to "get a zip from Drive, extract it into a cache dir" -- this module is just that step,
plus the one bit of tidy-up ("Drive wrapped everything in a folder") that both need.

Three ways to get the zip (in the priority `tiny_lora.data.ensure_gdrive_dataset` applies them):

- `extract_local_zip` -- the zip is already sitting on the training machine's own disk (e.g.
  uploaded straight into the Colab VM, no Drive involved at all). No network, no auth, nothing
  Colab-specific; just an extract. Tried first since it needs the least to go right.
- `extract_from_drive_mount` -- reads the file straight off an *already*-mounted Drive,
  authenticated as its owner exactly like a browser download. Sidesteps `gdown`'s throttling
  (below) entirely since Google never serves the anonymous-abuse response to an authenticated
  request. Does NOT mount Drive itself -- `layer_grow_install.sh`'s dataset-prep step runs this
  from a `poetry run` subprocess, in its own venv, and `google.colab.drive.mount()` only works
  from the actual Colab kernel process (it talks to the notebook frontend for the OAuth flow);
  mount Drive yourself from a real notebook cell first. Once mounted it's an ordinary (FUSE)
  filesystem, visible to every process on the VM, so nothing here needs to import `google.colab`.
- `download_and_extract_zip` -- `gdown`'s anonymous public-link download. Works with just a file
  id, no Colab dependency, but Google's abuse heuristics throttle anonymous requests for large
  ("can't scan for viruses") files hard: we hit "Too many users have viewed or downloaded this
  file recently" on two different, freshly-shared files in a row, neither of which had actually
  been viewed enough times to earn that -- it is a response to the request pattern (scripted,
  cookie-less, large file), not real popularity. Last resort; kept for machines with neither a
  local copy nor a Drive mount.

All three leave the same mess a Drive folder-zip can leave (everything one level deeper than
expected, in a directory named after the original folder) for the caller to clean up with
`flatten_single_wrapper_dir`/`tiny_lora.data._flatten_dataset_dir`.
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


def extract_local_zip(cache_dir: Path, zip_path: str) -> None:
    """Extract a zip already sitting on this machine's own disk into `cache_dir`."""
    source = Path(zip_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found. data.gdrive.local_zip_path should be the zip's path on this "
            "machine (e.g. wherever you uploaded it in the Colab file browser)."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(cache_dir)


def extract_from_drive_mount(
    cache_dir: Path, mount_path: str, drive_root: str = "/content/drive/MyDrive"
) -> None:
    """Extract a zip from an *already*-mounted Google Drive into `cache_dir`.

    `mount_path` is the file's path relative to "My Drive", e.g. "ds-assistant-grpo-v2.zip" for
    a file uploaded to Drive's root, or "some/folder/ds-assistant-grpo-v2.zip" for one filed in a
    subfolder.

    Does not mount Drive -- that has to happen from an actual Colab notebook cell, before running
    whatever calls this (e.g. `bash layer_grow_install.sh`):

        from google.colab import drive
        drive.mount('/content/drive')

    A `poetry run` subprocess (which is how the install script gets here) is a separate process in
    its own venv; `drive.mount()` needs the real Colab kernel process to drive the OAuth flow, and
    `google.colab` usually isn't even importable outside it. `/content/drive` itself is an ordinary
    (FUSE) filesystem once mounted, though, so reading from it needs nothing Colab-specific.
    """
    root = Path(drive_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found -- Drive isn't mounted. From an actual Colab notebook cell (not "
            "this script), run:\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n\n"
            "then re-run this."
        )

    source = root / mount_path
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
