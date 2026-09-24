"""Upload path validation.

A tool argument is chosen by a model. A path argument therefore has to be
confined to a directory the developer configured, or the model can read any
file the process can.
"""

from __future__ import annotations

from pathlib import Path


class GoodMemUploadError(ValueError):
    """A requested upload path is not permitted."""


def resolve_upload_path(candidate: str, upload_dir: str | Path | None) -> Path:
    """Resolve ``candidate`` inside ``upload_dir``, or refuse.

    ``upload_dir`` must be configured explicitly; there is no default. Both
    sides are fully resolved before comparison, so symlinks that leave the
    directory are rejected along with ``..`` traversal.
    """
    if upload_dir is None:
        raise GoodMemUploadError(
            "File uploads are disabled. Set upload_dir on the tool to the "
            "directory whose files agents may upload."
        )

    try:
        base = Path(upload_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise GoodMemUploadError(f"upload_dir is not usable: {exc}") from exc
    if not base.is_dir():
        raise GoodMemUploadError(f"upload_dir is not a directory: {base}")

    requested = Path(candidate).expanduser()
    joined = requested if requested.is_absolute() else base / requested

    try:
        # strict=True: refuse before the file is opened, and resolve every
        # symlink so a link inside the directory cannot point outside it.
        resolved = joined.resolve(strict=True)
    except OSError as exc:
        raise GoodMemUploadError(f"Cannot read '{candidate}': {exc}") from exc

    if not resolved.is_relative_to(base):
        raise GoodMemUploadError(
            f"'{candidate}' resolves outside the configured upload_dir ({base})."
        )
    if not resolved.is_file():
        raise GoodMemUploadError(f"'{candidate}' is not a regular file.")
    return resolved
