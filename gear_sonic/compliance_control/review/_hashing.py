"""Version-portable hashing helpers for already bounded binary streams."""

from __future__ import annotations

import hashlib
from typing import BinaryIO


def sha256_stream(stream: BinaryIO, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash from the current position without reopening the verified file."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
