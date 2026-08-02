"""Free, resumable-enough trace download helpers with checksum verification."""

import hashlib
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from delphi.data.azure_functions import ARCHIVE_SHA256, ARCHIVE_URL


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def fetch_azure_functions(
    destination: Path,
    *,
    url: str = ARCHIVE_URL,
    expected_sha256: str = ARCHIVE_SHA256,
    timeout: float = 60.0,
) -> Path:
    """Download the Azure trace atomically and reject a checksum mismatch."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
    actual = sha256_file(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Azure archive checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    partial.replace(destination)
    return destination
