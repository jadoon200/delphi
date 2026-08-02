import hashlib
from pathlib import Path

import httpx
import pytest
import respx

from delphi.data.fetch import fetch_azure_functions, sha256_file


@respx.mock
def test_fetch_is_atomic_and_checksum_verified(tmp_path: Path) -> None:
    payload = b"small archive stand-in"
    expected = hashlib.sha256(payload).hexdigest()
    route = respx.get("https://example.test/azure.tar.xz").mock(
        return_value=httpx.Response(200, content=payload)
    )
    destination = tmp_path / "azure.tar.xz"

    assert (
        fetch_azure_functions(
            destination,
            url="https://example.test/azure.tar.xz",
            expected_sha256=expected,
        )
        == destination
    )
    assert route.called
    assert destination.read_bytes() == payload
    assert sha256_file(destination) == expected
    assert not destination.with_suffix(".xz.part").exists()


@respx.mock
def test_fetch_removes_partial_file_on_checksum_mismatch(tmp_path: Path) -> None:
    respx.get("https://example.test/azure.tar.xz").mock(
        return_value=httpx.Response(200, content=b"corrupt")
    )
    destination = tmp_path / "azure.tar.xz"

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_azure_functions(
            destination,
            url="https://example.test/azure.tar.xz",
            expected_sha256="0" * 64,
        )
    assert not destination.exists()
    assert not destination.with_suffix(".xz.part").exists()
