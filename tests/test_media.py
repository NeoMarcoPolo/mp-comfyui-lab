from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pytest

from mp_comfyui_lab import media as media_module
from mp_comfyui_lab.media import MediaVerificationPolicy, inspect_media_file
from tests.helpers import valid_media_bytes


POLICY = MediaVerificationPolicy(max_image_pixels=1_000_000)


def _zip_bytes(
    entries: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def _inspect_zip(path: Path, content: bytes, *, max_bytes: int) -> object:
    path.write_bytes(content)
    return inspect_media_file(
        path,
        "application/zip",
        policy=POLICY,
        max_bytes=max_bytes,
    )


def test_small_valid_zip_passes_bounded_verification(tmp_path: Path) -> None:
    content = _zip_bytes([("result.txt", b"ordinary output")])

    inspection = _inspect_zip(
        tmp_path / "valid.zip",
        content,
        max_bytes=1_024,
    )

    assert inspection.valid
    assert inspection.detected_mime_type == "application/zip"


def test_zip_preflight_rejects_excessive_member_count(tmp_path: Path) -> None:
    content = _zip_bytes(
        [
            (f"entry-{index:04d}.txt", b"")
            for index in range(media_module.MAX_ZIP_MEMBERS + 1)
        ],
        compression=zipfile.ZIP_STORED,
    )

    inspection = _inspect_zip(
        tmp_path / "too-many.zip",
        content,
        max_bytes=len(content) + 1,
    )

    assert not inspection.valid
    assert inspection.error_code == "invalid_archive"


def test_zip_preflight_rejects_oversized_member(tmp_path: Path) -> None:
    content = _zip_bytes([("large.txt", b"A" * 2_048)])
    assert len(content) < 1_024

    inspection = _inspect_zip(
        tmp_path / "large-member.zip",
        content,
        max_bytes=1_024,
    )

    assert inspection.error_code == "invalid_archive"


def test_zip_preflight_rejects_oversized_aggregate(tmp_path: Path) -> None:
    content = _zip_bytes(
        [("first.txt", b"A" * 600), ("second.txt", b"B" * 600)]
    )
    assert len(content) < 1_024

    inspection = _inspect_zip(
        tmp_path / "large-total.zip",
        content,
        max_bytes=1_024,
    )

    assert inspection.error_code == "invalid_archive"


def test_zip_preflight_rejects_extreme_compression_ratio(tmp_path: Path) -> None:
    content = _zip_bytes([("compressed.txt", b"A" * 20_000)])

    inspection = _inspect_zip(
        tmp_path / "ratio.zip",
        content,
        max_bytes=30_000,
    )

    assert inspection.error_code == "invalid_archive"


@pytest.mark.parametrize("kind", ["encrypted", "unsupported"])
def test_zip_preflight_rejects_encrypted_or_unsupported_entries(
    tmp_path: Path,
    kind: str,
) -> None:
    content = bytearray(
        _zip_bytes([("result.txt", b"ordinary output")], compression=zipfile.ZIP_STORED)
    )
    local = content.index(b"PK\x03\x04")
    central = content.index(b"PK\x01\x02")
    if kind == "encrypted":
        content[local + 6 : local + 8] = (1).to_bytes(2, "little")
        content[central + 8 : central + 10] = (1).to_bytes(2, "little")
    else:
        content[local + 8 : local + 10] = (99).to_bytes(2, "little")
        content[central + 10 : central + 12] = (99).to_bytes(2, "little")

    inspection = _inspect_zip(
        tmp_path / f"{kind}.zip",
        bytes(content),
        max_bytes=1_024,
    )

    assert inspection.error_code == "invalid_archive"


def test_malformed_zip_is_invalid_media(tmp_path: Path) -> None:
    inspection = _inspect_zip(
        tmp_path / "malformed.zip",
        b"PK\x03\x04malformed-central-directory",
        max_bytes=1_024,
    )

    assert inspection.error_code == "invalid_archive"


def test_zip_preflight_runs_before_any_decompression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _zip_bytes([("large.txt", b"A" * 2_048)])
    calls: list[str] = []

    def forbidden_testzip(_archive):
        calls.append("testzip")
        raise AssertionError("testzip decompressed before preflight")

    monkeypatch.setattr(zipfile.ZipFile, "testzip", forbidden_testzip)

    inspection = _inspect_zip(
        tmp_path / "preflight.zip",
        content,
        max_bytes=1_024,
    )

    assert inspection.error_code == "invalid_archive"
    assert calls == []


@pytest.mark.parametrize(
    "media_type",
    [
        "image/png",
        "video/mp4",
        "audio/wav",
    ],
)
def test_bounded_zip_checks_preserve_ordinary_media(
    tmp_path: Path,
    media_type: str,
) -> None:
    path = tmp_path / "ordinary.bin"
    content = valid_media_bytes(media_type)
    path.write_bytes(content)

    inspection = inspect_media_file(
        path,
        media_type,
        policy=POLICY,
        max_bytes=len(content),
    )

    assert inspection.valid
