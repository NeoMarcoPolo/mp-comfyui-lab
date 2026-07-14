from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Literal
import warnings
import zipfile

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from mp_comfyui_lab.security import SafetyError


MediaFamily = Literal["image", "video", "audio", "unknown"]
AlphaBackground = Literal["white", "checkerboard"]
MediaErrorCode = Literal[
    "unreadable_file",
    "empty_file",
    "byte_limit_exceeded",
    "unrecognized_signature",
    "declared_mime_mismatch",
    "invalid_image",
    "image_pixel_limit_exceeded",
    "invalid_archive",
]

PHASH_ALPHA_MATTE = (255, 255, 255)
TRANSPARENCY_CHECKER_LIGHT = (242, 244, 246)
TRANSPARENCY_CHECKER_DARK = (202, 208, 212)
TRANSPARENCY_CHECKER_SIZE = 8
MEDIA_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "application/zip": "zip",
}


@dataclass(frozen=True)
class MediaVerificationPolicy:
    max_image_pixels: int

    def __post_init__(self) -> None:
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")


# Pillow 10-12 use 89,478,485 pixels as the default warning threshold. The
# engine historically accepted images through that boundary.
ENGINE_MEDIA_POLICY = MediaVerificationPolicy(max_image_pixels=89_478_485)
COMPARISON_MEDIA_POLICY = MediaVerificationPolicy(max_image_pixels=40_000_000)


@dataclass(frozen=True)
class MediaInspection:
    declared_mime_type: str
    declared_family: MediaFamily
    detected_mime_type: str | None
    detected_family: MediaFamily | None
    image: Image.Image | None = None
    source_mode: str | None = None
    error_code: MediaErrorCode | None = None
    error_message: str | None = None

    @property
    def valid(self) -> bool:
        return self.error_code is None


class MediaVerificationError(SafetyError):
    def __init__(self, inspection: MediaInspection) -> None:
        self.inspection = inspection
        super().__init__(inspection.error_message or "media verification failed")


class _ImagePixelLimitError(ValueError):
    pass


def media_family(mime_type: str | None) -> MediaFamily:
    if not isinstance(mime_type, str):
        return "unknown"
    for family in ("image", "video", "audio"):
        if mime_type.startswith(f"{family}/"):
            return family
    return "unknown"


def inspect_media_file(
    path: Path,
    declared_mime_type: str,
    *,
    policy: MediaVerificationPolicy,
    max_bytes: int | None = None,
) -> MediaInspection:
    return _inspect_media_file(
        path,
        declared_mime_type,
        policy=policy,
        max_bytes=max_bytes,
        retain_image=True,
    )


def _inspect_media_file(
    path: Path,
    declared_mime_type: str,
    *,
    policy: MediaVerificationPolicy,
    max_bytes: int | None,
    retain_image: bool,
) -> MediaInspection:
    declared_family = media_family(declared_mime_type)
    try:
        file_stat = path.lstat()
    except OSError:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "unreadable_file",
            "media artifact is not readable",
        )
    if not stat.S_ISREG(file_stat.st_mode):
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "unreadable_file",
            "media artifact is not a regular file",
        )
    if file_stat.st_size == 0:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "empty_file",
            "media artifact is empty",
        )
    if max_bytes is not None and file_stat.st_size > max_bytes:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "byte_limit_exceeded",
            "media artifact exceeds byte limit",
        )
    try:
        with path.open("rb") as source:
            prefix = source.read(65_536)
    except OSError:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "unreadable_file",
            "media artifact cannot be inspected",
        )
    detected_mime_type = detect_media_type(prefix, file_stat.st_size)
    detected_family = (
        None if detected_mime_type is None else media_family(detected_mime_type)
    )
    if declared_mime_type not in MEDIA_EXTENSIONS:
        return MediaInspection(
            declared_mime_type=declared_mime_type,
            declared_family=declared_family,
            detected_mime_type=detected_mime_type,
            detected_family=detected_family,
        )
    if detected_mime_type is None:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            None,
            "unrecognized_signature",
            "media signature is not recognized",
        )
    if detected_mime_type != declared_mime_type:
        return _inspection_error(
            declared_mime_type,
            declared_family,
            detected_mime_type,
            "declared_mime_mismatch",
            "detected media type does not match declared media type",
        )
    if declared_family == "image":
        try:
            if retain_image:
                image, source_mode = load_normalized_image(
                    path,
                    alpha_background="white",
                    policy=policy,
                )
            else:
                image = None
                source_mode = _verify_image_file(path, policy=policy)
        except _ImagePixelLimitError:
            return _inspection_error(
                declared_mime_type,
                declared_family,
                detected_mime_type,
                "image_pixel_limit_exceeded",
                "image exceeds the decoded pixel limit",
            )
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            return _inspection_error(
                declared_mime_type,
                declared_family,
                detected_mime_type,
                "image_pixel_limit_exceeded",
                "image exceeds the decoded pixel limit",
            )
        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ):
            return _inspection_error(
                declared_mime_type,
                declared_family,
                detected_mime_type,
                "invalid_image",
                "image failed safe decoding",
            )
        return MediaInspection(
            declared_mime_type=declared_mime_type,
            declared_family=declared_family,
            detected_mime_type=detected_mime_type,
            detected_family=detected_family,
            image=image,
            source_mode=source_mode,
        )
    if declared_mime_type == "application/zip":
        try:
            with zipfile.ZipFile(path) as archive:
                valid_archive = archive.testzip() is None
        except (OSError, zipfile.BadZipFile, RuntimeError):
            valid_archive = False
        if not valid_archive:
            return _inspection_error(
                declared_mime_type,
                declared_family,
                detected_mime_type,
                "invalid_archive",
                "archive failed verification",
            )
    return MediaInspection(
        declared_mime_type=declared_mime_type,
        declared_family=declared_family,
        detected_mime_type=detected_mime_type,
        detected_family=detected_family,
    )


def verify_media_file(
    path: Path,
    declared_mime_type: str,
    *,
    max_bytes: int,
    policy: MediaVerificationPolicy,
) -> None:
    inspection = _inspect_media_file(
        path,
        declared_mime_type,
        policy=policy,
        max_bytes=max_bytes,
        retain_image=False,
    )
    if inspection.image is not None:
        inspection.image.close()
    if not inspection.valid:
        raise MediaVerificationError(inspection)


def detect_media_type(prefix: bytes, size: int) -> str | None:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "image/webp"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    box_size = int.from_bytes(prefix[:4], "big") if len(prefix) >= 4 else 0
    if (
        len(prefix) >= 16
        and prefix[4:8] == b"ftyp"
        and 16 <= box_size <= size
        and prefix[8:12] != b"\x00\x00\x00\x00"
    ):
        return "video/mp4"
    if prefix.startswith(b"\x1aE\xdf\xa3") and b"webm" in prefix:
        return "video/webm"
    if _valid_mp3(prefix, size):
        return "audio/mpeg"
    if (
        len(prefix) >= 12
        and prefix[:4] == b"RIFF"
        and prefix[8:12] == b"WAVE"
        and int.from_bytes(prefix[4:8], "little") + 8 <= size
    ):
        return "audio/wav"
    segment_count = prefix[26] if len(prefix) >= 27 else 0
    header_size = 27 + segment_count
    payload_size = (
        sum(prefix[27:header_size]) if len(prefix) >= header_size else size + 1
    )
    if (
        len(prefix) >= header_size
        and prefix[:4] == b"OggS"
        and prefix[4] == 0
        and header_size + payload_size <= size
    ):
        return "audio/ogg"
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "application/zip"
    return None


def load_normalized_image(
    path: Path,
    *,
    alpha_background: AlphaBackground,
    policy: MediaVerificationPolicy,
) -> tuple[Image.Image, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            _validate_image_size(source.size, policy)
            source.load()
            source_mode = source.mode
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in source.info:
                rgba = oriented.convert("RGBA")
                background = _alpha_background(rgba.size, alpha_background)
                background.paste(rgba, mask=rgba.getchannel("A"))
                normalized = background
            else:
                normalized = oriented.convert("RGB")
    return normalized, source_mode


def _verify_image_file(
    path: Path,
    *,
    policy: MediaVerificationPolicy,
) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            _validate_image_size(source.size, policy)
            source_mode = source.mode
            source.verify()
    return source_mode


def _validate_image_size(
    size: tuple[int, int],
    policy: MediaVerificationPolicy,
) -> None:
    width, height = size
    if width <= 0 or height <= 0 or width * height > policy.max_image_pixels:
        raise _ImagePixelLimitError


def _alpha_background(
    size: tuple[int, int],
    kind: AlphaBackground,
) -> Image.Image:
    if kind == "white":
        return Image.new("RGB", size, PHASH_ALPHA_MATTE)
    background = Image.new("RGB", size, TRANSPARENCY_CHECKER_LIGHT)
    draw = ImageDraw.Draw(background)
    for y in range(0, size[1], TRANSPARENCY_CHECKER_SIZE):
        for x in range(0, size[0], TRANSPARENCY_CHECKER_SIZE):
            if ((x // TRANSPARENCY_CHECKER_SIZE) + (y // TRANSPARENCY_CHECKER_SIZE)) % 2:
                draw.rectangle(
                    (
                        x,
                        y,
                        min(x + TRANSPARENCY_CHECKER_SIZE - 1, size[0] - 1),
                        min(y + TRANSPARENCY_CHECKER_SIZE - 1, size[1] - 1),
                    ),
                    fill=TRANSPARENCY_CHECKER_DARK,
                )
    return background


def _inspection_error(
    declared_mime_type: str,
    declared_family: MediaFamily,
    detected_mime_type: str | None,
    code: MediaErrorCode,
    message: str,
) -> MediaInspection:
    return MediaInspection(
        declared_mime_type=declared_mime_type,
        declared_family=declared_family,
        detected_mime_type=detected_mime_type,
        detected_family=(
            None if detected_mime_type is None else media_family(detected_mime_type)
        ),
        error_code=code,
        error_message=message,
    )


def _valid_mp3(prefix: bytes, size: int) -> bool:
    if prefix.startswith(b"ID3") and len(prefix) >= 10:
        encoded_size = prefix[6:10]
        if any(value >= 128 for value in encoded_size):
            return False
        tag_size = (
            (encoded_size[0] << 21)
            | (encoded_size[1] << 14)
            | (encoded_size[2] << 7)
            | encoded_size[3]
        )
        return 10 + tag_size <= size
    return (
        len(prefix) >= 4
        and prefix[0] == 0xFF
        and prefix[1] & 0xE0 == 0xE0
        and prefix[1] & 0x18 != 0x08
        and prefix[2] & 0xF0 != 0xF0
    )
