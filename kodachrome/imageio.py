"""Image file I/O, and the one place colour management happens.

Every pixel that enters this project comes through ``load_rgb``, which does
two things no other module should have to remember:

**EXIF orientation.** A camera that was held sideways records the rotation
as a tag rather than rotating the pixels. The trainer crops a fixed 6% from
each edge, so an unoriented image gets the wrong edges cropped, and a
portrait frame would be sampled as landscape. ``ImageOps.exif_transpose``
resolves the tag into real pixel order.

**ICC profiles.** The whole project is a colour measurement, so treating an
Adobe RGB or ProPhoto scan as if it were sRGB would bake a systematic error
into the learned look - the more so because the target corpus is archival
scans whose colour management is part of what we are matching. When a
profile is embedded, the image is converted to sRGB with ``ImageCms``. When
none is present, sRGB is assumed, which is the web convention and correct
for Commons JPEGs. When one is present but unreadable, sRGB is assumed and
the failure is recorded rather than swallowed, so the report can count it.

``ImageMeta`` travels with the pixels so the trainer can publish profile
statistics: a corpus that is secretly half Adobe RGB should be visible, not
silently averaged in.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_EXIF_ORIENTATION_TAG = 274


@dataclass
class ImageMeta:
    profile: str
    oriented: bool
    profile_error: str | None
    width: int
    height: int


@lru_cache(maxsize=1)
def srgb_profile():
    """The sRGB profile used as the working space, built once."""
    return ImageCms.createProfile("sRGB")


def _describe(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileDescription(profile).strip() or "unnamed profile"
    except Exception:  # noqa: BLE001 - a profile can be readable but have no description
        return "unnamed profile"


def load_rgb(path: str | Path, *, colour_manage: bool = True) -> tuple[np.ndarray, ImageMeta]:
    """Load an image as sRGB RGB uint8, applying EXIF orientation and ICC conversion."""
    path = Path(path)
    with Image.open(path) as im:
        exif = im.getexif()
        oriented = bool(exif.get(_EXIF_ORIENTATION_TAG, 1) not in (1, None))
        im = ImageOps.exif_transpose(im)

        raw_profile = im.info.get("icc_profile")
        profile_name = "sRGB (assumed)"
        profile_error: str | None = None

        if raw_profile and colour_manage:
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(raw_profile))
                profile_name = _describe(src)
                # Do not convert the mode first: littlecms needs the image in the
                # colour space the profile describes. Pre-converting a LAB or CMYK
                # image to RGB makes the transform unbuildable, and the fallback
                # below would then silently mislabel a perfectly good profile.
                im = ImageCms.profileToProfile(
                    im, src, srgb_profile(), renderingIntent=0, outputMode="RGB"
                )
            except Exception as exc:  # noqa: BLE001 - any malformed profile falls back to sRGB
                profile_name = "invalid"
                profile_error = f"{type(exc).__name__}: {exc}"
        elif raw_profile:
            profile_name = "sRGB (assumed, colour management off)"

        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    return rgb, ImageMeta(
        profile=profile_name,
        oriented=oriented,
        profile_error=profile_error,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )


def save_jpeg(
    rgb_u8: np.ndarray, path: str | Path, quality: int = 95, embed_srgb: bool = True
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB")
    kwargs = {}
    if embed_srgb:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    image.save(path, "JPEG", quality=quality, **kwargs)
    return path


def list_images(dir_path: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
