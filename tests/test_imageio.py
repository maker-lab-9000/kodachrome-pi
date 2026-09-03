import numpy as np
import pytest
from PIL import Image, ImageCms

from kodachrome.imageio import ImageMeta, list_images, load_rgb, save_jpeg, srgb_profile


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 200  # red-dominant: proves channel order survives
    out = save_jpeg(img, tmp_path / "nested" / "a.jpg", quality=100)
    assert out.exists()
    back, meta = load_rgb(out)
    assert back.shape == (10, 20, 3) and back.dtype == np.uint8
    assert back[..., 0].mean() > 150 and back[..., 2].mean() < 30
    assert isinstance(meta, ImageMeta) and meta.width == 20 and meta.height == 10


def test_saved_jpeg_carries_an_srgb_profile(tmp_path):
    out = save_jpeg(np.full((8, 8, 3), 120, dtype=np.uint8), tmp_path / "p.jpg")
    with Image.open(out) as im:
        assert im.info.get("icc_profile")
    _, meta = load_rgb(out)
    assert "sRGB" in meta.profile or "embedded" in meta.profile


def test_exif_orientation_is_applied(tmp_path):
    # A 4x2 image tagged orientation=6 (rotate 90 CW on display) must come back 2x4.
    base = Image.new("RGB", (4, 2), (10, 20, 30))
    exif = base.getexif()
    exif[274] = 6  # Orientation
    path = tmp_path / "rot.jpg"
    base.save(path, exif=exif)
    arr, meta = load_rgb(path)
    assert arr.shape[:2] == (4, 2)  # height, width swapped
    assert meta.oriented is True


def test_icc_profile_is_converted_not_ignored(tmp_path, wide_gamut_icc):
    """The same bytes under a wide-gamut profile must not decode to the same pixels."""
    pixels = np.full((8, 8, 3), (200, 60, 60), dtype=np.uint8)
    srgb_path = tmp_path / "srgb.jpg"
    wide_path = tmp_path / "wide.jpg"
    Image.fromarray(pixels).save(
        srgb_path, quality=100, icc_profile=ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    )
    Image.fromarray(pixels).save(wide_path, quality=100, icc_profile=wide_gamut_icc)

    srgb_arr, srgb_meta = load_rgb(srgb_path)
    wide_arr, wide_meta = load_rgb(wide_path)

    assert wide_meta.profile_error is None
    assert "Wide Gamut" in wide_meta.profile
    assert not np.array_equal(srgb_arr, wide_arr), "a non-sRGB profile must change the pixels"
    # Verified against the real Adobe RGB (1998) profile: (200, 60, 60) lands on
    # (231, 57, 56) in sRGB. Allow a little slack for JPEG and littlecms rounding.
    assert abs(int(wide_arr[0, 0, 0]) - 231) <= 3
    assert abs(int(wide_arr[0, 0, 2]) - 56) <= 3
    # An sRGB-tagged image must come back essentially unchanged.
    assert np.abs(srgb_arr.astype(int) - pixels.astype(int)).max() <= 2


def test_malformed_profile_falls_back_and_reports(tmp_path):
    path = tmp_path / "bad.jpg"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(path, icc_profile=b"not-a-profile")
    arr, meta = load_rgb(path)
    assert arr.shape == (8, 8, 3)
    assert meta.profile == "invalid"
    assert meta.profile_error


def test_colour_manage_can_be_disabled(tmp_path):
    path = tmp_path / "x.jpg"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(path)
    arr, meta = load_rgb(path, colour_manage=False)
    assert arr.shape == (8, 8, 3)
    assert "assumed" in meta.profile


def test_load_converts_modes(tmp_path):
    Image.new("L", (8, 8), 77).save(tmp_path / "grey.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "rgba.png")
    assert load_rgb(tmp_path / "grey.png")[0].shape == (8, 8, 3)
    assert load_rgb(tmp_path / "rgba.png")[0].shape == (8, 8, 3)


def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.JPG", "a.jpeg", "c.png", "notes.txt", "d.tif"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpeg", "b.JPG", "c.png", "d.tif"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "nope.jpg")
