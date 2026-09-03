import numpy as np
import pytest

from kodachrome.lut import LUT3D, CubeError, read_cube, sha1_hex, write_cube


def _smooth_test_lut(n=17):
    """Identity warped by a smooth, invertible tweak so interpolation is exercised."""
    ident = LUT3D.identity(n).table
    warped = ident.copy()
    warped[..., 0] = ident[..., 0] ** 1.2
    warped[..., 1] = 0.9 * ident[..., 1] + 0.1 * ident[..., 0]
    warped[..., 2] = np.clip(ident[..., 2] * 1.05 - 0.02, 0, 1)
    return LUT3D(warped)


def test_identity_leaves_image_unchanged():
    lut = LUT3D.identity(33)
    rgb = np.random.default_rng(0).random((20, 30, 3), dtype=np.float32)
    assert np.allclose(lut.apply_numpy(rgb), rgb, atol=1e-6)
    u8 = (rgb * 255).round().astype(np.uint8)
    # Pillow stores the table in 16-bit fixed point, so allow one 8-bit level
    assert np.abs(lut.apply_pillow(u8).astype(int) - u8.astype(int)).max() <= 1


def test_flat_order_is_red_fastest():
    flat = LUT3D.identity(2).to_flat()
    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float32,
    )
    assert np.array_equal(flat, expected)
    assert np.array_equal(LUT3D.from_flat(flat, 2).table, LUT3D.identity(2).table)


def test_cube_roundtrip_and_domain_written(tmp_path):
    lut = _smooth_test_lut(9)
    path = tmp_path / "t.cube"
    write_cube(lut, path, title="test")
    back = read_cube(path)
    assert back.size == 9
    assert np.allclose(back.table, lut.table, atol=1e-6)
    text = path.read_text()
    assert text.startswith('TITLE "test"\nLUT_3D_SIZE 9\n')
    assert "DOMAIN_MIN 0.0 0.0 0.0" in text and "DOMAIN_MAX 1.0 1.0 1.0" in text


def test_numpy_and_pillow_agree():
    lut = _smooth_test_lut(17)
    u8 = np.random.default_rng(3).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ref = np.round(lut.apply_numpy(u8.astype(np.float32) / 255.0) * 255).astype(int)
    diff = np.abs(ref - lut.apply_pillow(u8).astype(int))
    assert diff.max() <= 1
    assert diff.mean() < 0.3


def test_sha1_is_stable_and_content_sensitive():
    a = sha1_hex(LUT3D.identity(9))
    assert a == sha1_hex(LUT3D.identity(9))
    assert len(a) == 40
    tweaked = LUT3D.identity(9).table.copy()
    tweaked[4, 4, 4, 0] += 0.01
    assert sha1_hex(LUT3D(tweaked)) != a


@pytest.mark.parametrize(
    "text, message",
    [
        ("LUT_3D_SIZE 1\n0 0 0\n", "2..65"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "expected 8"),
        ("LUT_3D_SIZE 2\n" + "0 0 x\n" * 8, "line 2"),
        ("LUT_1D_SIZE 4\n", "1D"),
        ("0 0 0\n", "LUT_3D_SIZE"),
        ("LUT_3D_SIZE 2\n" + "0 0 nan\n" * 8, "finite"),
        ("LUT_3D_SIZE 2\n" + "0 0 2.0\n" * 8, "[0, 1]"),
        ("LUT_3D_SIZE 2\nDOMAIN_MAX 2.0 2.0 2.0\n" + "0 0 0\n" * 8, "DOMAIN"),
    ],
)
def test_cube_errors(tmp_path, text, message):
    path = tmp_path / "bad.cube"
    path.write_text(text)
    with pytest.raises(CubeError, match=re_escape(message)):
        read_cube(path)


def re_escape(s):
    import re

    return re.escape(s)


@pytest.mark.parametrize(
    "table, message",
    [
        (np.zeros((3, 3, 2, 3), dtype=np.float32), "shape"),
        (np.zeros((66, 66, 66, 3), dtype=np.float32), "2..65"),
        (np.full((4, 4, 4, 3), np.nan, dtype=np.float32), "finite"),
        (np.full((4, 4, 4, 3), 1.5, dtype=np.float32), "[0, 1]"),
    ],
)
def test_table_validation(table, message):
    with pytest.raises(ValueError, match=re_escape(message)):
        LUT3D(table)
