import json

import numpy as np
from PIL import Image

from kodachrome.imageio import save_jpeg
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams
from kodachrome.train.dataset import CorpusSplit, PixelPool, SampleConfig
from kodachrome.train.evaluate import check_gates
from kodachrome.train.report import (
    render_contact_sheet,
    render_diagnostics,
    render_ramps,
    write_report,
)


def _images(dir_path, n, seed):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = dir_path / f"{i}.jpg"
        save_jpeg(rng.integers(30, 220, (60, 80, 3), dtype=np.uint8), p)
        paths.append(p)
    return paths


def _pool(seed, n_images=3):
    rng = np.random.default_rng(seed)
    return PixelPool(
        srgb=rng.random((1500, 3), dtype=np.float32),
        n_images=n_images,
        clamp_rate=0.25,
        wb_gains=[[0.9, 1.0, 1.1]] * n_images,
        exposure_gains=[1.1] * n_images,
        profiles={"sRGB (assumed)": n_images},
    )


def _split(tmp_path, name, seed):
    paths = _images(tmp_path / name, 4, seed)
    return CorpusSplit(paths[:3], paths[3:], _pool(seed), _pool(seed + 1, 1), "abc123")


def _darkening_lut(n=9):
    return LUT3D(LUT3D.identity(n).table**1.5)


def test_render_ramps_writes_a_readable_png(tmp_path):
    out = render_ramps(LUT3D.identity(9), tmp_path / "ramps.png")
    with Image.open(out) as im:
        assert im.size[0] >= 512 and im.size[1] > 100


def test_render_diagnostics_writes_a_png(tmp_path):
    out = render_diagnostics(_pool(0), _pool(1), tmp_path / "diag.png")
    with Image.open(out) as im:
        assert im.size[0] > 200 and im.size[1] > 100


def test_contact_sheet_shows_three_rows(tmp_path):
    src = _images(tmp_path / "src", 3, 0)
    tgt = _images(tmp_path / "tgt", 3, 1)
    out = render_contact_sheet(
        src, tgt, _darkening_lut(), NormalizeParams(), NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80), tmp_path / "sheet.png", n=3, thumb=64,
    )
    with Image.open(out) as im:
        assert im.height > 3 * 64  # three labelled rows stacked


def test_write_report_produces_every_artifact(tmp_path):
    metrics = {
        "swd_before": 0.1, "swd_after": 0.05, "swd_identity": 0.1, "swd_seed_spread": 0.001,
        "train_swd_before": 0.1, "train_swd_after": 0.04,
        "transport_gamut_clip_deltaE": 0.001, "lut_fit_rms_deltaE": 0.01,
        "grey_axis_monotone": True, "channel_monotone": True,
        "neutral_axis_max_chroma": 0.004, "clipped_volume_fraction": 0.01,
        "hue_bins": [{"bin": 0, "hue_deg": 7.5, "count": 10, "delta_L": -0.01,
                      "chroma_ratio": 1.1, "delta_hue_deg": 0.5}],
    }
    out_dir = tmp_path / "report"
    write_report(
        out_dir,
        _darkening_lut(),
        metrics,
        check_gates(metrics),
        _split(tmp_path, "src", 0),
        _split(tmp_path, "tgt", 5),
        NormalizeParams(),
        NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80),
    )
    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "ramps.png").exists()
    assert (out_dir / "diagnostics.png").exists()
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["swd_after"] == 0.05
    assert saved["gates"][0]["passed"] in (True, False)
    summary = (out_dir / "summary.txt").read_text()
    assert "held-out" in summary and "PASS" in summary
