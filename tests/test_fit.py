import json

import numpy as np
import pytest

from kodachrome.artifacts import Artifacts
from kodachrome.color import lch_to_oklab, oklab_to_lch, oklab_to_srgb, srgb_to_oklab
from kodachrome.imageio import save_jpeg
from kodachrome.train.dataset import PixelPool, SampleConfig
from kodachrome.train.fit import FitConfig, fit, main, train


def _curve_and_rotation(lab, gamma=1.15, chroma=1.25, degrees=10.0):
    """A tone curve, a saturation boost, and a genuine hue rotation."""
    lch = oklab_to_lch(lab)
    lch[..., 0] = np.clip(lch[..., 0], 0, 1) ** gamma
    lch[..., 1] = lch[..., 1] * chroma
    lch[..., 2] = lch[..., 2] + np.deg2rad(degrees)
    return lch_to_oklab(lch)


def _pool(rng, n=30000):
    return (rng.random((n, 3), dtype=np.float32) * 0.7 + 0.15).astype(np.float32)


def test_fit_recovers_a_tone_curve_and_a_ten_degree_hue_rotation():
    """The spec claims this capability, so it is tested directly."""
    rng = np.random.default_rng(0)
    src = _pool(rng)
    transformed = oklab_to_srgb(_curve_and_rotation(srgb_to_oklab(_pool(rng))))
    tgt = np.clip(transformed, 0, 1).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    expected = _curve_and_rotation(srgb_to_oklab(held))
    got = srgb_to_oklab(result.lut.apply_numpy(held))
    assert float(np.sqrt(np.sum((got - expected) ** 2, axis=1)).mean()) < 0.03


def test_a_large_hue_rotation_is_only_partly_recovered():
    """Pins the documented limit: reweighting damps rotations beyond about one bin."""
    rng = np.random.default_rng(1)
    src = _pool(rng)
    tgt = np.clip(
        oklab_to_srgb(
            _curve_and_rotation(srgb_to_oklab(_pool(rng)), gamma=1.0, chroma=1.0, degrees=90.0)
        ),
        0, 1,
    ).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    before = oklab_to_lch(srgb_to_oklab(held))[:, 2]
    after = oklab_to_lch(srgb_to_oklab(result.lut.apply_numpy(held)))[:, 2]
    achieved = np.degrees(np.angle(np.exp(1j * (after - before)))).mean()
    # Measured: about +5 degrees out of the 90 requested, stable across seeds.
    # A third of the request is a generous ceiling that still fails loudly if
    # reweighting ever stops damping large rotations.
    assert abs(achieved) < 30.0, (
        f"a 90 degree rotation must be heavily damped, got {achieved:.1f} degrees"
    )


def test_strength_zero_gives_an_identity_lut():
    rng = np.random.default_rng(2)
    src = PixelPool(rng.random((5000, 3), dtype=np.float32), 1)
    tgt = PixelPool((rng.random((5000, 3), dtype=np.float32) * 0.5).astype(np.float32), 1)
    result = fit(src, tgt, FitConfig(lut_size=9, iterations=5, strength=0.0))
    x = rng.random((500, 3), dtype=np.float32)
    assert np.abs(result.lut.apply_numpy(x) - x).max() < 0.02


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"lut_size": 1}, "lut_size"),
        ({"iterations": 0}, "iterations"),
        ({"hue_bins": 0}, "hue_bins"),
        ({"strength": 1.5}, "strength"),
        ({"lambda_smooth": -1.0}, "lambda_smooth"),
    ],
)
def test_invalid_fit_config_names_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        FitConfig(**kwargs)


def _image_dir(dir_path, n, seed, transform=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = np.linspace(0.2, 0.8, 64, dtype=np.float32)[None, :, None] * rng.uniform(
            0.7, 1.0, 3
        ).astype(np.float32)
        img = np.repeat(base, 48, axis=0)
        if transform:
            img = transform(img)
        save_jpeg((np.clip(img, 0, 1) * 255).astype(np.uint8), dir_path / f"{i:02d}.jpg")


def test_train_end_to_end_publishes_a_loadable_artifact(tmp_path):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    metrics, gates = train(
        tmp_path / "src",
        tmp_path / "tgt",
        out,
        FitConfig(lut_size=9, iterations=5),
        SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25),
        grain=None,
        proxy_source=True,
        allow_small=True,
        command="test",
    )
    art = Artifacts.load(out)
    assert art.lut.size == 9
    assert art.training["source"]["proxy"] is True
    assert art.training["source"]["corpus_sha1"] and art.training["target"]["corpus_sha1"]
    assert art.training["split"]["n_source_val_images"] == 2
    assert art.training["fit"]["lut_size"] == 9
    assert art.training["metrics"]["swd_after"] == metrics["swd_after"]
    assert art.training["code_revision"]
    assert (out / "report" / "contact_sheet.png").exists()
    assert (out / "report" / "summary.txt").exists()
    assert isinstance(gates, list) and gates


def test_train_leaves_the_previous_artifact_intact_if_publication_fails(tmp_path, monkeypatch):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    good = (out / "params.json").read_text()

    monkeypatch.setattr(
        "kodachrome.train.fit.publish", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    assert (out / "params.json").read_text() == good


def test_main_rejects_identical_and_missing_dirs(tmp_path, capsys):
    _image_dir(tmp_path / "src", 2, 0)
    assert main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1
    assert "same" in capsys.readouterr().err
    assert main(["--source", str(tmp_path / "missing"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1


def test_main_refuses_a_small_corpus_then_accepts_the_flag(tmp_path, capsys):
    _image_dir(tmp_path / "src", 3, 0)
    _image_dir(tmp_path / "tgt", 3, 1, transform=lambda im: im**1.2)
    args = ["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
            "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
            "--max-side", "64", "--pixels-per-image", "300", "--proxy-source"]
    assert main(args) == 1
    assert "--allow-small" in capsys.readouterr().err
    assert main([*args, "--allow-small"]) in (0, 3)
    data = json.loads((tmp_path / "o" / "params.json").read_text())
    assert data["training"]["fit"]["strength"] == 1.0
    assert data["grain"]["strength"] == pytest.approx(0.025)


def test_main_reports_a_failed_gate_with_exit_code_3(tmp_path, monkeypatch, capsys):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    from kodachrome.train import fit as fit_module
    from kodachrome.train.evaluate import Gate

    monkeypatch.setattr(
        fit_module, "check_gates",
        lambda m: [Gate("improvement_exceeds_noise", 0.0, 0.1, False, "forced failure")],
    )
    code = main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
                 "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
                 "--max-side", "64", "--allow-small"])
    assert code == 3
    err = capsys.readouterr().err
    assert "improvement_exceeds_noise" in err
    assert Artifacts.load(tmp_path / "o").lut.size == 9  # written despite the failure
