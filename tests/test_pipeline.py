import numpy as np
import pytest

from kodachrome.artifacts import Artifacts, write_artifact
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, sha1_hex
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import Pipeline


@pytest.fixture
def pipeline(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(tmp_path))


def test_process_shapes_and_info(pipeline):
    frame = np.random.default_rng(0).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out, info = pipeline.process(frame, rng=np.random.default_rng(0))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert set(info) == {"wb_gains", "exposure_gain", "clamped", "lut_sha1"}
    assert len(info["wb_gains"]) == 3
    assert info["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    assert set(info["clamped"]) == {"wb", "exposure"}


def test_grain_can_be_skipped(pipeline):
    frame = np.full((36, 64, 3), 128, dtype=np.uint8)
    no_grain, _ = pipeline.process(frame, grain=False)
    with_grain, _ = pipeline.process(frame, grain=True, rng=np.random.default_rng(0))
    assert no_grain.std() < with_grain.std()


def test_same_seed_reproduces_the_output(pipeline):
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    a, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    b, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_process_rejects_wrong_input(pipeline):
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4), dtype=np.uint8))


def test_disabled_grain_in_artifact_wins(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams(enabled=False))
    pipe = Pipeline(Artifacts.load(tmp_path))
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    out, _ = pipe.process(frame, grain=True, rng=np.random.default_rng(0))
    assert out.std() < 1.0
