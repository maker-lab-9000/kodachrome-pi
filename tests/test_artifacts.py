import json

import numpy as np
import pytest

from kodachrome.artifacts import (
    PARAMS_VERSION,
    Artifacts,
    ArtifactsError,
    publish,
    write_artifact,
)
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, sha1_hex, write_cube
from kodachrome.normalize import NormalizeParams


@pytest.fixture
def staged(tmp_path):
    d = tmp_path / "staging"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams(), training={"note": "t"})
    return d


def test_write_then_load(staged):
    data = json.loads((staged / "params.json").read_text())
    assert data["version"] == PARAMS_VERSION
    assert data["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    art = Artifacts.load(staged)
    assert art.lut.size == 9
    assert art.normalize == NormalizeParams()
    assert art.grain == GrainParams()
    assert art.training == {"note": "t"}
    assert art.lut_sha1 == data["lut_sha1"]


def test_missing_params_is_clear(tmp_path):
    with pytest.raises(ArtifactsError, match="params.json"):
        Artifacts.load(tmp_path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{not json", "JSON"),
        ('{"version": 99}', "version"),
        ('{"lut_file": "k.cube"}', "version"),
        ('[1, 2, 3]', "object"),
        ('{"version": 2, "normalize": 5}', "normalize"),
        ('{"version": 2, "grain": "x"}', "grain"),
    ],
)
def test_schema_rejections(tmp_path, payload, message):
    (tmp_path / "params.json").write_text(payload)
    with pytest.raises(ArtifactsError, match=message):
        Artifacts.load(tmp_path)


def test_missing_cube_is_clear(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"version": 2, "lut_file": "k.cube"}))
    with pytest.raises(ArtifactsError, match="k.cube"):
        Artifacts.load(tmp_path)


def test_lut_hash_mismatch_is_refused(staged):
    data = json.loads((staged / "params.json").read_text())
    data["lut_sha1"] = "0" * 40
    (staged / "params.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactsError, match="sha1"):
        Artifacts.load(staged)


def test_packaged_default_loads_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    art = Artifacts.default()
    assert art.lut.size >= 2
    assert art.lut_sha1 == sha1_hex(art.lut)


def test_resolve_prefers_the_override(staged, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Artifacts.resolve(staged).lut.size == 9
    assert Artifacts.resolve(None).path != staged


def test_publish_moves_only_after_validation(staged, tmp_path):
    dest = tmp_path / "live"
    published = publish(staged, dest)
    assert published == dest
    assert (dest / "kodachrome.cube").exists() and (dest / "params.json").exists()
    assert Artifacts.load(dest).lut.size == 9
    assert not staged.exists()


def test_publish_refuses_an_invalid_staging_dir_and_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    before = (dest / "params.json").read_text()

    bad = tmp_path / "bad"
    bad.mkdir()
    write_cube(LUT3D.identity(9), bad / "kodachrome.cube")
    (bad / "params.json").write_text(json.dumps({"version": 2, "lut_file": "kodachrome.cube",
                                                 "lut_sha1": "0" * 40}))
    with pytest.raises(ArtifactsError):
        publish(bad, dest)
    assert (dest / "params.json").read_text() == before
    assert Artifacts.load(dest).lut.size == 5


def test_publish_replaces_an_existing_artifact(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    staging = tmp_path / "new"
    write_artifact(staging, LUT3D.identity(9), NormalizeParams(), GrainParams())
    publish(staging, dest)
    assert Artifacts.load(dest).lut.size == 9


def test_committed_default_is_loadable(repo_root):
    art = Artifacts.load(repo_root / "kodachrome" / "data")
    assert art.lut.size in (2, 9, 33)
    assert np.isfinite(art.lut.table).all()
