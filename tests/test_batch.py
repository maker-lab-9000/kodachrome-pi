import numpy as np
import pytest

from kodachrome.capture.batch import main, output_path, process_dir, select_inputs
from kodachrome.imageio import save_jpeg


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)


def _capture_dir(tmp_path):
    """Looks like a real capture folder: originals plus already-graded siblings."""
    d = tmp_path / "shots"
    d.mkdir()
    for stem in ("120001", "120002"):
        save_jpeg(_img(1), d / f"{stem}_original.jpg")
        save_jpeg(_img(2), d / f"{stem}_kodachrome.jpg")
    (d / "captures.jsonl").write_text("{}\n")
    return d


def test_select_inputs_prefers_originals_and_always_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")))]
    assert chosen == ["120001_original.jpg", "120002_original.jpg"]


def test_select_inputs_all_still_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    save_jpeg(_img(3), d / "loose.jpg")
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")), all_files=True)]
    assert "loose.jpg" in chosen
    assert not any("_kodachrome" in n for n in chosen)


def test_plain_folder_processes_everything(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    save_jpeg(_img(2), d / "b.jpg")
    assert len(select_inputs(sorted(d.glob("*.jpg")))) == 2


def test_capture_dir_is_not_double_graded(tmp_path):
    d = _capture_dir(tmp_path)
    result = process_dir(d, tmp_path / "out")
    assert [p.name for p in result.written] == [
        "120001_original_kodachrome.jpg",
        "120002_original_kodachrome.jpg",
    ]
    assert result.skipped_graded == 2


def test_same_stem_different_extensions_do_not_collide(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    from PIL import Image

    Image.fromarray(_img(2)).save(d / "a.png")
    written = {p.name for p in process_dir(d, tmp_path / "out").written}
    assert written == {"a_jpg_kodachrome.jpg", "a_png_kodachrome.jpg"}


def test_output_path_without_disambiguation():
    from pathlib import Path

    assert output_path(Path("x/a.jpg"), Path("out"), False).name == "a_kodachrome.jpg"
    assert output_path(Path("x/a.jpg"), Path("out"), True).name == "a_jpg_kodachrome.jpg"


def test_existing_outputs_are_skipped_then_overwritten(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    out = tmp_path / "out"
    first = process_dir(d, out)
    assert len(first.written) == 1
    second = process_dir(d, out)
    assert second.written == [] and second.skipped_existing == 1
    third = process_dir(d, out, overwrite=True)
    assert len(third.written) == 1


def test_nested_or_identical_output_is_refused(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d)
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d / "sub")


def test_main_uses_the_packaged_default_from_any_cwd(tmp_path, monkeypatch, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    monkeypatch.chdir(tmp_path)
    assert main([str(d), str(tmp_path / "out")]) == 0
    assert "1 image" in capsys.readouterr().out


def test_main_reports_empty_input(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    assert main([str(tmp_path / "in"), str(tmp_path / "out")]) == 1
    assert "no images" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    assert main([str(d), str(tmp_path / "out"), "--artifacts", str(tmp_path / "none")]) == 2
    assert "params.json" in capsys.readouterr().err
