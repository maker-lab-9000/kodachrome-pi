"""Build a wheel, install it into a clean venv, and run a command from elsewhere.

This is the only test that proves the shipped artifact really travels with
the package: everything else imports from the source tree, where
`kodachrome/data/` happens to be on disk anyway.
"""

import subprocess
import sys
import venv

import numpy as np
import pytest

from kodachrome.imageio import save_jpeg

pytestmark = pytest.mark.slow


def test_wheel_installs_and_runs_from_another_directory(tmp_path, repo_root):
    dist = tmp_path / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(repo_root)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1

    env_dir = tmp_path / "env"
    venv.create(env_dir, with_pip=True)
    python = env_dir / "bin" / "python"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", f"{wheels[0]}[opencv]"],
        capture_output=True, text=True,
    )
    assert install.returncode == 0, install.stderr[-3000:]

    work = tmp_path / "elsewhere"
    (work / "in").mkdir(parents=True)
    save_jpeg(
        np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8),
        work / "in" / "a.jpg",
    )
    run = subprocess.run(
        [str(env_dir / "bin" / "kodachrome-process"), "in", "out"],
        cwd=work, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert (work / "out" / "a_kodachrome.jpg").exists(), "the bundled artifact was not found"
