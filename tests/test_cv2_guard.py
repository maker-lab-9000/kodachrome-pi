import builtins

import pytest

from kodachrome._cv2 import require_cv2


def test_require_cv2_returns_the_module():
    cv2 = require_cv2()
    assert hasattr(cv2, "LUT")


def _patch_cv2_import(monkeypatch, error):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_cv2_names_both_remedies(monkeypatch):
    # This is exactly what Python raises for an absent module: a
    # ModuleNotFoundError whose .name is the module. A bare ImportError with
    # the same text never occurs in practice.
    _patch_cv2_import(monkeypatch, ModuleNotFoundError("No module named 'cv2'", name="cv2"))
    with pytest.raises(ImportError) as exc:
        require_cv2()
    message = str(exc.value)
    assert "python3-opencv" in message      # the Pi remedy
    assert "[opencv]" in message            # the pip remedy


@pytest.mark.parametrize(
    "error",
    [
        ImportError("libGL.so.1: cannot open shared object file"),
        # The error text names the .so inside the cv2 package, so any check
        # that sniffs for "cv2" in the message misclassifies this one.
        ImportError(
            "/usr/lib/python3/dist-packages/cv2/cv2.abi3.so: undefined symbol: "
            "_ZN2cv6String8allocateEm"
        ),
    ],
    ids=["missing-libGL", "abi-mismatch"],
)
def test_a_broken_native_install_is_not_reported_as_missing(monkeypatch, error):
    """cv2 present but unable to load must not advise installing cv2."""
    _patch_cv2_import(monkeypatch, error)
    with pytest.raises(ImportError) as exc:
        require_cv2()
    message = str(exc.value)
    assert "failed to import" in message
    assert "libgl1" in message
    assert "is required but not installed" not in message


def test_a_missing_dependency_of_cv2_is_not_reported_as_missing_cv2(monkeypatch):
    """ModuleNotFoundError naming something else is an installation problem."""
    _patch_cv2_import(monkeypatch, ModuleNotFoundError("No module named 'numpy'", name="numpy"))
    with pytest.raises(ImportError) as exc:
        require_cv2()
    assert "failed to import" in str(exc.value)
