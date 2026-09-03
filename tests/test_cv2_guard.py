import builtins

import pytest

from kodachrome._cv2 import require_cv2


def test_require_cv2_returns_the_module():
    cv2 = require_cv2()
    assert hasattr(cv2, "LUT")


def test_missing_cv2_names_both_remedies(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as exc:
        require_cv2()
    message = str(exc.value)
    assert "python3-opencv" in message      # the Pi remedy
    assert "[opencv]" in message            # the pip remedy
