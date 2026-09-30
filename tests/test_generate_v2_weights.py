"""Regression test for incomplete IndexTTS 2.0 installations."""

import pytest

from mlx_indextts.generate_v2 import IndexTTSv2


def test_missing_gpt_is_rejected_before_generation(tmp_path):
    (tmp_path / "config.yaml").write_text("gpt: {}\n", encoding="utf-8")
    (tmp_path / "s2mel.safetensors").touch()
    (tmp_path / "bigvgan.safetensors").touch()

    with pytest.raises(FileNotFoundError, match="gpt.safetensors"):
        IndexTTSv2(str(tmp_path))
