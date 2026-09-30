"""The IndexTTS 2.0 reset keeps the measured faster setting model-specific."""

from mlx_indextts import webui


def test_indextts2_reset_uses_balanced_flow_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(webui, "OUTPUT_DIR", tmp_path)

    webui.switch_model_settings("IndexTTS 2.0")
    webui.reset_selected_model_settings("IndexTTS 2.0")
    assert webui.read_user_config()["diffusion_steps"] == 16

    webui.switch_model_settings("IndexTTS 2.5")
    assert webui.read_user_config()["diffusion_steps"] == 25
