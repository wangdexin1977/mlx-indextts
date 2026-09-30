"""User-facing routing and settings for the optional CosyVoice3 backend."""

from mlx_indextts import webui


def test_cosyvoice3_is_selectable_with_private_quality_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "CONFIG_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(webui, "OUTPUT_DIR", tmp_path)
    webui.update_user_config(model_backend="Fish Audio S2 Pro", speed=1.2)

    controls = webui.switch_model_settings("CosyVoice 3")
    assert controls[-2]["visible"] is True
    webui.save_cosyvoice3_settings("参考原文", "fp32", 12, 8)
    webui.switch_model_settings("Fish Audio S2 Pro")
    assert webui.read_user_config()["speed"] == 1.2
    webui.switch_model_settings("CosyVoice 3")
    config = webui.read_user_config()
    assert config["cosy_nfe"] == 12
    assert config["cosy_precision"] == "fp32"


def test_cosyvoice3_controls_feed_existing_generate_event():
    demo = webui.build_ui()
    labels = {
        component["id"]: component.get("props", {}).get("label")
        for component in demo.config["components"]
    }
    selector = next(
        component for component in demo.config["components"]
        if component.get("props", {}).get("label") == "合成模型"
    )
    assert any(choice[1] == "CosyVoice 3" for choice in selector["props"]["choices"])
    event = next(
        dependency for dependency in demo.config["dependencies"]
        if dependency.get("api_name") == "synthesize_stream"
    )
    input_labels = {labels.get(component_id) for component_id in event["inputs"]}
    assert "语言模型精度" in input_labels
    assert "流匹配步数（音质优先推荐 10）" in input_labels
    assert "参考音频原文（留空将自动识别并按音色缓存）" in input_labels
