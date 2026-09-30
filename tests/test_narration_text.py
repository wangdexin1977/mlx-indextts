import json
from pathlib import Path

import mlx_indextts.webui as webui
from mlx_indextts.narration_text import (
    assign_voices_to_units,
    clean_narration_text,
    split_narration_units,
)


def test_clean_narration_text_removes_non_spoken_markup_conservatively():
    source = "# 标题（12）\n\n今天是 2026 年 9 月 23 日，[说明文字]。*** 😀\nhttps://example.com"

    cleaned = clean_narration_text(source)

    assert cleaned == "标题\n\n今天是 2026 年 9 月 23 日,说明文字。"
    assert "2026" in cleaned
    assert "说明文字" in cleaned
    assert "（12）" not in cleaned
    assert "http" not in cleaned


def test_split_and_round_robin_assignment_support_paragraph_line_and_sentence():
    text = "第一段。\n\n第二段！\n\n第三段？"

    paragraphs = split_narration_units(text, "paragraph")
    sentences = split_narration_units(text, "sentence")
    assignments = assign_voices_to_units(paragraphs, ["voice-a", "voice-b"], 1)

    assert paragraphs == ["第一段。", "第二段！", "第三段？"]
    assert sentences == paragraphs
    assert [voice_id for _unit, voice_id in assignments] == [
        "voice-a",
        "voice-b",
        "voice-a",
    ]
    assert split_narration_units("甲\n乙", "line") == ["甲", "乙"]


def test_multi_voice_generation_merges_all_assigned_units(tmp_path: Path, monkeypatch):
    voices = {
        "voice-a": {"id": "voice-a", "name": "甲声"},
        "voice-b": {"id": "voice-b", "name": "乙声"},
    }
    calls = []

    monkeypatch.setattr(webui, "_load_voice_entry", lambda voice_id: voices.get(voice_id))
    monkeypatch.setattr(webui, "_activate_voice_entry", lambda *_args, **_kwargs: (None, "", None))
    monkeypatch.setattr(webui, "update_user_config", lambda **_kwargs: None)
    monkeypatch.setattr(
        webui,
        "read_user_config",
        lambda: {"voice_library_id": "voice-a"},
    )
    monkeypatch.setattr(webui, "_start_sleep_prevention", lambda: None)
    monkeypatch.setattr(webui, "_stop_sleep_prevention", lambda _process: None)

    def fake_synthesize(**kwargs):
        calls.append((kwargs["text"], kwargs["voice_library_id"], kwargs["omnivoice_ref_text"], kwargs["persist_output_settings"]))
        target = Path(kwargs["output_directory"]) / f"part-{len(calls)}.wav"
        target.write_bytes(f"part-{len(calls)}".encode())
        return str(target), "生成完成", str(target)

    def fake_concat(paths, target, _gap):
        target.write_bytes(b"".join(path.read_bytes() for path in paths))

    monkeypatch.setattr(webui, "_synthesize_unlocked", fake_synthesize)
    monkeypatch.setattr(webui, "_concatenate_wav_batches", fake_concat)

    output, status, location = webui.synthesize_multi_voice(
        "第一段。\n\n第二段。\n\n第三段。",
        ["voice-a", "voice-b"],
        "paragraph",
        1,
        {
            "model_backend": "OmniVoice",
            "omnivoice_mode": "clone",
            "fish_mode": "clone",
            "output_format": "wav",
            "output_directory": str(tmp_path),
            "interval_silence": 250,
        },
    )

    assert calls == [
        ("第一段。", "voice-a", "", False),
        ("第二段。", "voice-b", "", False),
        ("第三段。", "voice-a", "", False),
    ]
    assert Path(output).read_bytes() == b"part-1part-2part-3"
    assert output == location
    assert "3 个单元" in status
    assert "甲声、乙声" in status


def test_saved_multi_voice_temporary_directory_recovers_parent(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "output_directory": str(tmp_path / ".multi_voice_old"),
    }), encoding="utf-8")
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    assert webui.read_user_config()["output_directory"] == str(tmp_path)
