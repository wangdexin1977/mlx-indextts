import shutil
import inspect
import time
import wave
from pathlib import Path

import gradio as gr
import pytest

import mlx_indextts.webui as webui


def test_generation_progress_reports_segments_percentage_and_eta(monkeypatch):
    monkeypatch.setattr(webui.time, "perf_counter", lambda: 130.0)
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="running",
            current=2,
            total=5,
            message="已完成第 2/5 段",
            started_at=100.0,
            paused_at=None,
            paused_seconds=0.0,
            finished_elapsed=None,
        )

    rendered = webui.render_generation_progress()

    assert "40%" in rendered
    assert "已完成百分比" in rendered
    assert "已完成片段" in rendered
    assert "剩余片段" in rendered
    assert "<strong>2</strong>已完成片段" in rendered
    assert "<strong>3</strong>剩余片段" in rendered
    assert "共 5 个片段" in rendered
    assert "已用 30秒" in rendered
    assert "预计剩余 45秒" in rendered
    assert 'aria-valuenow="40"' in rendered


def test_generation_progress_excludes_paused_time(monkeypatch):
    monkeypatch.setattr(webui.time, "perf_counter", lambda: 160.0)
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="paused",
            current=1,
            total=4,
            message="转换已暂停",
            started_at=100.0,
            paused_at=145.0,
            paused_seconds=15.0,
            finished_elapsed=None,
        )

    rendered = webui.render_generation_progress()

    assert "25%" in rendered
    assert "已用 30秒" in rendered
    assert "预计剩余 1分30秒" in rendered


def test_voice_preview_has_no_default_and_prefers_optimized_custom_audio(
    tmp_path: Path, monkeypatch
):
    custom_source = tmp_path / "custom.wav"
    optimized = tmp_path / "optimized.wav"
    for path in (custom_source, optimized):
        path.write_bytes(b"RIFF-test")
    monkeypatch.setattr(webui, "OPTIMIZED_VOICE_PATH", optimized)

    assert webui.resolve_voice_preview(None) is None
    assert webui.resolve_voice_preview(str(custom_source)) == str(optimized)


def test_completed_generation_always_displays_100_percent():
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="completed",
            current=3,
            total=3,
            message="全部片段已完成",
            started_at=100.0,
            paused_at=None,
            paused_seconds=0.0,
            finished_elapsed=42.0,
        )

    rendered = webui.render_generation_progress()

    assert "100%" in rendered
    assert "预计剩余 0秒" in rendered
    assert "已用 42秒" in rendered


def test_progress_stream_runs_only_for_active_generation(monkeypatch):
    def fake_synthesize(*_args, **_kwargs):
        with webui._generation_progress_lock:
            webui._generation_progress_state.update(
                state="running",
                current=1,
                total=2,
                message="已完成第 1/2 段",
                started_at=time.perf_counter(),
                paused_at=None,
                paused_seconds=0.0,
                finished_elapsed=None,
            )
        time.sleep(0.6)
        with webui._generation_progress_lock:
            webui._generation_progress_state.update(
                state="completed",
                current=2,
                total=2,
                message="全部完成",
                finished_elapsed=0.6,
            )
        return "result.mp3", "生成完成", "/tmp/result.mp3"

    monkeypatch.setattr(webui, "synthesize", fake_synthesize)
    stream = webui.synthesize_stream(
        "测试",
        None,
        "自然/平静",
        0.6,
        1.0,
        42,
        250,
        50,
        120,
        0.8,
        25,
        1500,
        0.8,
        30,
        10.0,
        0.7,
        False,
        "mp3",
        "/tmp",
    )

    updates = list(stream)

    assert len(updates) == 2
    assert "50%" in updates[0][2]
    assert updates[-1][0] == "result.mp3"
    assert "100%" in updates[-1][2]
    assert updates[-1][3] == "/tmp/result.mp3"


def test_live_playback_streams_each_completed_batch(tmp_path, monkeypatch):
    batch_directory = tmp_path / ".live-test.parts"
    batch_directory.mkdir()
    first_batch = batch_directory / "live_0001.wav"
    second_batch = batch_directory / "live_0002.wav"
    first_batch.write_bytes(b"first")
    second_batch.write_bytes(b"second")
    final_audio = tmp_path / "result.wav"
    final_audio.write_bytes(b"final")
    cleanup_requests = []

    def cleanup_immediately(directory):
        cleanup_requests.append(directory)
        shutil.rmtree(directory, ignore_errors=True)

    monkeypatch.setattr(webui, "_schedule_directory_cleanup", cleanup_immediately)

    def fake_synthesize(*_args, **kwargs):
        callback = kwargs["batch_ready_callback"]
        callback(str(first_batch))
        time.sleep(0.6)
        callback(str(second_batch))
        return str(final_audio), "生成完成", str(final_audio)

    monkeypatch.setattr(webui, "synthesize", fake_synthesize)
    updates = list(
        webui.synthesize_stream(
            "测试",
            None,
            "自然/平静",
            0.6,
            1.0,
            42,
            250,
            50,
            120,
            0.8,
            25,
            1500,
            0.8,
            30,
            10.0,
            0.7,
            False,
            "wav",
            str(tmp_path),
            True,
        )
    )

    preview_updates = [update[4] for update in updates if isinstance(update[4], gr.Audio)]
    assert [Path(component.value["path"]).name for component in preview_updates] == [
        "live_0001.wav",
        "live_0002.wav",
    ]
    assert all(component.autoplay for component in preview_updates)
    assert cleanup_requests == [batch_directory]
    assert not batch_directory.exists()


def test_live_playback_toggle_defaults_to_off_and_is_wired_to_generation():
    demo = webui.build_ui()
    components_by_id = {component["id"]: component for component in demo.config["components"]}
    toggle = next(
        component
        for component in demo.config["components"]
        if component.get("props", {}).get("label") == "边生成边播放"
    )
    generation_event = next(
        dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("api_name") == "synthesize_stream"
    )
    input_labels = [
        components_by_id[component_id].get("props", {}).get("label")
        for component_id in generation_event["inputs"]
    ]
    output_labels = [
        components_by_id[component_id].get("props", {}).get("label")
        for component_id in generation_event["outputs"]
    ]

    assert toggle["props"]["value"] is False
    assert "边生成边播放" in input_labels
    assert "实时试听（已完成批次）" in output_labels


def test_output_settings_are_validated_and_persisted(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "settings.json"
    output_directory = tmp_path / "custom-output"
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    webui.update_user_config(
        output_format="mp3",
        output_directory=str(output_directory),
    )
    config = webui.read_user_config()

    assert config["output_format"] == "mp3"
    assert config["output_directory"] == str(output_directory)
    assert webui._resolve_output_directory(str(output_directory)) == output_directory
    assert output_directory.is_dir()


def test_invalid_saved_output_format_falls_back_to_wav(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "settings.json"
    config_path.write_text('{"output_format": "aac"}', encoding="utf-8")
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    assert webui.read_user_config()["output_format"] == "wav"


def test_progress_emphasises_percentage_and_elapsed_time():
    rendered = webui.render_generation_progress()

    assert 'class="generation-progress-percent"' in rendered
    assert 'class="generation-progress-time"' in rendered
    assert ".generation-progress-percent { color: #dc2626" in webui.APP_CSS
    assert ".generation-progress-time { color: #dc2626" in webui.APP_CSS


def test_workspace_gives_text_and_voice_two_thirds_of_desktop_width():
    source = inspect.getsource(webui.build_ui)

    assert 'scale=2, min_width=0, elem_classes=["panel", "workspace-main"]' in source
    assert 'elem_classes=["workspace-sidebar"]' in source
    assert ".workspace-main" in webui.APP_CSS
    assert "width: 66.666% !important" in webui.APP_CSS
    assert ".workspace-sidebar" in webui.APP_CSS
    assert "width: 33.333% !important" in webui.APP_CSS


def test_text_editor_is_large_and_readable_without_modifying_content():
    source = inspect.getsource(webui.build_ui)

    assert "lines=16" in source
    assert "max_lines=28" in source
    assert "min-height: 420px !important" in webui.APP_CSS
    assert "font-size: 17px !important" in webui.APP_CSS
    assert "line-height: 1.85 !important" in webui.APP_CSS
    assert "min-height: 300px !important" in webui.APP_CSS


def test_synthesis_controls_are_collapsed_and_compact_by_default():
    source = inspect.getsource(webui.build_ui)

    assert '"02 · 合成参数｜点击展开设置"' in source
    assert "open=False" in source
    assert 'elem_classes=["compact-parameter-accordion"]' in source
    assert 'elem_classes=["compact-generation-controls"]' in source
    assert "min-height: 28px !important" in webui.APP_CSS
    assert "min-height: 34px !important" in webui.APP_CSS


def test_oversized_synthesis_is_rejected_before_model_work(monkeypatch):
    monkeypatch.setattr(
        webui,
        "get_model",
        lambda: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    with pytest.raises(gr.Error, match="超过长文合成上限"):
        webui._validate_synthesis_text("测" * (webui.MAX_SYNTHESIS_CHARACTERS + 1))


def test_text_counter_marks_oversized_input():
    rendered = webui.count_text_characters("测" * (webui.MAX_SYNTHESIS_CHARACTERS + 1))

    assert "text-counter-over-limit" in rendered
    assert f"/ {webui.MAX_SYNTHESIS_CHARACTERS:,} 字" in rendered


def test_long_text_limit_accepts_one_hundred_thousand_characters():
    text = "测" * 100_000

    assert webui._validate_synthesis_text(text) == text


def test_long_text_batches_preserve_content_and_stay_within_batch_limit():
    text = ("第一句用于测试自动拆分。\n第二句继续测试！" * 400) + ("尾" * 3_000)

    batches = webui._split_synthesis_batches(text)

    assert "".join("".join(batch.split()) for batch in batches) == "".join(text.split())
    assert all(
        webui.count_effective_characters(batch) <= webui.SYNTHESIS_BATCH_CHARACTERS
        for batch in batches
    )


def test_model_batches_recursively_stay_below_segment_guard():
    class Tokenizer:
        @staticmethod
        def tokenize(text):
            return list(text)

        @staticmethod
        def split_segments(tokens, max_tokens_per_segment):
            del max_tokens_per_segment
            return [[token] for token in tokens]

    class Model:
        tokenizer = Tokenizer()

    prepared = webui._prepare_model_batches("测" * 200, Model(), 120)

    assert "".join(batch for batch, _ in prepared) == "测" * 200
    assert all(count <= webui.MAX_SYNTHESIS_BATCH_SEGMENTS for _, count in prepared)


def test_duplicate_generation_is_rejected_while_background_job_is_active():
    assert webui._synthesis_job_lock.acquire(blocking=False)
    try:
        with pytest.raises(gr.Error, match="已有音频合成任务"):
            webui.synthesize()
    finally:
        webui._synthesis_job_lock.release()


def test_synthesis_holds_and_releases_idle_sleep_assertion(monkeypatch):
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 2
            return 0

        def kill(self):
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(webui.shutil, "which", lambda command: "/usr/bin/caffeinate" if command == "caffeinate" else None)
    monkeypatch.setattr(webui.subprocess, "Popen", lambda *args, **kwargs: process)

    started = webui._start_sleep_prevention()
    webui._stop_sleep_prevention(started)

    assert started is process
    assert process.terminated
    assert not process.killed


def test_wav_batches_are_joined_on_disk_with_inter_batch_gap(tmp_path: Path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for long-text batch merging")

    batch_paths = [tmp_path / "batch-1.wav", tmp_path / "batch-2.wav"]
    for batch_path in batch_paths:
        with wave.open(str(batch_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(22_050)
            audio_file.writeframes(b"\0\0" * 2_205)

    target_path = tmp_path / "joined.wav"
    webui._concatenate_wav_batches(batch_paths, target_path, gap_ms=100)

    with wave.open(str(target_path), "rb") as audio_file:
        assert audio_file.getframerate() == 22_050
        assert audio_file.getnframes() >= 6_500


def test_voice_upload_persists_immediately_without_building_conditioning(tmp_path: Path, monkeypatch):
    source = tmp_path / "uploaded.wav"
    source.write_bytes(b"RIFF-new-voice")
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    optimized_path = voice_dir / "optimized.wav"
    conditioning_path = voice_dir / "conditioning.npz"
    conditioning_path.parent.mkdir(parents=True)
    conditioning_path.write_bytes(b"old-cache")

    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)
    monkeypatch.setattr(webui, "OPTIMIZED_VOICE_PATH", optimized_path)
    monkeypatch.setattr(webui, "VOICE_CONDITIONING_PATH", conditioning_path)
    def write_preview(_source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF-preview")
        return 5.0

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)
    monkeypatch.setattr(
        webui,
        "_build_voice_conditioning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("upload must not build heavy conditioning")),
    )

    profile = webui.persist_voice(str(source))
    config = webui.read_user_config()

    assert "uploaded" in profile
    assert "将在首次使用时建立" in profile
    assert Path(config["reference_audio"]).is_file()
    assert str(voice_dir / "library") in config["reference_audio"]
    assert config["reference_conditioning"] is None
    assert config["voice_name"] == "uploaded"
    assert config["voice_library_id"]
    restored = webui.load_saved_state()
    assert restored[0] == config["reference_audio"]
    assert "uploaded" in restored[1]


def test_batch_voice_import_deduplicates_and_persists(tmp_path: Path, monkeypatch):
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.mp3"
    duplicate = tmp_path / "duplicate.wav"
    first.write_bytes(b"voice-one")
    second.write_bytes(b"voice-two")
    duplicate.write_bytes(first.read_bytes())

    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    def write_preview(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preview-" + source.read_bytes())
        return 4.5

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)
    selector, cleared, summary, status = webui.import_voice_files(
        [str(first), str(second), str(duplicate)],
        progress=lambda *_args, **_kwargs: None,
    )

    assert cleared is None
    assert "新增 2 个" in status
    assert "已存在 1 个" in status
    assert "已保存 2 个音色 · 常用 0/10" in summary
    assert len(webui.list_voice_library()) == 2
    assert len(selector.choices) == 2
    assert all("默认音色" not in label for label, _value in selector.choices)


def test_batch_save_button_accepts_single_uploader_and_empty_click_is_friendly(
    tmp_path: Path, monkeypatch
):
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    single = tmp_path / "single.wav"
    single.write_bytes(b"single-voice")
    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    def write_preview(_source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preview")
        return 5.0

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)

    _selector, _cleared, _summary, status = webui.import_voice_files(
        None,
        str(single),
        progress=lambda *_args, **_kwargs: None,
    )
    assert "单个音色已保存" in status
    assert len(webui.list_voice_library()) == 1

    _selector, _cleared, _summary, empty_status = webui.import_voice_files(
        None,
        None,
        progress=lambda *_args, **_kwargs: None,
    )
    assert "尚未选择音色" in empty_status


def test_select_saved_voice_updates_preview_and_generation_config(tmp_path: Path, monkeypatch):
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    source = tmp_path / "narrator.flac"
    source.write_bytes(b"narrator-voice")
    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    def write_preview(_source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preview")
        return 5.0

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)
    entry, added = webui._store_voice_in_library(source, display_name="新闻男声")
    assert added

    reference, profile, preview, nearby_preview, status = webui.select_voice_from_library(entry["id"])
    config = webui.read_user_config()

    assert reference == entry["source_path"]
    assert preview == entry["preview_path"] == nearby_preview
    assert "新闻男声" in profile
    assert "已选用音色" in status
    assert config["voice_library_id"] == entry["id"]
    assert config["reference_audio"] == entry["source_path"]


def test_manual_favorites_limit_remove_and_recoverable_delete(tmp_path: Path, monkeypatch):
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    def write_preview(_source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preview")
        return 5.0

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)
    entries = []
    for index in range(11):
        source = tmp_path / f"voice-{index}.wav"
        source.write_bytes(f"voice-{index}".encode())
        entry, _added = webui._store_voice_in_library(source)
        entries.append(entry)

    for entry in entries[:10]:
        assert "设为常用" in webui.set_voice_as_favorite(entry["id"])
    quick = webui._quick_voice_entries()
    assert len(quick) == 10
    assert quick[0]["id"] == entries[0]["id"]
    assert "已满 10 个" in webui.set_voice_as_favorite(entries[10]["id"])

    remove_status = webui.remove_voice_from_favorites(entries[0]["id"])
    assert "移出常用" in remove_status
    assert webui._load_voice_entry(entries[0]["id"]) is not None
    assert len(webui._quick_voice_entries()) == 9
    assert "设为常用" in webui.set_voice_as_favorite(entries[10]["id"])
    assert len(webui._quick_voice_entries()) == 10

    result = webui.delete_voice_from_library(entries[10]["id"])
    status = result[-1]
    assert "已删除音色" in status
    assert webui._load_voice_entry(entries[10]["id"]) is None
    assert any((voice_dir / "trash").iterdir())
    assert entries[10]["id"] not in [entry["id"] for entry in webui._quick_voice_entries()]


def test_follow_synthesis_uses_selector_voice_and_never_legacy_conditioning(
    tmp_path: Path, monkeypatch
):
    voice_dir = tmp_path / "voices"
    config_path = tmp_path / "settings.json"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)

    def write_preview(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"preview-" + source.read_bytes())
        return 5.0

    monkeypatch.setattr(webui, "_write_voice_preview", write_preview)
    old_source = tmp_path / "old.wav"
    selected_source = tmp_path / "selected.wav"
    old_source.write_bytes(b"old-voice")
    selected_source.write_bytes(b"selected-voice")
    old_entry, _ = webui._store_voice_in_library(old_source)
    selected_entry, _ = webui._store_voice_in_library(selected_source)
    legacy_conditioning = voice_dir / "current_voice_v2.npz"
    legacy_conditioning.parent.mkdir(parents=True, exist_ok=True)
    legacy_conditioning.write_bytes(b"legacy-conditioning")
    webui.update_user_config(
        voice_library_id=old_entry["id"],
        reference_audio=old_entry["source_path"],
        reference_conditioning=str(legacy_conditioning),
        reference_cache_version=webui.VOICE_CACHE_VERSION,
        voice_name="old",
    )

    expected_conditioning = Path(selected_entry["conditioning_path"])
    conditioning_calls = []

    def build_conditioning(source, **kwargs):
        conditioning_calls.append((source, kwargs))
        expected_conditioning.write_bytes(b"selected-conditioning")
        return str(expected_conditioning), 5.0

    monkeypatch.setattr(webui, "_build_voice_conditioning", build_conditioning)
    monkeypatch.setattr(
        webui,
        "analyze_audio_quality",
        lambda *_args: {"passed": True, "issues": [], "high_frequency_mean": 0.01},
    )

    class FakeModel:
        def __init__(self):
            class Tokenizer:
                @staticmethod
                def tokenize(text):
                    return list(text)

                @staticmethod
                def split_segments(tokens, max_tokens_per_segment):
                    return [tokens[index : index + max_tokens_per_segment] for index in range(0, len(tokens), max_tokens_per_segment)]

            self.tokenizer = Tokenizer()
            self.cache = {"audio_path": str(legacy_conditioning)}
            self.last_quality_fallback_used = False
            self.reference_used = None
            self.cache_at_generate = None

        def generate(self, **kwargs):
            self.reference_used = kwargs["reference_audio"]
            self.cache_at_generate = dict(self.cache)
            Path(kwargs["output_path"]).write_bytes(b"generated")
            return kwargs["output_path"]

    fake_model = FakeModel()
    monkeypatch.setattr(webui, "get_model", lambda: fake_model)

    webui.synthesize(
        "测试文字",
        selected_entry["id"],
        "跟随参考音频",
        0.6,
        1.0,
        42,
        230,
        50,
        130,
        0.8,
        16,
        1300,
        0.9,
        30,
        8.0,
        0.65,
        False,
        "wav",
        str(output_dir),
        progress=lambda *_args, **_kwargs: None,
    )

    assert conditioning_calls[0][0] == Path(selected_entry["source_path"])
    assert conditioning_calls[0][1]["conditioning_path"] == expected_conditioning
    assert fake_model.reference_used == str(expected_conditioning)
    assert fake_model.reference_used != str(legacy_conditioning)
    assert fake_model.cache_at_generate == {}
    config = webui.read_user_config()
    assert config["voice_library_id"] == selected_entry["id"]
    assert config["reference_audio"] == selected_entry["source_path"]
    assert config["reference_conditioning"] == str(expected_conditioning)


def test_emotion_backend_dispatch_preserves_follow_and_named_modes(monkeypatch):
    v25_model = object()
    emotion_model = object()
    monkeypatch.setattr(webui, "get_model", lambda: v25_model)
    monkeypatch.setattr(webui, "get_legacy_emotion_model", lambda: emotion_model)

    assert webui._resolve_emotion_backend("跟随参考音频") == (
        v25_model,
        None,
        "conditioning_path",
        "IndexTTS 2.5 · 跟随参考音频",
    )
    assert webui._resolve_emotion_backend("高兴") == (
        emotion_model,
        "happy",
        "conditioning_v2_path",
        "IndexTTS 2.0 · 高兴情绪控制",
    )


def test_page_restore_keeps_conditioning_for_identical_temporary_copy(tmp_path: Path, monkeypatch):
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    saved_voice = voice_dir / "current_voice.wav"
    temporary_copy = tmp_path / "gradio" / "current_voice.wav"
    temporary_copy.parent.mkdir()
    saved_voice.write_bytes(b"RIFF-identical-restored-voice")
    temporary_copy.write_bytes(saved_voice.read_bytes())
    conditioning = voice_dir / "current_voice_v2.npz"
    conditioning.write_bytes(b"cached-conditioning")
    config_path = tmp_path / "settings.json"

    monkeypatch.setattr(webui, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(webui, "CONFIG_PATH", config_path)
    monkeypatch.setattr(webui, "VOICE_CONDITIONING_PATH", conditioning)
    webui.update_user_config(
        reference_audio=str(saved_voice),
        reference_conditioning=str(conditioning),
        reference_cache_version=webui.VOICE_CACHE_VERSION,
        optimized_duration=5.0,
        voice_name="my_voice.wav",
    )
    monkeypatch.setattr(
        webui,
        "_optimize_reference_audio",
        lambda _source: (_ for _ in ()).throw(AssertionError("restoration must not reprocess the saved voice")),
    )

    profile = webui.persist_voice(str(temporary_copy))
    config = webui.read_user_config()

    assert "加速缓存：已预计算" in profile
    assert config["reference_audio"] == str(saved_voice)
    assert config["reference_conditioning"] == str(conditioning)
    assert config["reference_cache_version"] == webui.VOICE_CACHE_VERSION


def test_pause_and_terminate_buttons_control_active_task(monkeypatch):
    monkeypatch.setattr(webui.time, "perf_counter", lambda: 200.0)
    webui._generation_active.set()
    webui._generation_paused.clear()
    webui._generation_system_paused.clear()
    webui._generation_cancelled.clear()
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="running",
            current=1,
            total=3,
            message="正在生成",
            started_at=100.0,
            paused_at=None,
            paused_seconds=0.0,
            finished_elapsed=None,
        )

    pause_label, pause_status = webui.toggle_generation_pause()
    assert pause_label == "继续转换"
    assert "已暂停" in pause_status
    assert webui._generation_paused.is_set()
    assert webui._generation_progress_state["state"] == "paused"

    reset_label, terminate_status = webui.terminate_generation()
    assert reset_label == "暂停转换"
    assert "正在终止" in terminate_status
    assert webui._generation_cancelled.is_set()
    assert not webui._generation_paused.is_set()
    assert webui._generation_progress_state["state"] == "cancelling"

    webui._generation_active.clear()
    webui._generation_cancelled.clear()


def test_system_sleep_pauses_and_wake_resumes_active_task(monkeypatch):
    clock = iter([200.0, 260.0])
    monkeypatch.setattr(webui.time, "perf_counter", lambda: next(clock))
    webui._generation_active.set()
    webui._generation_paused.clear()
    webui._generation_system_paused.clear()
    webui._generation_cancelled.clear()
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="running",
            current=2,
            total=5,
            message="正在生成",
            started_at=100.0,
            paused_at=None,
            paused_seconds=0.0,
            finished_elapsed=None,
        )

    webui._system_will_sleep()
    assert webui._generation_system_paused.is_set()
    assert webui._generation_progress_state["state"] == "paused"
    assert "自动暂停" in webui._generation_progress_state["message"]

    webui._system_did_wake()
    assert not webui._generation_system_paused.is_set()
    assert webui._generation_progress_state["state"] == "running"
    assert webui._generation_progress_state["paused_seconds"] == 60.0
    assert "自动继续" in webui._generation_progress_state["message"]

    webui._generation_active.clear()


def test_system_wake_does_not_override_manual_pause(monkeypatch):
    monkeypatch.setattr(webui.time, "perf_counter", lambda: 200.0)
    webui._generation_active.set()
    webui._generation_paused.set()
    webui._generation_system_paused.clear()
    webui._generation_cancelled.clear()
    with webui._generation_progress_lock:
        webui._generation_progress_state.update(
            state="paused",
            paused_at=150.0,
            paused_seconds=0.0,
        )

    webui._system_will_sleep()
    webui._system_did_wake()

    assert webui._generation_paused.is_set()
    assert not webui._generation_system_paused.is_set()
    assert webui._generation_progress_state["state"] == "paused"
    assert webui._generation_progress_state["paused_at"] == 150.0

    webui._generation_active.clear()
    webui._generation_paused.clear()


def test_pause_and_terminate_ui_events_bypass_generation_queue():
    demo = webui.build_ui()
    control_events = {
        dependency["api_name"]: dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("api_name")
        in {"toggle_generation_pause", "terminate_generation"}
    }

    assert set(control_events) == {
        "toggle_generation_pause",
        "terminate_generation",
    }
    assert all(event["queue"] is False for event in control_events.values())


def test_generate_event_uses_voice_selector_id_not_audio_component():
    demo = webui.build_ui()
    labels_by_id = {
        component["id"]: component.get("props", {}).get("label")
        for component in demo.config["components"]
    }
    generation_event = next(
        dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("api_name") == "synthesize_stream"
    )
    input_labels = [labels_by_id.get(component_id) for component_id in generation_event["inputs"]]

    assert "选择已保存音色" in input_labels
    assert "单个音色上传 / 麦克风录音（自动保存）" not in input_labels


def test_reference_audio_persists_only_after_upload_or_recording_finishes():
    demo = webui.build_ui()
    labels_by_id = {
        component["id"]: component.get("props", {}).get("label")
        for component in demo.config["components"]
    }
    persistence_events = [
        dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("api_name", "").startswith(
            "persist_uploaded_voice_with_library"
        )
    ]

    assert len(persistence_events) == 2
    assert {
        trigger
        for dependency in persistence_events
        for component_id, trigger in dependency["targets"]
        if labels_by_id.get(component_id)
        == "单个音色上传 / 麦克风录音（自动保存）"
    } == {"upload", "stop_recording"}


def test_empty_recording_transition_keeps_current_voice(monkeypatch, tmp_path):
    voice_path = tmp_path / "saved.wav"
    voice_path.write_bytes(b"voice")
    monkeypatch.setattr(
        webui,
        "read_user_config",
        lambda: {
            "reference_audio": str(voice_path),
            "reference_conditioning": None,
            "optimized_duration": 6.4,
            "voice_name": "保留音色",
            "voice_library_id": "voice-id",
        },
    )
    monkeypatch.setattr(webui, "_voice_library_choices", lambda: [("保留音色", "voice-id")])
    monkeypatch.setattr(webui, "resolve_saved_voice_preview", lambda: str(voice_path))
    monkeypatch.setattr(webui, "render_voice_library_summary", lambda: "音色库未改变")

    result = webui.persist_uploaded_voice_with_library(None)

    assert "保留音色" in result[0]
    assert result[1] == str(voice_path)
    assert result[3] == str(voice_path)
    assert result[4] == "音色库未改变"
    assert result[5] == "录音尚未完成；当前音色保持不变。"
