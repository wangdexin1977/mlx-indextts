"""Regression tests for IndexTTS 2.5 long-form alignment safeguards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlx_indextts.generate_v25 import (
    SAMPLE_RATE,
    V25_LONG_FORM_DIFFUSION_STEPS,
    V25_SAFE_CJK_TEXT_TOKEN_LIMIT,
    V25_SAFE_LATIN_TEXT_TOKEN_LIMIT,
    IndexTTSv25,
)


def _adapter_with_runtime(runtime, monkeypatch) -> IndexTTSv25:
    adapter = object.__new__(IndexTTSv25)
    adapter.runtime = runtime
    adapter.cache = {}
    adapter.last_quality_fallback_used = False
    adapter.last_speed_optimization_used = False
    monkeypatch.setattr(adapter, "_speaker", lambda _reference: object())
    return adapter


def test_v25_split_text_uses_language_adaptive_safe_limits():
    pieces = IndexTTSv25.split_text("测试" * 65, max_tokens_per_segment=120)
    english = IndexTTSv25.split_text("word " * 40, max_tokens_per_segment=120)

    assert len(pieces) == 2
    assert max(map(len, pieces)) <= V25_SAFE_CJK_TEXT_TOKEN_LIMIT
    assert "".join(pieces) == "测试" * 65
    assert max(map(len, english)) <= V25_SAFE_LATIN_TEXT_TOKEN_LIMIT


def test_v25_duration_guard_discards_and_resplits_stretched_segment(
    monkeypatch, tmp_path: Path
):
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            # Reproduce the bug: a 60-character sentence becomes a distorted
            # 32-second utterance.  Its shorter retry pieces are normal clips.
            seconds = 32.0 if len(text) > 45 else 7.0
            return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16)

    runtime = FakeRuntime()
    adapter = _adapter_with_runtime(runtime, monkeypatch)
    output_path = tmp_path / "guarded.wav"

    audio = adapter.generate(
        text="测" * 60,
        reference_audio="speaker.npz",
        output_path=str(output_path),
        max_text_tokens_per_segment=120,
        interval_silence=100,
        seed=42,
    )

    assert [len(text) for text, _kwargs in runtime.calls] == [60, 45, 15]
    assert all(
        kwargs["max_text_tokens_per_segment"] == V25_SAFE_CJK_TEXT_TOKEN_LIMIT
        for _text, kwargs in runtime.calls
    )
    assert runtime.calls[0][1]["max_mel_tokens"] <= 900
    assert adapter.last_quality_fallback_used is True
    assert 14.0 < audio.size / SAMPLE_RATE < 14.2
    assert output_path.exists()


def test_v25_duration_guard_leaves_normal_segment_untouched(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.calls = 0

        def synthesize(self, _text, **_kwargs):
            self.calls += 1
            return np.zeros(SAMPLE_RATE * 3, dtype=np.int16)

    runtime = FakeRuntime()
    adapter = _adapter_with_runtime(runtime, monkeypatch)

    audio = adapter.generate(
        text="这是一句正常的测试文本。",
        reference_audio="speaker.npz",
        seed=7,
    )

    assert runtime.calls == 1
    assert audio.size == SAMPLE_RATE * 3
    assert adapter.last_quality_fallback_used is False


def test_v25_long_form_caps_diffusion_steps_but_short_text_keeps_quality(
    monkeypatch
):
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return np.zeros(SAMPLE_RATE * 3, dtype=np.int16)

    runtime = FakeRuntime()
    adapter = _adapter_with_runtime(runtime, monkeypatch)
    adapter.generate(
        text="测" * (V25_SAFE_CJK_TEXT_TOKEN_LIMIT * 4),
        reference_audio="speaker.npz",
        diffusion_steps=25,
    )

    assert len(runtime.calls) == 4
    assert {
        kwargs["n_timesteps"] for _text, kwargs in runtime.calls
    } == {V25_LONG_FORM_DIFFUSION_STEPS}
    assert adapter.last_speed_optimization_used is True

    runtime.calls.clear()
    adapter.generate(
        text="短文本测试。",
        reference_audio="speaker.npz",
        diffusion_steps=25,
    )

    assert runtime.calls[0][1]["n_timesteps"] == 25
    assert adapter.last_speed_optimization_used is False
