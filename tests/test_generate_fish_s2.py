from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from mlx_indextts.generate_fish_s2 import FishS2ProTTS
from mlx_indextts.generate_v2 import GenerationCancelled


def _adapter_with_runtime(runtime) -> FishS2ProTTS:
    adapter = FishS2ProTTS.__new__(FishS2ProTTS)
    adapter.runtime = runtime
    adapter.sample_rate = 44_100
    adapter.cache = {}
    adapter.last_reference_transcript = ""
    adapter.last_quality_fallback_used = False
    adapter.last_speed_optimization_used = False
    return adapter


def test_fish_generation_uses_bounded_segments_and_retries_only_capped_piece(tmp_path):
    class Runtime:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            token_limit = kwargs["max_tokens"]
            token_count = token_limit if len(self.calls) == 1 else 200
            return [
                SimpleNamespace(
                    audio=np.linspace(-0.1, 0.1, 4_410, dtype=np.float32),
                    token_count=token_count,
                )
            ]

    runtime = Runtime()
    adapter = _adapter_with_runtime(runtime)
    text = "这是需要保持上下文连续的长文。" * 20
    target = tmp_path / "fish.wav"

    adapter.generate(
        text=text,
        reference_audio=None,
        output_path=str(target),
        fish_mode="auto",
        max_tokens=256,
        chunk_length=300,
    )

    assert [call["max_tokens"] for call in runtime.calls[:2]] == [1024, 2048]
    assert runtime.calls[0]["text"] == runtime.calls[1]["text"]
    successful_calls = runtime.calls[1:]
    assert "".join(call["text"] for call in successful_calls) == text
    assert all(len(call["text"]) <= 60 for call in runtime.calls)
    assert target.is_file()
    assert not (tmp_path / "fish.partial.wav").exists()


def test_fish_cancel_preserves_each_completed_segment(tmp_path):
    class Runtime:
        @staticmethod
        def generate(**_kwargs):
            return [
                SimpleNamespace(
                    audio=np.linspace(-0.1, 0.1, 4_410, dtype=np.float32),
                    token_count=200,
                )
            ]

    adapter = _adapter_with_runtime(Runtime())
    target = tmp_path / "cancelled.wav"
    cancelled = [False]

    def progress(current, _total, _message):
        if current == 1:
            cancelled[0] = True

    with pytest.raises(GenerationCancelled):
        adapter.generate(
            text="第一段应该完成并保存。" * 6,
            reference_audio=None,
            output_path=str(target),
            fish_mode="auto",
            progress_callback=progress,
            cancel_requested=lambda: cancelled[0],
        )

    partial = tmp_path / "cancelled.partial.wav"
    assert partial.is_file()
    assert sf.info(partial).duration > 0
    assert not target.exists()


def test_fish_generation_rejects_audio_still_capped_at_safety_limit(tmp_path):
    class Runtime:
        @staticmethod
        def generate(**kwargs):
            return [
                SimpleNamespace(
                    audio=np.zeros(4_410, dtype=np.float32),
                    token_count=kwargs["max_tokens"],
                )
            ]

    adapter = _adapter_with_runtime(Runtime())
    target = tmp_path / "truncated.wav"

    with pytest.raises(RuntimeError, match="4096 Token 安全上限"):
        adapter.generate(
            text="没有自然结束标记的超长内容" * 50,
            reference_audio=None,
            output_path=str(target),
            fish_mode="auto",
            max_tokens=4096,
        )

    assert not target.exists()
