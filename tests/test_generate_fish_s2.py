from types import SimpleNamespace

import numpy as np
import pytest

from mlx_indextts.generate_fish_s2 import FishS2ProTTS


def _adapter_with_runtime(runtime) -> FishS2ProTTS:
    adapter = FishS2ProTTS.__new__(FishS2ProTTS)
    adapter.runtime = runtime
    adapter.sample_rate = 44_100
    adapter.cache = {}
    adapter.last_reference_transcript = ""
    adapter.last_quality_fallback_used = False
    adapter.last_speed_optimization_used = False
    return adapter


def test_fish_generation_preserves_long_form_context_and_retries_token_cap(tmp_path):
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

    assert [call["max_tokens"] for call in runtime.calls] == [1024, 2048]
    assert all(call["text"] == text for call in runtime.calls)
    assert target.is_file()


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
