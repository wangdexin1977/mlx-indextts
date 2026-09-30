import sys
import types

import numpy as np

from mlx_indextts.generate_omnivoice import OmniVoiceTTS


def test_mlx_clone_keeps_expression_instruct(monkeypatch, tmp_path):
    calls = []

    class Runtime:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return [types.SimpleNamespace(audio=np.ones(2400, dtype=np.float32) * 0.1)]

    fake_core = types.ModuleType('mlx.core')
    fake_core.random = types.SimpleNamespace(seed=lambda _seed: None)
    fake_mlx = types.ModuleType('mlx')
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, 'mlx', fake_mlx)
    monkeypatch.setitem(sys.modules, 'mlx.core', fake_core)

    adapter = OmniVoiceTTS.__new__(OmniVoiceTTS)
    adapter.runtime = Runtime()
    adapter.last_reference_transcript = ''
    adapter._prepare_reference_prompt = lambda *_args: ('reference-tokens', '参考文本')
    target = tmp_path / 'clone-expression.wav'

    adapter.generate(
        text='平静地讲述。',
        reference_audio='reference.wav',
        output_path=target,
        omnivoice_mode='clone',
        instruct='moderate pitch',
    )

    assert calls[0]['ref_tokens'] == 'reference-tokens'
    assert calls[0]['instruct'] == 'moderate pitch'
    assert target.exists()


def test_reference_excerpt_enforces_ten_second_quality_limit():
    sample_rate = 24_000
    audio = np.ones(sample_rate * 15, dtype=np.float32) * 0.1

    excerpt, shortened = OmniVoiceTTS._reference_excerpt(audio, sample_rate, 15.0)

    assert shortened is True
    assert sample_rate * 3 <= excerpt.size <= sample_rate * 10
    assert abs(float(np.mean(excerpt))) < 1e-4


def test_reference_excerpt_keeps_short_prompt():
    sample_rate = 24_000
    audio = np.sin(np.linspace(0, 100, sample_rate * 5, dtype=np.float32)) * 0.1

    excerpt, shortened = OmniVoiceTTS._reference_excerpt(audio, sample_rate, 10.0)

    assert shortened is False
    assert excerpt.size <= audio.size
    assert excerpt.size >= audio.size - int(sample_rate * 0.2)


def test_postprocess_audio_removes_dc_and_fades_edges():
    audio = np.ones(24_000, dtype=np.float32) * 0.2
    cleaned = OmniVoiceTTS._postprocess_audio(audio)

    assert np.isfinite(cleaned).all()
    assert abs(float(np.mean(cleaned))) < 1e-5
    assert cleaned[0] == 0.0
    assert cleaned[-1] == 0.0
