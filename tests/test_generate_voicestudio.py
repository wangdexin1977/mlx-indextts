from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mlx_indextts.generate_voicestudio import VoiceStudioTTS
from mlx_indextts.generate_fish_s2 import FishS2ProTTS
from mlx_indextts.generate_v2 import GenerationCancelled


def fake_adapter():
    adapter = VoiceStudioTTS.__new__(VoiceStudioTTS)
    adapter.last_reference_transcript = ''
    adapter.last_quality_fallback_used = False
    adapter.last_speed_optimization_used = False
    return adapter


def test_native_parameters_reach_worker_and_audio_is_saved(tmp_path):
    adapter = fake_adapter()
    calls = []

    def request(payload, cancel):
        calls.append(payload)
        sf.write(payload['output'], np.ones(2400) * 0.1, 24000)

    adapter._request = request
    target = tmp_path / 'native.wav'
    adapter.generate(text='本机测试。', reference_audio=None, output_path=target,
                     omnivoice_mode='design', instruct='female', num_steps=24,
                     speed=1.1, seed=7, language='chinese', guidance_scale=2.3)
    assert calls[0]['config']['num_step'] == 24
    assert calls[0]['config']['guidance_scale'] == 2.3
    assert calls[0]['seed'] == 7
    assert calls[0]['instruct'] == 'female'
    assert calls[0]['speed'] == 1.1
    assert sf.info(target).samplerate == 24000


def test_native_clone_keeps_expression_instruct(monkeypatch, tmp_path):
    adapter = fake_adapter()
    calls = []

    def request(payload, cancel):
        calls.append(payload)
        sf.write(payload['output'], np.ones(2400) * 0.1, 24000)

    adapter._request = request
    monkeypatch.setattr(
        FishS2ProTTS,
        '_prepare_reference',
        lambda *_args, **_kwargs: (np.ones(2400) * 0.1, '参考文本'),
    )
    target = tmp_path / 'clone-expression.wav'
    adapter.generate(text='平静地讲述。', reference_audio='reference.wav', output_path=target,
                     omnivoice_mode='clone', instruct='moderate pitch')

    assert calls[0]['reference']
    assert calls[0]['instruct'] == 'moderate pitch'
    assert target.exists()


def test_native_cancel_preserves_finished_piece(tmp_path):
    adapter = fake_adapter()
    stopped = [False]

    def request(payload, cancel):
        sf.write(payload['output'], np.ones(2400) * 0.1, 24000)

    adapter._request = request
    target = tmp_path / 'cancel.wav'
    with pytest.raises(GenerationCancelled):
        adapter.generate(text='一段语音。' * 20, reference_audio=None, output_path=target,
                         omnivoice_mode='auto', max_text_tokens_per_segment=30,
                         cancel_requested=lambda: stopped[0],
                         audio_chunk_callback=lambda *_: stopped.__setitem__(0, True))
    assert not target.exists()
    assert sf.info(tmp_path / 'cancel.partial.wav').frames == 2400
