import shutil

import numpy as np
import pytest
import soundfile as sf

from mlx_indextts.audio_gain import boost_wav

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')


@pytest.mark.parametrize('amplitude', [0.05, 0.8, 0.0])
def test_gain_preserves_waveform_and_limits_peak(tmp_path, amplitude):
    path = tmp_path / 'audio.wav'
    wave = amplitude * np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100)
    sf.write(path, wave, 44100, subtype='PCM_16')
    before, rate = sf.read(path)
    gain = boost_wav(path)
    after, after_rate = sf.read(path)
    assert after_rate == rate
    assert after.shape == before.shape
    np.testing.assert_allclose(after, before * 10 ** (gain / 20), atol=1 / 32768)
    assert np.max(np.abs(after)) <= 10 ** (-1 / 20)
    assert gain == 6 if amplitude == 0.05 else gain < 6
    assert not path.with_name('audio.gain.wav').exists()
