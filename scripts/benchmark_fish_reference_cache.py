"""Compare identical Fish generation with and without encoded-reference reuse."""
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_indextts.generate_fish_s2 import FishS2ProTTS
from mlx_indextts.audio_gain import boost_wav

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/webui/fish_s2_gain_validation'
out.mkdir(parents=True, exist_ok=True)
a = FishS2ProTTS(str(root / 'models/fish-audio-s2-pro-8bit'))
ref = root / 'outputs/webui/voices/library/b4185ce911d2131e'
meta = json.loads((ref / 'metadata.json').read_text())
audio, transcript = a._prepare_reference(str(ref / 'source.wav'), meta['fish_ref_text'], 15)
args = dict(text='今天的阳光很好，我们一起出发。', ref_audio=audio, ref_text=transcript,
            temperature=0.7, top_p=0.7, top_k=30, max_tokens=1024, stream=False, verbose=False)
rows = []
waves = {}
# Alternate methods after a cold warm-up; each sample uses identical random seed.
for label in ['warmup', 'baseline_1', 'cached_1', 'cached_2', 'baseline_2']:
    mx.random.seed(42)
    start = time.perf_counter()
    results = a._generate_results(**args) if label.startswith('cached') else list(a.runtime.generate(**args))
    wave = np.asarray(results[0].audio).copy()
    elapsed = time.perf_counter() - start
    waves[label] = wave
    row = dict(label=label, seconds=elapsed, samples=wave.size,
               tokens=int(results[0].token_count), peak_memory=mx.get_peak_memory())
    rows.append(row)
    print(json.dumps(row), flush=True)
    sf.write(out / f'{label}.wav', wave, a.sample_rate, subtype='PCM_16')
    del results
    mx.clear_cache()
comparison = {label: dict(equal=bool(np.array_equal(waves['baseline_1'], wave)),
                         max_error=float(np.max(np.abs(waves['baseline_1'] - wave))) if wave.shape == waves['baseline_1'].shape else None)
              for label,wave in waves.items()}
boosted = out / 'boosted_6db.wav'
sf.write(boosted, waves['cached_2'], a.sample_rate, subtype='PCM_16')
gain = boost_wav(boosted)
report = dict(runs=rows, comparison=comparison, applied_gain_db=gain)
(out / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report), flush=True)
