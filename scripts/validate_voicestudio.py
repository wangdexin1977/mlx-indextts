"""Generate one real native VoiceStudio clone and save measurable validation."""
import json
from pathlib import Path
import time

import soundfile as sf

from mlx_indextts.generate_voicestudio import VoiceStudioTTS

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/webui/voicestudio_validation'
out.mkdir(parents=True, exist_ok=True)
voice = root / 'outputs/webui/voices/library/b4185ce911d2131e'
meta = json.loads((voice / 'metadata.json').read_text())
adapter = VoiceStudioTTS(str(root.parent / 'VoiceStudio'), str(root / 'models/VoiceStudio-OmniVoice'),
                         str(root / 'models/Qwen3-ASR-0.6B-8bit'))
try:
    start = time.perf_counter()
    audio = adapter.generate(
        text='你好，这是本机 VoiceStudio 的声音克隆测试。', reference_audio=str(voice / 'source.wav'),
        ref_text=meta['fish_ref_text'], ref_audio_max_duration_s=15,
        output_path=str(out / 'VoiceStudio原生克隆测试.wav'),
        num_steps=32, guidance_scale=2.0, seed=42,
        progress_callback=lambda a,b,msg: print(msg, flush=True),
    )
    report = dict(seconds=time.perf_counter()-start, duration=len(audio)/24000,
                  sample_rate=24000, peak=float(abs(audio).max()))
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)
finally:
    adapter.close()
