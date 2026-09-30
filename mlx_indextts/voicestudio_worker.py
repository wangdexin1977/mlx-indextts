"""Private local worker for the pinned VoiceStudio native OmniVoice engine."""
import contextlib
import json
from pathlib import Path
import sys
import traceback


def main():
    source, model_dir = sys.argv[1:3]
    sys.path.insert(0, source)
    protocol = sys.stdout
    model = None
    prompt_cache = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                import numpy as np
                import soundfile as sf
                import torch
                from omnivoice import OmniVoice, OmniVoiceGenerationConfig

                if model is None:
                    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                    model = OmniVoice.from_pretrained(
                        model_dir, device_map=device,
                        dtype=torch.float16 if device == 'mps' else torch.float32,
                        load_asr=False, local_files_only=True,
                    ).eval()
                torch.manual_seed(request['seed'])
                prompt = None
                if request.get('reference'):
                    ref = Path(request['reference'])
                    key = (str(ref), ref.stat().st_mtime_ns, request['ref_text'])
                    if prompt_cache is None or prompt_cache[0] != key:
                        wave, rate = sf.read(ref, dtype='float32', always_2d=True)
                        prompt = model.create_voice_clone_prompt(
                            (torch.from_numpy(wave.mean(axis=1).copy()), rate),
                            ref_text=request['ref_text'], preprocess_prompt=True,
                        )
                        prompt_cache = (key, prompt)
                    prompt = prompt_cache[1]
                with torch.inference_mode():
                    result = model.generate(
                        text=request['text'], language=request['language'],
                        voice_clone_prompt=prompt, instruct=request.get('instruct'),
                        duration=request.get('duration'), speed=request['speed'],
                        generation_config=OmniVoiceGenerationConfig(**request['config']),
                    )
                wave = result[0].detach().float().cpu().numpy().squeeze()
                if wave.ndim != 1 or not np.isfinite(wave).all():
                    raise RuntimeError('Native engine returned invalid audio')
                sf.write(request['output'], wave, model.sampling_rate, subtype='FLOAT')
                response = {'ok': True, 'sample_rate': model.sampling_rate}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        protocol.write(json.dumps(response) + '\n')
        protocol.flush()


if __name__ == '__main__':
    main()
