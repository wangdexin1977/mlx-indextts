# MLX-IndexTTS

IndexTTS for Apple Silicon using MLX. Zero-shot text-to-speech with voice cloning capabilities.

Current WebUI release: **v0.5.9**. See [CHANGELOG.md](CHANGELOG.md) for the
version history; every release must update both that file and the in-app
"About / Version" panel.

## WebUI model switching

The WebUI can switch between IndexTTS 2.5, IndexTTS 2.0, OmniVoice, Fish
Audio S2 Pro, CosyVoice 3 and VoiceStudio OmniVoice. OmniVoice supports voice cloning, text-described voice design
and automatic voices. Fish S2 Pro supports cloning, automatic/multi-speaker
generation, inline expression tags and dedicated sampling controls. Models are
loaded only when selected.

IndexTTS 2.0 requires all three converted weights (`gpt.safetensors`,
`s2mel.safetensors`, and `bigvgan.safetensors`) in `models/mlx-IndexTTS-2`.
The WebUI now stops with a clear error if any is missing. Its default flow
setting is 16 steps, measured locally on a 667-character Chinese passage with
the natural/calm emotion setting; the other models keep their own profiles.

Fish S2 Pro uses the 8-bit MLX conversion and is governed by the Fish Audio
Research License: research and non-commercial use are free; commercial use
requires a separate Fish Audio license.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and the
upstream license link.

CosyVoice 3 uses the official `Fun-CosyVoice3-0.5B-2512` weights in
`models/Fun-CosyVoice3-0.5B-2512` and the Apple Silicon runtime in
`vendor/cosyvoice3-macos/.venv`. Clone the runtime separately at the pinned
commit and apply the two small local compatibility changes:

```bash
git clone https://github.com/drmhse/tts-funaudio-cozyvoice3.git vendor/cosyvoice3-macos
git -C vendor/cosyvoice3-macos checkout a63acae32fc2154a5bfab4fb783f2b033e65d9e1
git -C vendor/cosyvoice3-macos apply --unidiff-zero ../../patches/cosyvoice3-macos-local.patch
```

Install its isolated Python environment using
[`requirements-cosyvoice3-macos.txt`](requirements-cosyvoice3-macos.txt); model
weights and runtime environment are not stored in this repository. It outputs
24 kHz audio. Select an existing
voice, then choose **CosyVoice 3** in the same model dropdown. Its reference
transcript is attached to each saved voice; when left blank, the exact 3–10
second reference excerpt is transcribed locally with Qwen3-ASR. The dedicated
controls default to FP16 LLM precision and 10 flow steps. On the local Apple
Silicon test, FP16 preserved the reference speaker similarity and Chinese
transcript while cutting model load and inference time; the flow and vocoder
still run at FP32. FP32 remains available for comparison. Changing precision or flow steps
reloads this backend on the next request. The worker stays loaded across voice
changes, and the final status reports its synthesis RTF.

```bash
uv add mlx-audio==0.4.6
uv run hf download mlx-community/OmniVoice-bfloat16 \
  --local-dir models/OmniVoice-bfloat16
uv run hf download mlx-community/fish-audio-s2-pro-8bit \
  --local-dir models/fish-audio-s2-pro-8bit
uv run hf download mlx-community/Qwen3-ASR-0.6B-8bit \
  --local-dir models/Qwen3-ASR-0.6B-8bit
```

For stable cloning, OmniVoice must align the reference audio with its exact
transcript. Release v0.5.2 enforces the upstream recommended 3–10 second
reference window. Longer library audio is cut near a low-energy boundary and
that exact excerpt is transcribed locally with Qwen3-ASR before it is encoded.
The output keeps OmniVoice's native 24 kHz resolution and only receives
transparent edge/DC cleanup; it is not artificially upsampled.

OmniVoice model weights are licensed CC-BY-NC (non-commercial use). This is
the k2-fsa OmniVoice project, not Xiaomi's official MiMo TTS.

OmniVoice exposes follow-reference, calm, happy, sad, energetic, serious,
whisper, and custom expression choices in the WebUI and local synthesis skill.
The base model has no IndexTTS-style emotion vector: whisper is native, while
the named emotional choices are approximation presets using supported pitch,
speed, and sampling controls.

## Features

- Run IndexTTS 1.5/2.0 natively on Apple Silicon
- RTF ~0.5 (2x faster than real-time on M2 Max)
- Voice cloning from reference audio
- **v2.0**: Emotion control (8 emotions)
- Auto-detect model version (1.5/2.0)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/user/mlx-indextts.git
cd mlx-indextts

# Basic install (generation only)
uv sync

# With model conversion support (requires torch)
uv sync --extra convert
```

## Quick Start

### 1. Convert Model (auto-detects version)

```bash
# Convert IndexTTS 1.5
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    -o models/mlx-indexTTS-1.5

# Convert IndexTTS 2.0
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0
```

### 2. Generate Speech (auto-detects version)

```bash
# v1.5
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0 with emotion control
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -t "今天真是太开心了！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.6
```

### 3. Pre-compute Speaker (Faster Inference)

Pre-compute speaker conditioning to skip audio preprocessing on subsequent generations.

```bash
# v1.5
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -o speaker_v15.npz

# v2.0
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -o speaker_v20.npz

# Use pre-computed speaker (much faster loading)
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r speaker_v20.npz \
    -t "你好，世界！" \
    -o output.wav
```

**Note**: v1.5 and v2.0 speaker files are incompatible - each version requires its own .npz file.

## Python API

```python
# v1.5
from mlx_indextts.generate import IndexTTS

tts = IndexTTS.load_model("models/mlx-indexTTS-1.5")
audio = tts.generate(text="你好", ref_audio="reference.wav")
tts.save_audio(audio, "output.wav")

# v2.0
from mlx_indextts.generate_v2 import IndexTTSv2

tts = IndexTTSv2("models/mlx-indexTTS-2.0")
audio = tts.generate(
    text="你好",
    reference_audio="reference.wav",
    output_path="output.wav",
    emotion="happy",
    emo_alpha=0.6,
)
```

## CLI Options

```
mlx-indextts generate [OPTIONS]

Required:
  -m, --model        Model directory
  -r, --ref-audio    Reference audio (.wav or .npz)
  -t, --text         Text to synthesize
  -o, --output       Output file

Common options:
  --max-tokens       Max mel tokens (default: 800 for v1.5, 1500 for v2.0)
  --temperature      Sampling temperature (default: 1.0 for v1.5, 0.8 for v2.0)
  --seed, -s         Random seed for reproducibility
  -v, --verbose      Verbose output
  -p, --play         Play audio after generation
  --quantize, -q     Runtime quantization: 4, 8, or fp32

v2.0 only:
  --emotion          Emotion: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm
  --emo-alpha        Emotion intensity 0.0-1.0 (default: 0.6, recommend ≤ 0.8)
  --diffusion-steps  Diffusion steps (default: 25)
  --cfg-rate         CFG rate (default: 0.7)
```

## Version Comparison

| Feature | v1.5 | v2.0 |
|---------|------|------|
| Sample rate | 24000 Hz | 22050 Hz |
| Max tokens | 800 | 1815 |
| Default temperature | 1.0 | 0.8 |
| Emotion control | ❌ | ✅ 8 emotions |
| S2Mel (CFM) | ❌ | ✅ |
| BigVGAN | Custom | nvidia pretrained |
| Runtime quantization | ✅ | ✅ |
| Speaker pre-compute | ✅ | ✅ |

## Supported Emotions (v2.0)

| English | 中文 |
|---------|------|
| happy | 高兴 |
| angry | 愤怒 |
| sad | 悲伤 |
| afraid | 恐惧 |
| disgusted | 反感 |
| melancholic | 低落 |
| surprised | 惊讶 |
| calm | 自然 |

Mixed emotions: `--emotion "happy:0.6,sad:0.4"`

## Performance

| Metric | v1.5 | v2.0 |
|--------|------|------|
| RTF (M2 Max) | ~0.5 | ~1.3 |
| Load time (.wav) | ~0.3s | ~9s |
| Load time (.npz) | ~0.3s | ~1.5s |

## License

MIT License

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Original PyTorch implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework

### VoiceStudio native engine

The model selector also supports **VoiceStudio · OmniVoice 原生**, using the
OmniVoice model code from VoiceStudio v0.5.2 in a separate local Python process.
This is the native PyTorch/MPS engine, distinct from the existing OmniVoice MLX
entry. The full VoiceStudio desktop frontend and its other engines are not
started by this integration.

Keep the VoiceStudio checkout beside this repository (`../VoiceStudio`, tag
`v0.5.2`) and download `k2-fsa/OmniVoice` to `models/VoiceStudio-OmniVoice`, including
`audio_tokenizer/`. The runtime uses the existing PyTorch/torchaudio stack,
Transformers 5.5+ and `accelerate`. All inference runs offline. Apple Silicon
uses MPS for the speech model and CPU for its upstream audio tokenizer.

Recommended native defaults: cloning, Chinese, 32 steps, guidance 2.0, class
temperature 0, position temperature 5, layer penalty 5, T-shift 0.1, speed 1,
seed 42, reference duration 10 seconds. Supply a transcript matching that exact
reference duration, or leave it blank for local ASR. Auto and design modes are
also supported. Reference preprocessing and output processing use upstream
OmniVoice defaults; this does not apply VoiceStudio's separate application
mastering chain.

Each model now retains a separate parameter profile in `user_settings.json`.
Switching restores that model's values; “恢复推荐设置” resets only that model.
IndexTTS-only controls are hidden for the other engines. Existing saved values
are migrated on first selection. Switching away from the native engine releases
its worker, and cancellation can stop it during a segment. Completed segments
are preserved when generation is cancelled.

Run `.venv/bin/python scripts/validate_voicestudio.py` for a local voice-clone
smoke test using the existing test voice. Restart the WebUI to load code changes.
