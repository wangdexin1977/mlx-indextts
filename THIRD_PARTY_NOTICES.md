# Third-party notices

## IndexTTS 2.0 MLX weights

The local IndexTTS 2.0 weights are from
<https://huggingface.co/mlx-community/IndexTTS-2-MLX>, a conversion of
`IndexTeam/IndexTTS-2`. The upstream IndexTTS license applies to the weights;
see the model repository for its full terms.

## CosyVoice 3

The selectable CosyVoice 3 backend uses the official
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` model weights and the macOS runtime
adaptation from `drmhse/tts-funaudio-cozyvoice3`. Both upstream repositories
publish Apache-2.0 licenses. The model weights are downloaded separately into
the local `models/` directory.
The local macOS runtime was cloned at commit
`a63acae32fc2154a5bfab4fb783f2b033e65d9e1`.

Sources: <https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512>,
<https://github.com/drmhse/tts-funaudio-cozyvoice3>.

## Fish Audio S2 Pro

Built with Fish Audio.

This model is licensed under the Fish Audio Research License, Copyright © 39
AI, INC. All Rights Reserved.

Research and non-commercial use are permitted free of charge. Commercial use
requires a separate written license from Fish Audio. The complete current
license is published with the upstream model:
<https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md>

## VoiceStudio v0.5.2

Source: https://github.com/debpalash/VoiceStudio/tree/v0.5.2
Commit: 38c2405fa17841e9b53cb43a8af7641969bea90b.
The native model module is loaded from the separately cloned VoiceStudio checkout;
it is not copied into this project. VoiceStudio is AGPL-3.0; its bundled
OmniVoice source retains its upstream Apache-2.0 notices. The official
k2-fsa/OmniVoice weights and bundled audio tokenizer retain their own license terms.
See https://huggingface.co/k2-fsa/OmniVoice and its audio_tokenizer/LICENSE.
