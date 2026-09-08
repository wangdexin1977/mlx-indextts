"""IndexTTS 2.5 adapter for the local WebUI.

Keeps the WebUI's established generation contract: reusable speaker caches,
segment progress, cooperative pause/cancel, partial checkpoints and normalized
floating-point audio for the existing quality checks.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from index_tts_2_5_mlx import IndexTTS, SpeakerContext

from mlx_indextts.generate_v2 import GenerationCancelled


SAMPLE_RATE = 22_050
SPEAKER_CACHE_VERSION = 2.5


class _TokenizerAdapter:
    """Expose the tokenizer methods used by the WebUI batch planner."""

    def __init__(self, owner: "IndexTTSv25") -> None:
        self.owner = owner

    def tokenize(self, text: str) -> str:
        return text

    def split_segments(self, text: str, max_tokens_per_segment: int = 120) -> list[str]:
        return self.owner.split_text(text, max_tokens_per_segment)


class IndexTTSv25:
    """Compatibility wrapper around the native IndexTTS-2.5 MLX runtime."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = str(model_dir)
        self.runtime = IndexTTS(model_dir=self.model_dir)
        self.tokenizer = _TokenizerAdapter(self)
        self.cache: dict[str, SpeakerContext] = {}
        self.last_quality_fallback_used = False

    @staticmethod
    def split_text(text: str, max_tokens_per_segment: int = 120) -> list[str]:
        """Split on natural pauses, with a conservative hard limit."""
        limit = max(24, int(max_tokens_per_segment))
        units = re.split(r"(?<=[。！？；，.!?;,\n])", str(text))
        pieces: list[str] = []
        current = ""
        for unit in units:
            if not unit:
                continue
            while len(unit) > limit:
                if current.strip():
                    pieces.append(current.strip())
                    current = ""
                pieces.append(unit[:limit].strip())
                unit = unit[limit:]
            if current and len(current) + len(unit) > limit:
                pieces.append(current.strip())
                current = unit
            else:
                current += unit
        if current.strip():
            pieces.append(current.strip())
        return [piece for piece in pieces if piece]

    @staticmethod
    def _save_context(context: SpeakerContext, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=np.asarray([SPEAKER_CACHE_VERSION], dtype=np.float32),
            spk_cond_emb=context.spk_cond_emb,
            emovec=context.emovec,
            style=context.style,
            conds=context.conds,
            prompt_condition=context.prompt_condition,
            ref_mel=context.ref_mel,
        )

    @staticmethod
    def _load_context(input_path: str) -> SpeakerContext:
        with np.load(input_path) as data:
            version = float(np.asarray(data["version"]).reshape(-1)[0])
            if version < SPEAKER_CACHE_VERSION:
                raise ValueError("音色缓存不是 IndexTTS-2.5 格式")
            return SpeakerContext(
                spk_cond_emb=np.asarray(data["spk_cond_emb"]),
                emovec=np.asarray(data["emovec"]),
                style=np.asarray(data["style"]),
                conds=np.asarray(data["conds"]),
                prompt_condition=np.asarray(data["prompt_condition"]),
                ref_mel=np.asarray(data["ref_mel"]),
            )

    def save_speaker(self, reference_audio: str, output_path: str) -> None:
        context = self.runtime.build_speaker(reference_audio)
        self._save_context(context, output_path)
        self.cache[str(Path(output_path).resolve())] = context

    def _speaker(self, reference_audio: str) -> SpeakerContext:
        key = str(Path(reference_audio).resolve())
        if key in self.cache:
            return self.cache[key]
        if Path(reference_audio).suffix.lower() == ".npz":
            context = self._load_context(reference_audio)
        else:
            context = self.runtime.build_speaker(reference_audio)
        self.cache[key] = context
        return context

    @staticmethod
    def _join_audio(
        segments: list[np.ndarray], interval_silence: int, segment_overlap_ms: int
    ) -> np.ndarray:
        if len(segments) == 1:
            return segments[0]
        if interval_silence > 0:
            silence = np.zeros(int(SAMPLE_RATE * interval_silence / 1000), dtype=np.float32)
            joined: list[np.ndarray] = [segments[0]]
            for segment in segments[1:]:
                joined.extend((silence, segment))
            return np.concatenate(joined)
        overlap = int(SAMPLE_RATE * max(0, segment_overlap_ms) / 1000)
        output = segments[0]
        for segment in segments[1:]:
            width = min(overlap, len(output), len(segment))
            if width <= 0:
                output = np.concatenate((output, segment))
                continue
            fade_in = np.linspace(0.0, 1.0, width, dtype=np.float32)
            mixed = output[-width:] * (1.0 - fade_in) + segment[:width] * fade_in
            output = np.concatenate((output[:-width], mixed, segment[width:]))
        return output

    def generate(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None,
        max_mel_tokens: int = 1500,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        diffusion_steps: int = 25,
        cfg_rate: float = 0.7,
        emotion=None,
        emo_alpha: float = 0.6,
        seed: Optional[int] = None,
        verbose: bool = False,
        segment_overlap_ms: int = 50,
        speed: float = 1.0,
        fast_vocoder: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        audio_chunk_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
        pause_requested: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        del emotion, emo_alpha, fast_vocoder
        pieces = self.split_text(text, max_text_tokens_per_segment)
        if not pieces:
            raise ValueError("文本为空")
        speaker = self._speaker(reference_audio)
        generated: list[np.ndarray] = []
        partial_path = Path(output_path).with_suffix(".partial.wav") if output_path else None

        for index, piece in enumerate(pieces, start=1):
            while pause_requested and pause_requested():
                if cancel_requested and cancel_requested():
                    raise GenerationCancelled("用户已终止当前任务")
                time.sleep(0.1)
            if cancel_requested and cancel_requested():
                if partial_path and generated:
                    sf.write(partial_path, self._join_audio(generated, interval_silence, 0), SAMPLE_RATE)
                raise GenerationCancelled("用户已终止当前任务")
            if progress_callback:
                progress_callback(index - 1, len(pieces), f"正在生成片段 {index}/{len(pieces)}")
            pcm = self.runtime.synthesize(
                piece,
                lang="zh",
                spk=speaker,
                seed=None if seed is None else int(seed) + index - 1,
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                max_mel_tokens=int(max_mel_tokens),
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                interval_silence=0,
                duration_factor=1.0 / max(0.5, float(speed)),
                n_timesteps=int(diffusion_steps),
                cfg_rate=float(cfg_rate),
            )
            audio = np.asarray(pcm, dtype=np.float32) / 32768.0
            if audio.size == 0:
                raise RuntimeError(f"第 {index} 个片段未生成有效音频")
            generated.append(audio)
            if audio_chunk_callback:
                audio_chunk_callback(audio, SAMPLE_RATE)
            if partial_path and (index % 3 == 0 or index == len(pieces)):
                sf.write(partial_path, self._join_audio(generated, interval_silence, 0), SAMPLE_RATE)
            if progress_callback:
                progress_callback(index, len(pieces), f"已完成片段 {index}/{len(pieces)}")

        audio = self._join_audio(generated, interval_silence, segment_overlap_ms)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, SAMPLE_RATE)
            if partial_path and partial_path.exists():
                partial_path.unlink()
        if verbose:
            seconds = len(audio) / SAMPLE_RATE
            print(f"IndexTTS-2.5 generated {seconds:.2f}s from {len(pieces)} segment(s)")
        return audio
