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
# IndexTTS 2.5 can lose monotonic text/audio alignment when one autoregressive
# segment grows beyond roughly 25 seconds.  The upstream default of 120 text
# tokens sits on that boundary for Chinese and is too large for Latin text.
V25_SAFE_TEXT_TOKEN_LIMIT = 60
V25_MIN_TEXT_TOKEN_LIMIT = 12
V25_MAX_SEGMENT_SECONDS = 25.0
V25_DURATION_GUARD_RETRIES = 2


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
        """Split on natural pauses and keep every 2.5 segment in its safe range."""
        limit = max(
            V25_MIN_TEXT_TOKEN_LIMIT,
            min(V25_SAFE_TEXT_TOKEN_LIMIT, int(max_tokens_per_segment)),
        )
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
    def _reasonable_segment_seconds(text: str, speed: float = 1.0) -> float:
        """Return a conservative duration ceiling for one generated segment.

        This is deliberately a guard, not duration control.  It only catches
        the characteristic 2.5 alignment collapse where a short sentence is
        expanded into a very long, distorted utterance.
        """
        cjk_or_kana = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff]", text))
        latin_words = len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text))
        digits = len(re.findall(r"\d", text))
        speech_units = max(1, cjk_or_kana + latin_words + digits)
        natural_ceiling = max(6.0, speech_units * 0.55 + 4.0)
        base_ceiling = min(V25_MAX_SEGMENT_SECONDS, natural_ceiling)
        return base_ceiling / max(0.5, float(speed))

    @staticmethod
    def _split_failed_piece(text: str, limit: int) -> list[str]:
        """Split a failed piece more aggressively, even without punctuation."""
        pieces = IndexTTSv25.split_text(text, limit)
        if len(pieces) > 1 or len(text) < 2:
            return pieces
        midpoint = len(text) // 2
        return [part.strip() for part in (text[:midpoint], text[midpoint:]) if part.strip()]

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
        self.last_quality_fallback_used = False
        safe_text_limit = max(
            V25_MIN_TEXT_TOKEN_LIMIT,
            min(V25_SAFE_TEXT_TOKEN_LIMIT, int(max_text_tokens_per_segment)),
        )
        pieces = self.split_text(text, safe_text_limit)
        if not pieces:
            raise ValueError("文本为空")
        speaker = self._speaker(reference_audio)
        generated: list[np.ndarray] = []
        partial_path = Path(output_path).with_suffix(".partial.wav") if output_path else None

        def synthesize_guarded(piece: str, piece_seed: Optional[int], depth: int = 0) -> np.ndarray:
            # Acoustic tokens are roughly 30–35 Hz.  A per-piece cap prevents a
            # failed stop-token sample from running all the way to the global
            # 1500-token ceiling before the duration guard can reject it.
            base_seconds = self._reasonable_segment_seconds(piece, speed=1.0)
            guarded_mel_limit = min(
                int(max_mel_tokens),
                max(240, int(base_seconds * 36.0)),
            )
            pcm = self.runtime.synthesize(
                piece,
                lang="zh",
                spk=speaker,
                seed=piece_seed,
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                max_mel_tokens=guarded_mel_limit,
                max_text_tokens_per_segment=safe_text_limit,
                interval_silence=0,
                duration_factor=1.0 / max(0.5, float(speed)),
                n_timesteps=int(diffusion_steps),
                cfg_rate=float(cfg_rate),
            )
            audio = np.asarray(pcm, dtype=np.float32) / 32768.0
            if audio.size == 0:
                raise RuntimeError("IndexTTS 2.5 未生成有效音频")

            actual_seconds = audio.size / SAMPLE_RATE
            allowed_seconds = self._reasonable_segment_seconds(piece, speed=speed)
            if actual_seconds <= allowed_seconds:
                return audio

            self.last_quality_fallback_used = True
            if verbose:
                print(
                    "IndexTTS-2.5 duration guard rejected an abnormal segment: "
                    f"{actual_seconds:.2f}s > {allowed_seconds:.2f}s; retry depth {depth + 1}"
                )
            if depth >= V25_DURATION_GUARD_RETRIES:
                raise RuntimeError(
                    "IndexTTS 2.5 连续生成异常拉长片段；已停止保存失真音频，请缩短该句后重试"
                )

            retry_limit = max(
                V25_MIN_TEXT_TOKEN_LIMIT,
                safe_text_limit // (2 ** (depth + 1)),
            )
            retry_pieces = self._split_failed_piece(piece, retry_limit)
            retry_audio: list[np.ndarray] = []
            for retry_index, retry_piece in enumerate(retry_pieces):
                retry_seed = (
                    None
                    if piece_seed is None
                    else int(piece_seed) + 7_919 + retry_index
                )
                retry_audio.append(
                    synthesize_guarded(retry_piece, retry_seed, depth + 1)
                )
            return self._join_audio(
                retry_audio,
                min(max(0, int(interval_silence)), 120),
                0,
            )

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
            audio = synthesize_guarded(
                piece,
                None if seed is None else int(seed) + index - 1,
            )
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
