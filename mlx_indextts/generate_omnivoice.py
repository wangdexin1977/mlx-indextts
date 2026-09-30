"""Small WebUI adapter for the local MLX OmniVoice model."""

from __future__ import annotations

import re
import time
import gc
import hashlib
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from mlx_indextts.generate_v2 import GenerationCancelled
from mlx_indextts.narration_text import attach_narration_punctuation


SAMPLE_RATE = 24_000
REFERENCE_PROMPT_CACHE_VERSION = 2
MAX_QUALITY_REFERENCE_SECONDS = 10.0


class _TokenizerAdapter:
    def __init__(self, owner: "OmniVoiceTTS") -> None:
        self.owner = owner

    def tokenize(self, text: str) -> str:
        return text

    def split_segments(self, text: str, max_tokens_per_segment: int = 120) -> list[str]:
        return self.owner.split_text(text, max_tokens_per_segment)


class OmniVoiceTTS:
    """Match the generation contract used by the existing IndexTTS WebUI."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: str, asr_model_dir: str | None = None) -> None:
        # Import lazily so the two existing IndexTTS backends still start even if
        # the optional OmniVoice dependency or model has been removed.
        from mlx_audio.tts.utils import load_model

        self.model_dir = str(model_dir)
        self.asr_model_dir = str(asr_model_dir) if asr_model_dir else None
        self.runtime = load_model(Path(model_dir))
        self.tokenizer = _TokenizerAdapter(self)
        self.cache: dict = {}
        self.last_quality_fallback_used = False
        self.last_speed_optimization_used = False
        self.last_reference_transcript = ""

    @staticmethod
    def _reference_cache_key(
        reference_audio: str,
        max_duration_s: float,
        ref_text: str = "",
    ) -> str:
        path = Path(reference_audio).resolve()
        stat = path.stat()
        source = (
            f"v{REFERENCE_PROMPT_CACHE_VERSION}:{path}:{stat.st_size}:"
            f"{stat.st_mtime_ns}:{float(max_duration_s):.2f}:{str(ref_text).strip()}"
        )
        return hashlib.sha256(source.encode()).hexdigest()

    @staticmethod
    def _reference_excerpt(
        audio: np.ndarray,
        sample_rate: int,
        max_duration_s: float,
    ) -> tuple[np.ndarray, bool]:
        """Keep a clean prompt no longer than OmniVoice's recommended 10 seconds."""
        mono = np.asarray(audio, dtype=np.float32).squeeze()
        if mono.ndim != 1:
            mono = np.mean(mono, axis=-1, dtype=np.float32)
        mono = np.nan_to_num(mono, copy=False)
        if mono.size == 0:
            raise ValueError("OmniVoice 参考音频为空。")

        peak = float(np.max(np.abs(mono)))
        threshold = max(1e-4, peak * 0.01)
        active = np.flatnonzero(np.abs(mono) >= threshold)
        if active.size:
            padding = int(sample_rate * 0.08)
            start = max(0, int(active[0]) - padding)
            end = min(mono.size, int(active[-1]) + padding + 1)
            mono = mono[start:end]

        requested = max(3.0, float(max_duration_s))
        limit_s = min(requested, MAX_QUALITY_REFERENCE_SECONDS)
        limit = max(1, int(sample_rate * limit_s))
        was_shortened = mono.size > limit
        if was_shortened:
            # Prefer a quiet boundary close to the limit so the prompt does not
            # end in the middle of a syllable. Search only the final 1.5 seconds
            # to retain as much speaker evidence as possible.
            search_start = max(int(sample_rate * 3.0), limit - int(sample_rate * 1.5))
            window = max(1, int(sample_rate * 0.04))
            envelope = np.convolve(
                np.abs(mono[:limit]),
                np.ones(window, dtype=np.float32) / window,
                mode="same",
            )
            boundary = search_start + int(np.argmin(envelope[search_start:limit]))
            if boundary >= int(sample_rate * 3.0):
                limit = boundary
            mono = mono[:limit]

        mono = mono - float(np.mean(mono))
        fade = min(int(sample_rate * 0.01), mono.size // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            mono[:fade] *= ramp
            mono[-fade:] *= ramp[::-1]
        return mono.astype(np.float32, copy=False), was_shortened

    @staticmethod
    def _postprocess_audio(audio: np.ndarray) -> np.ndarray:
        """Apply transparent edge cleanup without changing the model timbre."""
        cleaned = np.asarray(audio, dtype=np.float32).squeeze()
        if cleaned.ndim != 1:
            cleaned = np.mean(cleaned, axis=-1, dtype=np.float32)
        cleaned = np.nan_to_num(cleaned, copy=False)
        if cleaned.size == 0:
            return cleaned
        cleaned = cleaned - float(np.mean(cleaned))
        peak = float(np.max(np.abs(cleaned)))
        threshold = max(1e-4, peak * 0.003)
        active = np.flatnonzero(np.abs(cleaned) >= threshold)
        if active.size:
            padding = int(SAMPLE_RATE * 0.05)
            start = max(0, int(active[0]) - padding)
            end = min(cleaned.size, int(active[-1]) + padding + 1)
            cleaned = cleaned[start:end]
        fade = min(int(SAMPLE_RATE * 0.01), cleaned.size // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            cleaned[:fade] *= ramp
            cleaned[-fade:] *= ramp[::-1]
        return np.clip(cleaned, -1.0, 1.0)

    def _prepare_reference_prompt(
        self,
        reference_audio: str,
        ref_text: str,
        max_duration_s: float,
    ):
        """Encode and, when needed, transcribe the exact preprocessed prompt."""
        import mlx.core as mx
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        manual_text = str(ref_text or "").strip()
        key = self._reference_cache_key(reference_audio, max_duration_s, manual_text)
        cached = self.cache.get(key)
        if cached is not None:
            self.last_reference_transcript = str(cached[1])
            return cached

        import librosa

        source_audio, source_sr = librosa.load(reference_audio, sr=None, mono=True)
        if int(source_sr) != SAMPLE_RATE:
            source_audio = librosa.resample(
                source_audio,
                orig_sr=int(source_sr),
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
        excerpt, was_shortened = self._reference_excerpt(
            source_audio,
            SAMPLE_RATE,
            max_duration_s,
        )
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary_name = temporary.name
            sf.write(temporary_name, excerpt, SAMPLE_RATE, subtype="PCM_24")
            ref_tokens = create_voice_clone_prompt(
                temporary_name,
                tokenizer=self.runtime.audio_tokenizer,
                max_duration_s=min(float(max_duration_s), MAX_QUALITY_REFERENCE_SECONDS),
            )
            mx.eval(ref_tokens)

            # A transcript for the original 15-second library preview no longer
            # aligns after selecting the high-quality 10-second excerpt. In that
            # case, transcribe the exact waveform that was encoded.
            aligned_text = "" if was_shortened else manual_text
            if not aligned_text:
                aligned_text = self._transcribe_reference(temporary_name)
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)

        cached = (ref_tokens, aligned_text)
        self.cache[key] = cached
        self.last_reference_transcript = aligned_text
        return cached

    def _transcribe_reference(self, reference_wav: str) -> str:
        """Transcribe the exact prompt waveform passed to the audio tokenizer."""
        import mlx.core as mx

        if not self.asr_model_dir or not Path(self.asr_model_dir).is_dir():
            raise ValueError(
                "OmniVoice 高质量克隆需要本地 ASR 对齐参考片段，"
                "但未找到 ASR 模型。"
            )
        temporary_name = ""
        try:
            if not self.asr_model_dir or not Path(self.asr_model_dir).is_dir():
                raise ValueError(
                    "OmniVoice 克隆缺少参考原文，且本地 ASR 模型未安装。"
                    "请填写参考音频原文。"
                )
            from mlx_audio.stt.utils import load_model as load_stt

            stt = load_stt(Path(self.asr_model_dir))
            transcription = stt.generate(reference_wav)
            aligned_text = str(getattr(transcription, "text", "") or "").strip()
            del stt
        finally:
            gc.collect()
            mx.clear_cache()
        if not aligned_text:
            raise RuntimeError("ASR 未能识别参考音频，请换用 3–10 秒清晰人声。")
        return aligned_text

    @staticmethod
    def split_text(text: str, max_tokens_per_segment: int = 120) -> list[str]:
        limit = max(30, min(240, int(max_tokens_per_segment)))
        units = re.split(r"(?<=[。！？；，.!?;,\n])", str(text))
        pieces: list[str] = []
        current = ""
        for unit in units:
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
        return attach_narration_punctuation(pieces)

    @staticmethod
    def _estimated_duration(text: str, speed: float) -> float:
        cjk = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff]", text))
        words = len(re.findall(r"[A-Za-z0-9]+", text))
        punctuation = len(re.findall(r"[。！？；，.!?;,:]", text))
        seconds = max(1.2, cjk / 4.3 + words / 2.6 + punctuation * 0.12)
        return seconds / max(0.5, float(speed))

    @staticmethod
    def _join_audio(segments: list[np.ndarray], interval_silence: int) -> np.ndarray:
        if len(segments) == 1:
            return segments[0]
        silence = np.zeros(int(SAMPLE_RATE * max(0, interval_silence) / 1000), np.float32)
        joined: list[np.ndarray] = []
        for index, segment in enumerate(segments):
            if index and silence.size:
                joined.append(silence)
            joined.append(segment)
        return np.concatenate(joined)

    def generate(
        self,
        *,
        text: str,
        reference_audio: str | None,
        output_path: str,
        speed: float = 1.0,
        seed: int = 42,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 250,
        progress_callback: Callable[[int, int, str], None] | None = None,
        audio_chunk_callback: Callable[[np.ndarray, int], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        pause_requested: Callable[[], bool] | None = None,
        omnivoice_mode: str = "clone",
        language: str = "chinese",
        ref_text: str = "",
        instruct: str = "",
        duration_s: float = 0.0,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
        position_temperature: float = 5.0,
        layer_penalty_factor: float = 5.0,
        t_shift: float = 0.1,
        ref_audio_max_duration_s: float = 10.0,
        **_ignored,
    ) -> np.ndarray:
        mode = str(omnivoice_mode or "clone")
        if mode == "clone" and not reference_audio:
            raise ValueError("OmniVoice 音色克隆需要先选择一个参考音色。")
        if mode == "design" and not str(instruct or "").strip():
            raise ValueError("音色设计模式需要填写“音色设计描述”。")

        import mlx.core as mx

        pieces = self.split_text(text, max_text_tokens_per_segment)
        ref_tokens = None
        aligned_ref_text = ""
        if mode == "clone":
            if progress_callback:
                progress_callback(0, len(pieces), "正在对齐 OmniVoice 参考音频与原文")
            ref_tokens, aligned_ref_text = self._prepare_reference_prompt(
                str(reference_audio), str(ref_text or ""), float(ref_audio_max_duration_s)
            )
        generated: list[np.ndarray] = []
        total_weight = max(1, sum(len(re.sub(r"\s", "", piece)) for piece in pieces))
        for index, piece in enumerate(pieces, start=1):
            while pause_requested and pause_requested():
                if cancel_requested and cancel_requested():
                    raise GenerationCancelled("用户已终止当前任务")
                time.sleep(0.1)
            if cancel_requested and cancel_requested():
                raise GenerationCancelled("用户已终止当前任务")
            if progress_callback:
                progress_callback(index - 1, len(pieces), f"OmniVoice 片段 {index}/{len(pieces)}")

            mx.random.seed(int(seed) + index - 1)
            piece_duration = None
            if float(duration_s) > 0:
                weight = len(re.sub(r"\s", "", piece))
                piece_duration = max(0.4, float(duration_s) * weight / total_weight)
            elif abs(float(speed) - 1.0) > 0.01:
                piece_duration = self._estimated_duration(piece, float(speed))

            results = list(
                self.runtime.generate(
                    text=piece,
                    duration_s=piece_duration,
                    language=str(language or "None"),
                    # OmniVoice supports combining a reference clip with valid
                    # style attributes.  Keep the reference as the identity
                    # anchor and pass the optional style in clone mode too.
                    instruct=str(instruct or "None"),
                    ref_audio=None,
                    ref_tokens=ref_tokens,
                    ref_text=aligned_ref_text or None,
                    ref_audio_max_duration_s=float(ref_audio_max_duration_s),
                    num_steps=int(num_steps),
                    guidance_scale=float(guidance_scale),
                    class_temperature=float(class_temperature),
                    position_temperature=float(position_temperature),
                    layer_penalty_factor=float(layer_penalty_factor),
                    t_shift=float(t_shift),
                )
            )
            if not results:
                raise RuntimeError("OmniVoice 未返回音频。")
            audio = self._postprocess_audio(results[-1].audio)
            generated.append(audio)
            if audio_chunk_callback:
                audio_chunk_callback(audio, SAMPLE_RATE)
            if progress_callback:
                progress_callback(index, len(pieces), f"OmniVoice 片段 {index}/{len(pieces)} 已完成")

        joined = self._join_audio(generated, int(interval_silence))
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), joined, SAMPLE_RATE, subtype="PCM_16")
        return joined
