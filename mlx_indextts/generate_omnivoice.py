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


SAMPLE_RATE = 24_000


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
    def _reference_cache_key(reference_audio: str, max_duration_s: float) -> str:
        path = Path(reference_audio).resolve()
        stat = path.stat()
        source = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{float(max_duration_s):.2f}"
        return hashlib.sha256(source.encode()).hexdigest()

    def _prepare_reference_prompt(
        self,
        reference_audio: str,
        ref_text: str,
        max_duration_s: float,
    ):
        """Encode and, when needed, transcribe the exact preprocessed prompt."""
        import mlx.core as mx
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        key = self._reference_cache_key(reference_audio, max_duration_s)
        cached = self.cache.get(key)
        manual_text = str(ref_text or "").strip()
        if cached is not None and (not manual_text or manual_text == cached[1]):
            self.last_reference_transcript = str(cached[1])
            return cached

        ref_tokens = create_voice_clone_prompt(
            reference_audio,
            tokenizer=self.runtime.audio_tokenizer,
            max_duration_s=float(max_duration_s),
        )
        mx.eval(ref_tokens)
        aligned_text = manual_text
        if not aligned_text:
            if not self.asr_model_dir or not Path(self.asr_model_dir).is_dir():
                raise ValueError(
                    "OmniVoice 克隆缺少参考原文，且本地 ASR 模型未安装。"
                    "请填写参考音频原文。"
                )
            from mlx_audio.stt.utils import load_model as load_stt

            decoded = np.asarray(
                self.runtime.audio_tokenizer.decode(ref_tokens).astype(mx.float32),
                dtype=np.float32,
            ).squeeze()
            temporary_name = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                    temporary_name = temporary.name
                sf.write(temporary_name, decoded, SAMPLE_RATE, subtype="PCM_16")
                stt = load_stt(Path(self.asr_model_dir))
                transcription = stt.generate(temporary_name)
                aligned_text = str(getattr(transcription, "text", "") or "").strip()
                del stt
            finally:
                if temporary_name:
                    Path(temporary_name).unlink(missing_ok=True)
                gc.collect()
                mx.clear_cache()
            if not aligned_text:
                raise RuntimeError("ASR 未能识别参考音频，请换用 5–15 秒清晰人声。")

        cached = (ref_tokens, aligned_text)
        self.cache[key] = cached
        self.last_reference_transcript = aligned_text
        return cached

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
        return [piece for piece in pieces if piece]

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
                    instruct=str(instruct or "None") if mode == "design" else "None",
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
            audio = np.asarray(results[-1].audio, dtype=np.float32).squeeze()
            audio = np.clip(audio, -1.0, 1.0)
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
