"""WebUI adapter for the local MLX Fish Audio S2 Pro model."""

from __future__ import annotations

import gc
import hashlib
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from mlx_indextts.generate_v2 import GenerationCancelled


SAMPLE_RATE = 44_100


class _TokenizerAdapter:
    def __init__(self, owner: "FishS2ProTTS") -> None:
        self.owner = owner

    def tokenize(self, text: str) -> str:
        return text

    def split_segments(self, text: str, max_tokens_per_segment: int = 120) -> list[str]:
        return self.owner.split_text(text, max_tokens_per_segment)


class FishS2ProTTS:
    """Match the generation contract used by the multi-model WebUI."""

    sample_rate = SAMPLE_RATE
    MIN_AUDIO_TOKENS = 1024
    MAX_AUDIO_TOKENS = 4096
    MAX_SEGMENT_CHARACTERS = 60

    def __init__(self, model_dir: str, asr_model_dir: str | None = None) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        self.wired_memory_limit = 0
        if mx.metal.is_available():
            recommended_limit = int(mx.device_info().get("max_recommended_working_set_size") or 0)
            if recommended_limit > 0:
                mx.set_wired_limit(recommended_limit)
                self.wired_memory_limit = recommended_limit
        self.model_dir = str(model_dir)
        self.asr_model_dir = str(asr_model_dir) if asr_model_dir else None
        self.runtime = load_model(Path(model_dir))
        self.sample_rate = int(self.runtime.sample_rate)
        self.tokenizer = _TokenizerAdapter(self)
        self.cache: dict[str, tuple[object, str]] = {}
        self.last_reference_transcript = ""
        self.last_quality_fallback_used = False
        self.last_speed_optimization_used = False

    @staticmethod
    def split_text(text: str, max_tokens_per_segment: int = 120) -> list[str]:
        limit = max(
            20,
            min(FishS2ProTTS.MAX_SEGMENT_CHARACTERS, int(max_tokens_per_segment)),
        )
        units = re.split(r"(?<=[。！？；，.!?;,\n])", str(text))
        pieces: list[str] = []
        current = ""
        for unit in units:
            while len(unit) > limit:
                if current.strip():
                    pieces.append(current.strip())
                    current = ""
                cut = min(len(unit), limit)
                pieces.append(unit[:cut].strip())
                unit = unit[cut:]
            if current and len(current + unit) > limit:
                pieces.append(current.strip())
                current = unit
            else:
                current += unit
        if current.strip():
            pieces.append(current.strip())
        return [piece for piece in pieces if piece]

    @staticmethod
    def _reference_cache_key(reference_audio: str, max_duration_s: float) -> str:
        path = Path(reference_audio).resolve()
        stat = path.stat()
        source = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{float(max_duration_s):.2f}"
        return hashlib.sha256(source.encode()).hexdigest()

    def _prepare_reference(self, reference_audio: str, ref_text: str, max_duration_s: float) -> tuple[object, str]:
        import mlx.core as mx
        from mlx_audio.utils import load_audio

        key = self._reference_cache_key(reference_audio, max_duration_s)
        manual_text = str(ref_text or "").strip()
        cached = self.cache.get(key)
        if cached is not None and (not manual_text or manual_text == cached[1]):
            self.last_reference_transcript = cached[1]
            return cached

        audio = load_audio(
            str(reference_audio),
            sample_rate=self.sample_rate,
        )
        max_samples = int(self.sample_rate * max(3.0, float(max_duration_s)))
        if int(audio.shape[0]) > max_samples:
            audio = audio[:max_samples]
        aligned_text = manual_text
        if not aligned_text:
            if not self.asr_model_dir or not Path(self.asr_model_dir).is_dir():
                raise ValueError("Fish S2 Pro 克隆需要参考音频原文，或安装本地 ASR 模型。")
            from mlx_audio.stt.utils import load_model as load_stt

            temporary_name = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                    temporary_name = temporary.name
                sf.write(
                    temporary_name,
                    np.asarray(audio, dtype=np.float32),
                    self.sample_rate,
                    subtype="PCM_16",
                )
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
                raise RuntimeError("ASR 未能识别参考音频，请填写参考原文或更换清晰人声。")

        prepared = (audio, aligned_text)
        self.cache[key] = prepared
        self.last_reference_transcript = aligned_text
        return prepared

    @staticmethod
    def _join_audio(segments: list[np.ndarray], interval_silence: int, sample_rate: int) -> np.ndarray:
        if len(segments) == 1:
            return segments[0]
        silence = np.zeros(int(sample_rate * max(0, interval_silence) / 1000), np.float32)
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
        fish_mode: str = "clone",
        ref_text: str = "",
        instruct: str = "",
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        max_tokens: int = 1024,
        chunk_length: int = 300,
        ref_audio_max_duration_s: float = 15.0,
        **_ignored,
    ) -> np.ndarray:
        import mlx.core as mx

        mode = str(fish_mode or "clone")
        if mode == "clone" and not reference_audio:
            raise ValueError("Fish S2 Pro 音色克隆需要先选择一个参考音色。")

        planned_pieces = self.split_text(text, max_text_tokens_per_segment)
        ref_audio = None
        aligned_ref_text = ""
        if mode == "clone":
            if progress_callback:
                progress_callback(0, len(planned_pieces), "正在对齐 Fish S2 Pro 参考音频与原文")
            ref_audio, aligned_ref_text = self._prepare_reference(
                str(reference_audio), str(ref_text or ""), float(ref_audio_max_duration_s)
            )

        generated: list[np.ndarray] = []
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.stem}.partial.wav")
        partial.unlink(missing_ok=True)
        partial_file = None
        silence = np.zeros(
            int(self.sample_rate * max(0, int(interval_silence)) / 1000),
            dtype=np.float32,
        )
        try:
            for piece_index, piece in enumerate(planned_pieces, start=1):
                while pause_requested and pause_requested():
                    if cancel_requested and cancel_requested():
                        raise GenerationCancelled("用户已终止当前任务")
                    time.sleep(0.1)
                if cancel_requested and cancel_requested():
                    raise GenerationCancelled("用户已终止当前任务")

                token_limit = max(self.MIN_AUDIO_TOKENS, int(max_tokens))
                token_limit = min(self.MAX_AUDIO_TOKENS, token_limit)
                while True:
                    if progress_callback:
                        progress_callback(
                            piece_index - 1,
                            len(planned_pieces),
                            f"Fish S2 Pro 片段 {piece_index}/{len(planned_pieces)} 正在生成",
                        )
                    mx.random.seed(int(seed) + piece_index - 1)
                    results = list(
                        self.runtime.generate(
                            text=piece,
                            ref_audio=ref_audio,
                            ref_text=aligned_ref_text or None,
                            instruct=str(instruct or "").strip() or None,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            top_k=int(top_k),
                            max_tokens=token_limit,
                            speed=float(speed),
                            chunk_length=int(chunk_length),
                            stream=False,
                            verbose=False,
                        )
                    )
                    if not results:
                        raise RuntimeError("Fish S2 Pro 未返回音频。")
                    capped = any(int(getattr(result, "token_count", 0)) >= token_limit for result in results)
                    if not capped:
                        break
                    del results
                    gc.collect()
                    mx.clear_cache()
                    if token_limit >= self.MAX_AUDIO_TOKENS:
                        raise RuntimeError(
                            "Fish S2 Pro 音频达到 4096 Token 安全上限，已停止保存，"
                            "避免生成被硬截断的残缺音频。请缩短单段文案或增加标点。"
                        )
                    token_limit = min(self.MAX_AUDIO_TOKENS, token_limit * 2)

                result_audio = [np.asarray(result.audio, dtype=np.float32).squeeze().copy() for result in results]
                piece_audio = np.clip(
                    self._join_audio(result_audio, int(interval_silence), self.sample_rate),
                    -1.0,
                    1.0,
                )
                if partial_file is None:
                    partial_file = sf.SoundFile(
                        str(partial),
                        mode="w",
                        samplerate=self.sample_rate,
                        channels=1,
                        subtype="PCM_16",
                    )
                elif silence.size:
                    partial_file.write(silence)
                partial_file.write(piece_audio)
                partial_file.flush()
                generated.append(piece_audio)
                if audio_chunk_callback:
                    audio_chunk_callback(piece_audio, self.sample_rate)
                if progress_callback:
                    progress_callback(
                        piece_index,
                        len(planned_pieces),
                        f"Fish S2 Pro 片段 {piece_index}/{len(planned_pieces)} 已完成并保存",
                    )
                del results, result_audio
                gc.collect()
                mx.clear_cache()
        except BaseException:
            if partial_file is not None:
                partial_file.close()
            if not generated:
                partial.unlink(missing_ok=True)
            raise
        else:
            if partial_file is not None:
                partial_file.close()
            if not generated:
                raise RuntimeError("Fish S2 Pro 未生成可保存的音频。")
            partial.replace(target)

        joined = self._join_audio(generated, int(interval_silence), self.sample_rate)
        return joined
