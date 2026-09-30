"""CosyVoice3 voice-cloning adapter for the existing local WebUI."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import select
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from mlx_indextts.generate_v2 import GenerationCancelled


SAMPLE_RATE = 24_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "cosyvoice3-macos"
PROMPT_CACHE = PROJECT_ROOT / "outputs" / "webui" / "voices" / "cosyvoice3_prompts"


class _TokenizerAdapter:
    def tokenize(self, text: str) -> str:
        return text

    def split_segments(self, text: str, max_tokens_per_segment: int = 120) -> list[str]:
        limit = max(25, min(100, int(max_tokens_per_segment)))
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
            if current and len(current + unit) > limit:
                pieces.append(current.strip())
                current = unit
            else:
                current += unit
        if current.strip():
            pieces.append(current.strip())
        return [piece for piece in pieces if piece]


class CosyVoice3TTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: str, asr_model_dir: str) -> None:
        self.model_dir = str(Path(model_dir).resolve())
        self.asr_model_dir = str(Path(asr_model_dir).resolve())
        self.tokenizer = _TokenizerAdapter()
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.runtime_config: tuple[str, int] | None = None
        self.requests_since_load = 0
        self.warm_footprint = 0
        self.last_reference_transcript = ""
        self.last_quality_fallback_used = False
        self.last_speed_optimization_used = False
        self.last_load_seconds = 0.0
        self.last_generation_rtf = 0.0

    def close(self) -> None:
        process = self.process
        self.process = None
        self.runtime_config = None
        self.requests_since_load = 0
        self.warm_footprint = 0
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    @staticmethod
    def _check_cancel(cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested and cancel_requested():
            raise GenerationCancelled("用户已终止当前任务")

    def _response(self, cancel_requested: Callable[[], bool] | None) -> dict:
        assert self.process is not None and self.process.stdout is not None
        while True:
            try:
                self._check_cancel(cancel_requested)
            except GenerationCancelled:
                self.close()
                raise
            ready, _, _ = select.select([self.process.stdout], [], [], 0.2)
            if ready:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("CosyVoice3 进程意外退出，请查看 outputs/webui/cosyvoice3-worker.log")
                response = json.loads(line)
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error") or "CosyVoice3 生成失败"))
                return response
            if self.process.poll() is not None:
                raise RuntimeError("CosyVoice3 进程意外退出，请查看 outputs/webui/cosyvoice3-worker.log")

    def _send(self, payload: dict) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _ensure_runtime(
        self, prompt_wav: str, prompt_text: str, precision: str, nfe: int,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        selected = (precision, nfe)
        if self.process is not None and self.process.poll() is None and self.runtime_config == selected and self.requests_since_load < 100:
            return
        self.close()
        python = RUNTIME_ROOT / ".venv" / "bin" / "python"
        if not python.is_file():
            raise RuntimeError("CosyVoice3 专用 Python 环境尚未安装。")
        log_path = PROJECT_ROOT / "outputs" / "webui" / "cosyvoice3-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((
            str(RUNTIME_ROOT), str(RUNTIME_ROOT / "third_party" / "Matcha-TTS"),
            str(PROJECT_ROOT), environment.get("PYTHONPATH", ""),
        ))
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        environment["COSYVOICE_LOCAL_ONLY"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        self.process = subprocess.Popen(
            [str(python), str(PROJECT_ROOT / "mlx_indextts" / "cosyvoice3_worker.py")],
            cwd=RUNTIME_ROOT, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
            text=True, bufsize=1,
        )
        self._send({
            "model_dir": self.model_dir, "prompt_wav": prompt_wav,
            "prompt_text": prompt_text, "spk_id": "initial",
            "device": "mps", "nfe": nfe, "llm_backend": "mlx",
            "llm_precision": precision, "llm_batch_size": 1,
        })
        try:
            response = self._response(cancel_requested)
        except BaseException:
            self.close()
            raise
        self.runtime_config = selected
        self.last_load_seconds = float(response["load_seconds"])
        self.sample_rate = int(response["sample_rate"])

    def _prepare_reference(self, reference_audio: str, ref_text: str, max_duration_s: float) -> tuple[str, str]:
        import librosa
        from mlx_indextts.generate_omnivoice import OmniVoiceTTS

        source = Path(reference_audio).resolve()
        stat = source.stat()
        manual_text = str(ref_text or "").strip()
        digest = hashlib.sha256(
            f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{max_duration_s:.2f}:{manual_text}".encode()
        ).hexdigest()
        PROMPT_CACHE.mkdir(parents=True, exist_ok=True)
        wav_path = PROMPT_CACHE / f"{digest}.wav"
        json_path = PROMPT_CACHE / f"{digest}.json"
        if wav_path.is_file() and json_path.is_file():
            transcript = json.loads(json_path.read_text(encoding="utf-8"))["text"]
            self.last_reference_transcript = transcript
            return str(wav_path), transcript

        audio, source_rate = librosa.load(str(source), sr=None, mono=True)
        if int(source_rate) != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=int(source_rate), target_sr=SAMPLE_RATE, res_type="soxr_hq")
        excerpt, shortened = OmniVoiceTTS._reference_excerpt(audio, SAMPLE_RATE, max_duration_s)
        if len(excerpt) < 3 * SAMPLE_RATE:
            raise ValueError("CosyVoice3 克隆需要至少 3 秒清晰参考人声。")
        sf.write(str(wav_path), excerpt, SAMPLE_RATE, subtype="PCM_24")
        transcript = "" if shortened else manual_text
        if not transcript:
            if not Path(self.asr_model_dir).is_dir():
                raise RuntimeError("缺少本地 ASR 模型，请填写参考音频原文。")
            import mlx.core as mx
            from mlx_audio.stt.utils import load_model as load_stt

            stt = load_stt(Path(self.asr_model_dir))
            try:
                result = stt.generate(str(wav_path))
                transcript = str(getattr(result, "text", "") or "").strip()
            finally:
                del stt
                gc.collect()
                mx.clear_cache()
        if not transcript:
            wav_path.unlink(missing_ok=True)
            raise RuntimeError("参考音频原文识别失败，请填写原文或更换清晰人声。")
        json_path.write_text(json.dumps({"text": transcript}, ensure_ascii=False), encoding="utf-8")
        self.last_reference_transcript = transcript
        return str(wav_path), transcript

    def generate(
        self, *, text: str, reference_audio: str | None, output_path: str,
        speed: float = 1.0, seed: int = 42, max_text_tokens_per_segment: int = 120,
        interval_silence: int = 250,
        progress_callback: Callable[[int, int, str], None] | None = None,
        audio_chunk_callback: Callable[[np.ndarray, int], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        pause_requested: Callable[[], bool] | None = None,
        cosy_ref_text: str = "", cosy_precision: str = "fp16", cosy_nfe: int = 10,
        cosy_ref_audio_max_duration_s: float = 10.0, **_ignored,
    ) -> np.ndarray:
        if not reference_audio:
            raise ValueError("CosyVoice3 音色克隆需要先选择参考音色。")
        precision = str(cosy_precision)
        nfe = int(cosy_nfe)
        if precision not in {"fp16", "fp32"} or nfe < 4 or nfe > 20:
            raise ValueError("CosyVoice3 精度或流匹配步数超出允许范围。")
        pieces = self.tokenizer.split_segments(text, max_text_tokens_per_segment)
        if not pieces:
            raise ValueError("没有可生成的文本。")
        if progress_callback:
            progress_callback(0, len(pieces), "正在对齐 CosyVoice3 参考音频与原文")
        prompt_wav, prompt_text = self._prepare_reference(
            reference_audio, cosy_ref_text, float(cosy_ref_audio_max_duration_s)
        )
        self._check_cancel(cancel_requested)
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.stem}.partial.wav")
        partial.unlink(missing_ok=True)
        generated: list[np.ndarray] = []
        elapsed = 0.0
        duration = 0.0
        silence = np.zeros(int(self.sample_rate * max(0, int(interval_silence)) / 1000), np.float32)
        try:
            for index, piece in enumerate(pieces, 1):
                while pause_requested and pause_requested():
                    self._check_cancel(cancel_requested)
                    time.sleep(0.1)
                self._check_cancel(cancel_requested)
                self._ensure_runtime(prompt_wav, prompt_text, precision, nfe, cancel_requested)
                if progress_callback:
                    progress_callback(index - 1, len(pieces), f"CosyVoice3 片段 {index}/{len(pieces)} 正在生成")
                piece_path = target.with_name(f"{target.stem}.cosy_{index:04d}.wav")
                try:
                    self._send({
                        "text": piece, "prompt_wav": prompt_wav, "prompt_text": prompt_text,
                        "speed": float(speed), "seed": int(seed) + index - 1,
                        "output_path": str(piece_path),
                    })
                    response = self._response(cancel_requested)
                    audio, sample_rate = sf.read(str(piece_path), dtype="float32", always_2d=False)
                    if int(sample_rate) != self.sample_rate:
                        raise RuntimeError("CosyVoice3 返回的音频采样率异常。")
                    audio = np.asarray(audio, np.float32).squeeze()
                    if audio.ndim != 1 or audio.size == 0:
                        raise RuntimeError("CosyVoice3 未返回有效音频。")
                    generated.append(audio)
                    elapsed += float(response["elapsed"])
                    duration += float(response["duration"])
                    self.requests_since_load += 1
                    footprint = int(response.get("footprint_bytes") or 0)
                    if not self.warm_footprint:
                        self.warm_footprint = footprint
                    if footprint >= 14 * 1024**3 or footprint - self.warm_footprint >= 2 * 1024**3:
                        self.close()
                    if audio_chunk_callback:
                        audio_chunk_callback(audio, self.sample_rate)
                    joined = np.concatenate([part for i, clip in enumerate(generated) for part in ((silence, clip) if i else (clip,))])
                    sf.write(str(partial), joined, self.sample_rate, subtype="PCM_24")
                    if progress_callback:
                        progress_callback(index, len(pieces), f"CosyVoice3 已完成 {index}/{len(pieces)} 片段")
                finally:
                    piece_path.unlink(missing_ok=True)
            partial.replace(target)
            self.last_generation_rtf = elapsed / duration if duration else 0.0
            return joined
        except Exception:
            raise
