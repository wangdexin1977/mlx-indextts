"""VoiceStudio v0.5.2 native OmniVoice, isolated from the WebUI MLX process."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import json

import numpy as np
import soundfile as sf

from mlx_indextts.generate_omnivoice import OmniVoiceTTS, _TokenizerAdapter
from mlx_indextts.generate_fish_s2 import FishS2ProTTS
from mlx_indextts.generate_v2 import GenerationCancelled


class VoiceStudioTTS(OmniVoiceTTS):
    def __init__(self, source_dir: str, model_dir: str, asr_model_dir: str | None = None):
        self.source_dir = Path(source_dir)
        self.model_dir = str(model_dir)
        if not (self.source_dir / 'omnivoice/models/omnivoice.py').is_file():
            raise RuntimeError('找不到 VoiceStudio 源码，请重新下载 v0.5.2。')
        if (not (Path(model_dir) / 'model.safetensors').is_file()
                or (Path(model_dir) / 'model.safetensors.aria2').exists()
                or not (Path(model_dir) / 'audio_tokenizer/model.safetensors').is_file()):
            raise RuntimeError('VoiceStudio 原生模型权重尚未下载完成。')
        self.asr_model_dir = asr_model_dir
        self.tokenizer = _TokenizerAdapter(self)
        self.cache = {}
        self.last_reference_transcript = ''
        self.last_quality_fallback_used = False
        self.last_speed_optimization_used = False
        self._process = None
        self._responses = queue.Queue()
        self._log = None

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self._log:
            self._log.close()
            self._log = None

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        self._responses = queue.Queue()
        log_path = Path(self.model_dir).parent.parent / 'outputs/webui/logs/voicestudio-worker.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open('a')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTORCH_ENABLE_MPS_FALLBACK='1', TOKENIZERS_PARALLELISM='false')
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name('voicestudio_worker.py')),
             str(self.source_dir), self.model_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1, env=env,
        )
        process, responses = self._process, self._responses

        def receive():
            try:
                for line in process.stdout:
                    responses.put(json.loads(line))
            except Exception as exc:
                responses.put({'ok': False, 'error': str(exc)})
            finally:
                responses.put({'ok': False, 'error': 'VoiceStudio 子进程已退出，详情见 voicestudio-worker.log'})

        threading.Thread(target=receive, daemon=True).start()

    def _request(self, payload, cancel_requested):
        self._start()
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self._process.stdin.flush()
        deadline = time.monotonic() + 300
        while True:
            if cancel_requested and cancel_requested():
                self.close()
                raise GenerationCancelled('用户已终止当前任务')
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError('VoiceStudio 单段生成超过 300 秒，已释放模型，请缩短片段后重试。')
            try:
                response = self._responses.get(timeout=0.1)
            except queue.Empty:
                continue
            if not response.get('ok'):
                self.close()
                raise RuntimeError(response.get('error', 'VoiceStudio 生成失败'))
            if response.get('sample_rate') != self.sample_rate:
                raise RuntimeError('VoiceStudio 采样率与当前适配器不一致')
            return

    def generate(self, *, text, reference_audio, output_path, speed=1.0, seed=42,
                 max_text_tokens_per_segment=120, interval_silence=250,
                 progress_callback=None, audio_chunk_callback=None,
                 cancel_requested=None, pause_requested=None,
                 omnivoice_mode='clone', language='chinese', ref_text='', instruct='',
                 duration_s=0.0, num_steps=32, guidance_scale=2.0,
                 class_temperature=0.0, position_temperature=5.0,
                 layer_penalty_factor=5.0, t_shift=0.1,
                 ref_audio_max_duration_s=10.0, **_ignored):
        if omnivoice_mode == 'clone' and not reference_audio:
            raise ValueError('VoiceStudio 克隆需要先选择参考音色。')
        if omnivoice_mode == 'design' and not instruct.strip():
            raise ValueError('VoiceStudio 音色设计需要填写描述。')
        pieces = self.split_text(text, max_text_tokens_per_segment)
        if not pieces:
            raise ValueError('请输入需要生成的文字。')
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.stem + '.partial.wav')
        generated = []
        total_weight = sum(len(piece) for piece in pieces)
        with tempfile.TemporaryDirectory(prefix='voicestudio-') as temporary:
            reference = None
            if omnivoice_mode == 'clone':
                audio, ref_text = FishS2ProTTS._prepare_reference(
                    self, reference_audio, ref_text, ref_audio_max_duration_s,
                )
                reference = str(Path(temporary) / 'reference.wav')
                sf.write(reference, np.asarray(audio), self.sample_rate, subtype='FLOAT')
            try:
                with sf.SoundFile(partial, 'w', samplerate=self.sample_rate,
                                  channels=1, subtype='PCM_16') as writer:
                    for index, piece in enumerate(pieces):
                        while pause_requested and pause_requested():
                            if cancel_requested and cancel_requested():
                                raise GenerationCancelled('用户已终止当前任务')
                            time.sleep(0.1)
                        if cancel_requested and cancel_requested():
                            raise GenerationCancelled('用户已终止当前任务')
                        if progress_callback:
                            progress_callback(index, len(pieces), f'VoiceStudio 片段 {index + 1}/{len(pieces)} 正在生成')
                        piece_path = str(Path(temporary) / 'piece.wav')
                        self._request(dict(
                            text=piece, seed=int(seed) + index, reference=reference,
                            ref_text=ref_text, language=None if language == 'None' else language,
                            # Native OmniVoice also accepts a validated style
                            # instruct alongside a clone reference.
                            instruct=instruct or None,
                            duration=float(duration_s) * len(piece) / total_weight if duration_s > 0 else None,
                            speed=float(speed), output=piece_path,
                            config=dict(num_step=int(num_steps), guidance_scale=float(guidance_scale),
                                        class_temperature=float(class_temperature), position_temperature=float(position_temperature),
                                        layer_penalty_factor=float(layer_penalty_factor), t_shift=float(t_shift)),
                        ), cancel_requested)
                        audio, _ = sf.read(piece_path, dtype='float32')
                        audio = np.clip(audio, -1, 1)
                        if index and interval_silence > 0:
                            writer.write(np.zeros(int(self.sample_rate * interval_silence / 1000)))
                        writer.write(audio)
                        writer.flush()
                        generated.append(audio)
                        if audio_chunk_callback:
                            audio_chunk_callback(audio, self.sample_rate)
                        if progress_callback:
                            progress_callback(index + 1, len(pieces), f'VoiceStudio 片段 {index + 1}/{len(pieces)} 已保存')
            except BaseException:
                if not generated:
                    partial.unlink(missing_ok=True)
                raise
        partial.replace(target)
        return self._join_audio(generated, interval_silence)
