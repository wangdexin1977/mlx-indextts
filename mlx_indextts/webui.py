"""Local Gradio WebUI for MLX IndexTTS2."""

from __future__ import annotations

import gc
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import gradio as gr

# All required weights are installed locally; keep the WebUI network-independent.
# Assign explicitly because a parent shell may define these variables as "0".
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from mlx_indextts.generate_v2 import (
    GenerationCancelled,
    IndexTTSv2,
    MAX_GENERATION_SEGMENTS,
    analyze_audio_quality,
)
from mlx_indextts.generate_v25 import IndexTTSv25
from mlx_indextts.generate_fish_s2 import FishS2ProTTS
from mlx_indextts.generate_omnivoice import OmniVoiceTTS
from mlx_indextts.document_import import (
    DocumentImportError,
    ImportedDocument,
    count_effective_characters,
    estimate_audio_minutes,
    import_document,
    select_document_text,
)
from mlx_indextts.power_monitor import start_macos_power_monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "0.3.3"
MODEL_DIR = PROJECT_ROOT / "models" / "mlx-IndexTTS-2.5-int8"
MODEL_V2_DIR = PROJECT_ROOT / "models" / "mlx-IndexTTS-2"
OMNIVOICE_MODEL_DIR = PROJECT_ROOT / "models" / "OmniVoice-bfloat16"
OMNIVOICE_ASR_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-ASR-0.6B-8bit"
FISH_S2_MODEL_DIR = PROJECT_ROOT / "models" / "fish-audio-s2-pro-8bit"
DEFAULT_SPEAKER = PROJECT_ROOT / "outputs" / "voice_01_speaker.npz"
DEFAULT_VOICE_PREVIEW = PROJECT_ROOT / "outputs" / "voice_01_preview.wav"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "webui"
VOICE_DIR = OUTPUT_DIR / "voices"
CONFIG_PATH = OUTPUT_DIR / "user_settings.json"
DOCUMENT_CACHE_DIR = OUTPUT_DIR / "document_cache"
OPTIMIZED_VOICE_PATH = VOICE_DIR / "current_voice_optimized.wav"
VOICE_CONDITIONING_PATH = VOICE_DIR / "current_voice_v25.npz"
VOICE_SAMPLE_RATE = 22_050
VOICE_MAX_DURATION_S = 15.0
VOICE_CACHE_VERSION = 25
OUTPUT_FORMATS = ("wav", "mp3", "flac")
SUPPORTED_VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff"}
MAX_SYNTHESIS_CHARACTERS = 100_000
SYNTHESIS_BATCH_CHARACTERS = 2_500
MAX_SYNTHESIS_BATCH_SEGMENTS = 32
MLX_CACHE_CLEAR_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024

DEFAULT_SETTINGS = {
    "model_backend": "IndexTTS 2.5",
    "emotion": "跟随参考音频",
    "emotion_strength": 0.6,
    "speed": 1.0,
    "seed": 42,
    "interval_silence": 250,
    "segment_overlap_ms": 50,
    "max_text_tokens": 120,
    "temperature": 0.8,
    "diffusion_steps": 25,
    "max_mel_tokens": 1500,
    "top_p": 0.8,
    "top_k": 30,
    "repetition_penalty": 10.0,
    "cfg_rate": 0.7,
    "fast_vocoder": False,
}

DEFAULT_OMNIVOICE_SETTINGS = {
    "omnivoice_mode": "clone",
    "omnivoice_language": "chinese",
    "omnivoice_ref_text": "",
    "omnivoice_instruct": "",
    "omnivoice_duration_s": 0.0,
    "omnivoice_num_steps": 32,
    "omnivoice_guidance_scale": 2.0,
    "omnivoice_class_temperature": 0.0,
    "omnivoice_position_temperature": 5.0,
    "omnivoice_layer_penalty_factor": 5.0,
    "omnivoice_t_shift": 0.1,
    "omnivoice_ref_audio_max_duration_s": 10.0,
}

DEFAULT_FISH_S2_SETTINGS = {
    "fish_mode": "clone",
    "fish_ref_text": "",
    "fish_instruct": "",
    "fish_temperature": 0.7,
    "fish_top_p": 0.7,
    "fish_top_k": 30,
    "fish_max_tokens": 1024,
    "fish_chunk_length": 300,
    "fish_ref_audio_max_duration_s": 15.0,
}

MODEL_BACKENDS = ("IndexTTS 2.5", "IndexTTS 2.0", "OmniVoice", "Fish Audio S2 Pro")

EMOTIONS = {
    "自然/平静": "calm",
    "高兴": "happy",
    "悲伤": "sad",
    "愤怒": "angry",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "跟随参考音频": None,
}

_model: IndexTTSv25 | IndexTTSv2 | OmniVoiceTTS | FishS2ProTTS | None = None
_model_backend: str | None = None
_model_lock = threading.Lock()
_synthesis_job_lock = threading.Lock()
_document_queue_active = threading.Event()
_document_queue_cancelled = threading.Event()
_config_lock = threading.RLock()
_voice_library_lock = threading.RLock()
_generation_active = threading.Event()
_generation_paused = threading.Event()
_generation_system_paused = threading.Event()
_generation_cancelled = threading.Event()
_generation_control_lock = threading.Lock()
_generation_progress_lock = threading.RLock()
_generation_progress_state = {
    "state": "idle",
    "current": 0,
    "total": 0,
    "message": "等待开始生成",
    "started_at": None,
    "paused_at": None,
    "paused_seconds": 0.0,
    "finished_elapsed": None,
}
_power_monitor = None


APP_CSS = """
:root {
  --app-navy: #0647a6;
  --app-blue: #087cf0;
  --app-line: #d8e1ec;
  --app-surface: #ffffff;
  --app-muted: #5d6b7d;
}

body, .gradio-container {
  background: #eef3f8 !important;
  color: #182235 !important;
  font-size: 16px !important;
}

.gradio-container {
  width: calc(100vw - 20px) !important;
  max-width: none !important;
  margin: 0 auto !important;
  padding: 10px 12px 2px !important;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  min-height: 62px;
  margin-bottom: 10px;
  padding: 10px 18px;
  border: 1px solid rgba(149,224,255,.58);
  border-radius: 12px;
  background: linear-gradient(115deg, #0757d9 0%, #0788ee 58%, #00aeea 100%);
  box-shadow: 0 8px 26px rgba(0, 105, 220, .24);
  color: #fff;
}

.app-brand { display: flex; align-items: center; gap: 12px; }
.app-mark {
  display: grid; place-items: center; flex: 0 0 40px; height: 40px;
  border: 1px solid rgba(255,255,255,.28); border-radius: 10px;
  background: rgba(255,255,255,.11); font-size: 20px; font-weight: 800;
}
.app-title { margin: 0; font-size: 23px; line-height: 1.15; font-weight: 760; letter-spacing: .2px; }
.app-title { color: #ffffff !important; }
.app-subtitle { margin-top: 3px; color: #e1f5ff; font-size: 14px; line-height: 1.25; }
.app-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }
.app-badge {
  padding: 5px 9px; border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
  background: rgba(255,255,255,.14); color: #f4fbff; font-size: 13px; font-weight: 620;
}

.workspace { gap: 12px !important; align-items: flex-start !important; }
.workspace-main {
  flex: 2 1 0% !important;
  width: 66.666% !important;
  min-width: 0 !important;
}
.workspace-sidebar {
  flex: 1 1 0% !important;
  width: 33.333% !important;
  min-width: 340px !important;
  gap: 12px !important;
}
.panel {
  min-width: 0 !important;
  padding: 12px 13px 11px !important;
  border: 1px solid var(--app-line) !important;
  border-radius: 11px !important;
  background: var(--app-surface) !important;
  box-shadow: 0 3px 12px rgba(30, 54, 82, .07) !important;
}
.panel-heading {
  margin: 0 0 8px !important; padding: 0 0 7px !important;
  border-bottom: 1px solid #e6ecf3; color: #213451;
  font-size: 16px !important; font-weight: 760 !important; letter-spacing: .15px;
}

.gradio-container label, .gradio-container .label-wrap {
  font-size: 15px !important;
  font-weight: 680 !important;
  color: #263b57 !important;
}
.gradio-container textarea, .gradio-container input, .gradio-container select {
  font-size: 16px !important;
}
.gradio-container .form { gap: 7px !important; }
.gradio-container .block { margin: 0 !important; }

.text-entry textarea {
  min-height: 420px !important;
  height: 420px !important;
  padding: 15px 16px !important;
  font-size: 17px !important;
  line-height: 1.85 !important;
  letter-spacing: .12px !important;
  overflow-wrap: anywhere !important;
  resize: vertical !important;
}
.text-entry {
  border: 1px solid #c5d5e8 !important;
  border-radius: 10px !important;
  background: #fbfdff !important;
  box-shadow: inset 0 1px 3px rgba(30, 64, 110, .05) !important;
}
.text-entry-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 28px;
  margin-bottom: 5px;
}
.text-entry-label {
  color: #263b57;
  font-size: 15px;
  font-weight: 680;
}
.text-counter {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
  padding: 3px 9px;
  border: 1px solid #cbd9ea;
  border-radius: 999px;
  background: #f3f7fc;
  color: #4b607b;
  font-size: 13px;
  font-weight: 650;
  white-space: nowrap;
}
.text-counter strong {
  color: #1d4ed8;
  font-size: 17px;
  font-weight: 800;
}
.text-counter-over-limit {
  border-color: #f3a7a7;
  background: #fff1f1;
  color: #b42318;
}
.text-counter-over-limit strong { color: #dc2626; }
.document-queue {
  margin-bottom: 8px !important;
  border: 1px solid #8eb9e8 !important;
  border-radius: 9px !important;
  background: #f3f8ff !important;
}
.document-queue summary {
  min-height: 42px !important;
  padding: 8px 10px !important;
  color: #164d88 !important;
  font-size: 15px !important;
  font-weight: 780 !important;
}
.document-queue .wrap { gap: 7px !important; }
.queue-guide {
  padding: 8px 10px;
  border: 1px solid #c5daf1;
  border-radius: 7px;
  background: #fff;
  color: #35536f;
  font-size: 13px;
  line-height: 1.55;
}
.queue-guide strong { color: #174b85; }
.queue-empty {
  padding: 9px 10px;
  border: 1px dashed #b8cbe0;
  border-radius: 7px;
  color: #5b6d82;
  font-size: 13px;
}
.queue-overview {
  overflow: hidden;
  border: 1px solid #cad9e9;
  border-radius: 8px;
  background: #fff;
}
.queue-overview-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 7px 9px;
  border-bottom: 1px solid #dde7f1;
  background: #edf5ff;
  color: #31506f;
  font-size: 12px;
}
.queue-list { max-height: 230px; overflow: auto; }
.queue-item {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  align-items: center;
  gap: 7px;
  padding: 7px 9px;
  border-bottom: 1px solid #e8eef5;
}
.queue-item:last-child { border-bottom: 0; }
.queue-order { color: #7890aa; font-size: 12px; font-weight: 760; }
.queue-item-main { min-width: 0; }
.queue-item-main strong {
  display: block;
  overflow: hidden;
  color: #223e5c;
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.queue-item-main small { color: #728399; font-size: 11px; }
.queue-state {
  padding: 2px 7px;
  border-radius: 999px;
  background: #edf1f6;
  color: #53677e;
  font-size: 11px;
  font-weight: 750;
}
.queue-status-confirmed .queue-state { background: #e8f5ed; color: #18743b; }
.queue-status-running .queue-state { background: #e8f1ff; color: #1d4ed8; }
.queue-status-completed .queue-state { background: #dcfce7; color: #166534; }
.queue-status-failed .queue-state { background: #fee2e2; color: #b91c1c; }
.queue-status-stopped .queue-state { background: #f1f5f9; color: #64748b; }
.queue-item-detail {
  overflow: hidden;
  margin-top: 2px;
  color: #2f6eb2;
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.queue-item-error { color: #b91c1c; }
.queue-file-picker { min-height: 84px !important; }
.queue-actions, .queue-order-actions { gap: 6px !important; }
.queue-actions button, .queue-order-actions button {
  min-width: 0 !important;
  min-height: 34px !important;
  padding: 5px 7px !important;
  font-size: 13px !important;
  font-weight: 700 !important;
}
.queue-confirm-action {
  border-color: #25834b !important;
  background: #edf9f1 !important;
  color: #176a38 !important;
}
.queue-start-action {
  border: 0 !important;
  background: linear-gradient(105deg, #0757d9, #0788ee) !important;
  color: #fff !important;
}
.document-import {
  margin-bottom: 8px !important;
  border: 1px solid #cdd9e7 !important;
  border-radius: 9px !important;
  background: #f8fbff !important;
}
.document-import summary {
  min-height: 40px !important;
  padding: 8px 10px !important;
  color: #203956 !important;
  font-size: 15px !important;
  font-weight: 760 !important;
}
.document-import .wrap { gap: 7px !important; }
.document-summary {
  padding: 9px 10px;
  border: 1px solid #d8e3ef;
  border-radius: 8px;
  background: #fff;
  color: #344b66;
  font-size: 13px;
  line-height: 1.5;
}
.document-summary-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 5px;
}
.document-summary-name {
  overflow: hidden;
  color: #17375f;
  font-size: 14px;
  font-weight: 760;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.document-summary-ready {
  flex: 0 0 auto;
  padding: 2px 7px;
  border-radius: 999px;
  background: #eaf7ee;
  color: #18743b;
  font-size: 12px;
  font-weight: 760;
}
.document-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 5px;
}
.document-metric {
  padding: 5px 6px;
  border-radius: 6px;
  background: #f0f5fb;
  text-align: center;
}
.document-metric strong { display: block; color: #1d4ed8; font-size: 15px; }
.document-warning { margin-top: 6px; color: #9a5200; }
.document-actions { gap: 6px !important; }
.document-action {
  min-height: 34px !important;
  padding: 5px 8px !important;
  font-size: 13px !important;
  font-weight: 700 !important;
}
.document-preview textarea {
  min-height: 76px !important;
  height: 76px !important;
  color: #47586d !important;
  font-size: 13px !important;
  line-height: 1.45 !important;
}
.reference-intro {
  margin: 2px 0 7px !important;
  color: var(--app-muted) !important;
  font-size: 14px !important;
  line-height: 1.45 !important;
}
.voice-library {
  margin: 7px 0 8px !important;
  border: 1px solid #bdd3ec !important;
  border-radius: 9px !important;
  background: #f6faff !important;
}
.voice-library summary {
  min-height: 40px !important;
  padding: 8px 10px !important;
  color: #174b85 !important;
  font-size: 15px !important;
  font-weight: 760 !important;
}
.voice-library-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 7px 9px;
  border: 1px solid #d6e4f3;
  border-radius: 7px;
  background: #fff;
  color: #45617e;
  font-size: 12px;
}
.voice-library-summary strong { color: #174b85; font-size: 13px; }
.voice-library-summary span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.voice-library-guide {
  padding: 10px 12px;
  border: 1px solid #8fc2f4;
  border-radius: 8px;
  background: linear-gradient(135deg, #eaf5ff 0%, #f5faff 100%);
  color: #174b85;
  font-size: 13px;
  line-height: 1.55;
}
.voice-library-guide strong {
  display: block;
  margin-bottom: 2px;
  color: #0b5cab;
  font-size: 14px;
}
.voice-library-preview audio { height: 38px !important; }
.voice-library-import {
  min-height: 42px !important;
  border: 1px solid #176fd1 !important;
  background: linear-gradient(135deg, #1474dc 0%, #2f8fee 100%) !important;
  color: #fff !important;
  font-weight: 780 !important;
  box-shadow: 0 5px 12px rgb(26 111 203 / 20%) !important;
}
.voice-library-import:hover {
  background: linear-gradient(135deg, #0c61bd 0%, #227fd9 100%) !important;
}
.voice-batch-files {
  min-height: 142px !important;
  border: 2px dashed #65a8e8 !important;
  border-radius: 9px !important;
  background: #fff !important;
  font-size: 13px !important;
}
.voice-library-divider {
  margin: 4px 0 0;
  color: #315878;
  font-size: 13px;
  font-weight: 730;
}
.saved-voice-actions {
  display: grid !important;
  grid-template-columns: minmax(0, 85fr) minmax(0, 7fr) minmax(0, 8fr) !important;
  align-items: start !important;
  gap: 5px !important;
}
.saved-voice-actions > * { min-width: 0 !important; width: 100% !important; }
.saved-voice-selector { min-width: 0 !important; }
.saved-voice-selector input {
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}
.quick-voice-title {
  margin: 5px 0 1px;
  color: #174b85;
  font-size: 13px;
  font-weight: 780;
}
.quick-voice-grid { gap: 6px !important; }
.quick-voice-button {
  min-width: 0 !important;
  min-height: 46px !important;
  padding: 6px 9px !important;
  border: 1px solid #91bce8 !important;
  background: #fff !important;
  color: #174b85 !important;
  font-size: 12px !important;
  font-weight: 720 !important;
  line-height: 1.3 !important;
  overflow: hidden !important;
  white-space: normal !important;
}
.quick-voice-remove {
  min-width: 46px !important;
  min-height: 46px !important;
  padding: 5px !important;
  border: 1px solid #efb2b2 !important;
  background: #fff7f7 !important;
  color: #b83232 !important;
  font-size: 11px !important;
  font-weight: 760 !important;
}
.favorite-voice-button {
  min-width: 0 !important;
  min-height: 34px !important;
  padding: 3px 2px !important;
  border: 1px solid #e1ae38 !important;
  background: #fff8df !important;
  color: #8a5a00 !important;
  font-size: 11px !important;
  font-weight: 760 !important;
}
.voice-delete-button {
  min-width: 0 !important;
  min-height: 34px !important;
  padding: 3px 2px !important;
  font-size: 11px !important;
  font-weight: 740 !important;
}
.reference-audio {
  min-height: 157px !important;
  overflow: visible !important;
}
.reference-audio .wrap,
.reference-audio .audio-container,
.reference-audio .record-button {
  min-height: 94px !important;
}
.reference-audio:has(.waveform-container) .audio-container {
  height: 132px !important;
  min-height: 132px !important;
}
.reference-audio:has(.waveform-container) .component-wrapper {
  height: 86px !important;
  min-height: 86px !important;
}
.reference-audio:has(.waveform-container) .waveform-container,
.reference-audio:has(.waveform-container) .waveform-container > div {
  height: 32px !important;
  min-height: 32px !important;
}
.reference-audio:has(.waveform-container) .timestamps {
  display: none !important;
}
.reference-audio:has(.waveform-container) .controls {
  height: 36px !important;
  min-height: 36px !important;
}
.reference-audio:has(.waveform-container) .source-selection {
  height: 36px !important;
  min-height: 36px !important;
}
.reference-audio button { font-size: 15px !important; }
.reference-audio button[aria-label="Upload file"],
.reference-audio button[aria-label="Record audio"] {
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  border: 1px solid #cbd8e6 !important;
  border-radius: 8px !important;
  background: #f6f9fd !important;
}
.reference-audio button[aria-label="Upload file"] svg,
.reference-audio button[aria-label="Record audio"] svg {
  width: 20px !important;
  height: 20px !important;
}
.gradio-container footer { display: none !important; }
.control-panel { gap: 7px !important; }
.control-pair { gap: 8px !important; }
.control-pair > div { min-width: 0 !important; }
.control-pair input { min-height: 42px !important; }
.compact-control-panel {
  min-height: 0 !important;
  padding: 4px 5px !important;
  gap: 3px !important;
}
.compact-parameter-accordion {
  border: 1px solid #cbd8e8 !important;
  border-radius: 8px !important;
  background: #f8fbff !important;
}
.compact-parameter-accordion summary {
  min-height: 28px !important;
  padding: 3px 7px !important;
  color: #213451 !important;
  font-size: 14px !important;
  font-weight: 760 !important;
}
.compact-parameter-accordion .wrap {
  gap: 6px !important;
  padding: 7px 8px 8px !important;
}
.compact-parameter-grid {
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 6px !important;
}
.compact-parameter-grid > div { min-width: 0 !important; }
.compact-parameter-grid input { min-height: 38px !important; }
.compact-generation-controls {
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 4px !important;
}
.compact-generation-controls > button {
  width: auto !important;
  min-width: 0 !important;
  min-height: 34px !important;
  margin: 0 !important;
  padding: 4px 5px !important;
  font-size: 12px !important;
}
.compact-generation-controls > .primary-action {
  flex: 2 1 0 !important;
  font-size: 14px !important;
}
.compact-generation-controls > .pause-action,
.compact-generation-controls > .stop-action {
  flex: 1 1 0 !important;
}
.secondary-options {
  margin-top: 2px !important;
  border: 1px solid #d8e1ec !important;
  border-radius: 8px !important;
  background: #f8fafc !important;
}
.secondary-options summary {
  min-height: 36px !important;
  padding: 7px 10px !important;
  color: #52657d !important;
  font-size: 13px !important;
  font-weight: 700 !important;
}
.secondary-options .wrap { padding-top: 2px !important; }

.primary-action {
  min-height: 48px !important;
  margin-top: 4px !important;
  border: 0 !important;
  border-radius: 9px !important;
  background: linear-gradient(105deg, #1d4ed8, #2563eb) !important;
  box-shadow: 0 5px 14px rgba(37, 99, 235, .22) !important;
  color: #fff !important;
  font-size: 17px !important;
  font-weight: 760 !important;
}
.primary-action:hover { filter: brightness(1.05); transform: translateY(-1px); }
.generation-controls {
  display: flex !important;
  flex-wrap: nowrap !important;
  gap: 7px !important;
}
.generation-controls > button {
  flex: 1 1 0 !important;
  width: auto !important;
  min-width: 0 !important;
}
.pause-action, .stop-action {
  min-height: 38px !important;
  border-radius: 8px !important;
  font-size: 14px !important;
  font-weight: 720 !important;
}
.pause-action {
  border: 1px solid #d3a72e !important;
  background: #fff8dc !important;
  color: #7a5200 !important;
}
.stop-action {
  border: 1px solid #e19a9a !important;
  background: #fff1f1 !important;
  color: #a51d1d !important;
}
.gradio-container input[type="range"] { accent-color: var(--app-blue) !important; }

.about-trigger {
  padding: 5px 10px;
  border: 1px solid rgba(255,255,255,.34);
  border-radius: 999px;
  background: rgba(255,255,255,.14);
  color: #fff;
  cursor: pointer;
  font: inherit;
  font-size: 13px;
  font-weight: 700;
}
.about-trigger:hover { background: rgba(255,255,255,.24); }
.about-modal {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(8, 20, 38, .58);
  backdrop-filter: blur(4px);
}
.about-modal.is-open { display: flex; }
.about-card {
  width: min(820px, calc(100vw - 40px));
  max-height: calc(100vh - 48px);
  overflow: auto;
  border: 1px solid #cdd8e6;
  border-radius: 14px;
  background: #fff;
  box-shadow: 0 22px 70px rgba(8, 24, 48, .28);
}
.about-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 16px 20px;
  border-bottom: 1px solid #e0e7f0;
  background: linear-gradient(110deg, #10233f, #1e4b82);
  color: #fff;
}
.about-card-head h2 { margin: 0; color: #fff !important; font-size: 21px; }
.about-close {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  padding: 0;
  border: 1px solid rgba(255,255,255,.3);
  border-radius: 8px;
  background: rgba(255,255,255,.12);
  color: #fff;
  cursor: pointer;
  font-size: 22px;
  font-family: Arial, sans-serif;
  line-height: 1;
}
.about-card-body { padding: 18px 20px 20px; }
.about-current {
  margin-bottom: 14px;
  padding: 12px 14px;
  border: 1px solid #bcd1ec;
  border-radius: 10px;
  background: #f0f6ff;
  color: #183b68;
  font-size: 15px;
  line-height: 1.55;
}
.about-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.about-table th, .about-table td {
  padding: 9px 10px;
  border: 1px solid #dce4ee;
  text-align: left;
  vertical-align: top;
}
.about-table th { background: #f4f7fb; color: #263d5b; font-weight: 750; }
.version-installed { color: #137333; font-weight: 760; }
.version-absent { color: #a44b00; font-weight: 760; }
.about-changelog-title {
  margin: 18px 0 9px;
  color: #203a5d;
  font-size: 17px;
  font-weight: 780;
}
.about-release {
  margin-top: 9px;
  padding: 11px 13px;
  border: 1px solid #dce4ee;
  border-radius: 9px;
  background: #fbfcfe;
  color: #33465f;
  font-size: 14px;
  line-height: 1.55;
}
.about-release-head { margin-bottom: 5px; color: #173f70; }
.about-release-head strong { margin-right: 7px; font-size: 15px; }
.about-release ul { margin: 5px 0 0 19px; padding: 0; }
.about-release li { margin: 3px 0; }
.about-note { margin: 13px 0 0; color: #5d6b7d; font-size: 13px; line-height: 1.55; }

.result-panel { min-height: 105px !important; gap: 7px !important; }
.result-panel audio { height: 42px !important; }
.status-box textarea { min-height: 58px !important; line-height: 1.4 !important; }
.result-section-label {
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 1px 0 0;
  color: #4d617a;
  font-size: 12px;
  font-weight: 760;
  letter-spacing: .5px;
  text-transform: uppercase;
}
.result-section-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #e2e8f0;
}
.primary-result-audio {
  padding: 2px 7px 7px !important;
  border: 1px solid #c9d8eb !important;
  border-radius: 9px !important;
  background: #f8fbff !important;
}
.primary-result-audio label {
  color: #173b6a !important;
  font-weight: 760 !important;
}
.voice-details {
  margin-top: 4px;
  padding-top: 2px;
}
.generation-progress {
  margin-bottom: 8px;
  padding: 10px 11px;
  border: 1px solid #cbd8e8;
  border-radius: 9px;
  background: #fff;
}
.generation-progress-head, .generation-progress-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.generation-progress-title { color: #1d3555; font-size: 14px; font-weight: 760; }
.generation-progress-percent { color: #dc2626; font-size: 20px; font-weight: 850; }
.generation-progress-track {
  height: 9px;
  margin: 8px 0 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #e5ebf3;
}
.generation-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #1d4ed8, #38a3ff);
  transition: width .35s ease;
}
.generation-progress-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
  margin: 8px 0 7px;
}
.generation-progress-metric {
  padding: 7px 5px;
  border: 1px solid #dbe5f1;
  border-radius: 7px;
  background: #f6f9fd;
  color: #607087;
  text-align: center;
  font-size: 14px;
}
.generation-progress-metric strong {
  display: block;
  margin-bottom: 2px;
  color: #173b6a;
  font-size: 19px;
  line-height: 1;
}
.generation-progress-metric:first-child strong { color: #dc2626; }
.generation-progress-meta { color: #3f5066; font-size: 15px; font-weight: 650; }
.generation-progress-time { color: #dc2626; font-weight: 800; }
.generation-progress-message {
  margin-top: 6px;
  overflow: hidden;
  color: #314966;
  font-size: 16px;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.output-settings {
  gap: 8px !important;
  margin-bottom: 7px !important;
}
.output-settings .form { min-width: 0 !important; }
.output-folder-action { min-width: 112px !important; }
.output-location textarea { font-size: 13px !important; }
.voice-preview { margin: -2px 0 2px !important; }
.voice-preview audio { height: 38px !important; }
.voice-profile {
  margin-bottom: 5px;
  padding: 10px 11px;
  border: 1px solid #cddbeb;
  border-radius: 9px;
  background: linear-gradient(135deg, #f7faff 0%, #eef5ff 100%);
}
.voice-profile-head {
  display: flex;
  align-items: center;
  gap: 7px;
  color: #28405f;
  font-size: 14px;
  font-weight: 720;
}
.voice-profile-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #16a34a;
  box-shadow: 0 0 0 3px rgba(22, 163, 74, .12);
}
.voice-profile-ready {
  margin-left: auto;
  padding: 2px 7px;
  border-radius: 999px;
  background: #dcfce7;
  color: #15803d;
  font-size: 12px;
  font-weight: 700;
}
.voice-profile-name {
  margin: 7px 0 5px;
  overflow: hidden;
  color: #172b47;
  font-size: 16px;
  font-weight: 760;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.voice-profile-meta {
  color: #607087;
  font-size: 13px;
  line-height: 1.4;
}
.voice-profile-reminder {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid #d9e5f2;
  color: #244d83;
  font-size: 13px;
  font-weight: 650;
  line-height: 1.4;
}
.app-note {
  margin-top: 8px; padding: 7px 10px; border: 1px solid #dbe5f0; border-radius: 8px;
  background: #f7faff; color: var(--app-muted); font-size: 14px; line-height: 1.35;
}
.settings-sidebar {
  background: #f7f9fc !important;
  border-left: 1px solid #d4dfeb !important;
}
/* Keep the two-column workspace at full width while the fixed settings
   drawer is open. Gradio otherwise shrinks the whole app by 370px, causing
   headings and controls to collide behind the drawer. */
.contain:has(.settings-sidebar.open),
.contain:has(.settings-sidebar.open) > .column {
  width: 100% !important;
  max-width: 100% !important;
  flex-basis: 100% !important;
}
.sidebar-parent:has(.settings-sidebar.open) {
  padding-right: 0 !important;
}
.settings-title {
  margin-bottom: 3px !important;
  color: #172b47 !important;
  font-size: 20px !important;
  font-weight: 780 !important;
}
.settings-description {
  margin-bottom: 10px !important;
  color: #5d6b7d !important;
  font-size: 14px !important;
  line-height: 1.55 !important;
}
.preset-row { gap: 6px !important; margin-bottom: 9px !important; }
.preset-button {
  min-width: 0 !important;
  padding: 7px 5px !important;
  border: 1px solid #b9c9dc !important;
  background: #ffffff !important;
  color: #274467 !important;
  font-size: 14px !important;
  font-weight: 720 !important;
}
.preset-button.preset-selected {
  border-color: #2563eb !important;
  background: #eff6ff !important;
  color: #1d4ed8 !important;
  box-shadow: inset 0 0 0 1px #2563eb, 0 2px 7px rgba(37, 99, 235, .14) !important;
}
.settings-sidebar .info {
  color: #66758a !important;
  font-size: 13px !important;
  line-height: 1.4 !important;
}
.settings-reset {
  margin-top: 8px !important;
  border: 1px solid #b9c9dc !important;
  background: #ffffff !important;
  color: #274467 !important;
  font-weight: 700 !important;
}
button[aria-label="Toggle Sidebar"] {
  width: 46px !important;
  min-width: 46px !important;
  height: 38px !important;
  padding: 0 !important;
  border: 1px solid #29496f !important;
  border-radius: 9px 0 0 9px !important;
  background: #17375f !important;
  box-shadow: 0 4px 12px rgba(16, 35, 63, .18) !important;
  color: #ffffff !important;
}
button[aria-label="Toggle Sidebar"] svg { display: none !important; }
button[aria-label="Toggle Sidebar"]::after {
  content: "⚙";
  margin: 0;
  font-size: 22px;
  line-height: 1;
}

@media (max-width: 980px) {
  .gradio-container { padding: 8px !important; }
  .app-header { align-items: flex-start; padding: 10px 12px; }
  .app-badges { display: none; }
  .workspace, .result-row { flex-direction: column !important; }
  .workspace-main, .workspace-sidebar {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
  }
  .text-entry textarea {
    min-height: 300px !important;
    height: 300px !important;
  }
  .voice-library-summary { align-items: flex-start; flex-direction: column; }
}
"""


def get_model() -> IndexTTSv25:
    """Load the IndexTTS-2.5 reference-following backend."""
    global _model, _model_backend
    if _model is None or _model_backend != "2.5":
        with _model_lock:
            if _model is None or _model_backend != "2.5":
                _release_loaded_model_unlocked()
                if not MODEL_DIR.exists():
                    raise RuntimeError(f"找不到模型目录：{MODEL_DIR}")
                _model = IndexTTSv25(str(MODEL_DIR))
                _model_backend = "2.5"
    assert isinstance(_model, IndexTTSv25)
    return _model


def _release_loaded_model_unlocked() -> None:
    """Release the inactive backend before loading the other large model."""
    global _model, _model_backend
    _model = None
    _model_backend = None
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, RuntimeError):
        pass


def get_legacy_emotion_model() -> IndexTTSv2:
    """Load IndexTTS-2.0 only when a named emotion is requested."""
    global _model, _model_backend
    if _model is None or _model_backend != "2.0-emotion":
        with _model_lock:
            if _model is None or _model_backend != "2.0-emotion":
                _release_loaded_model_unlocked()
                if not MODEL_V2_DIR.exists():
                    raise RuntimeError(f"找不到情绪模型目录：{MODEL_V2_DIR}")
                _model = IndexTTSv2(
                    str(MODEL_V2_DIR),
                    memory_limit_gb=12.0,
                    quantize_bits=4,
                )
                _model_backend = "2.0-emotion"
    assert isinstance(_model, IndexTTSv2)
    return _model


def get_omnivoice_model() -> OmniVoiceTTS:
    """Load the local Apple-Silicon OmniVoice backend on demand."""
    global _model, _model_backend
    if _model is None or _model_backend != "omnivoice":
        with _model_lock:
            if _model is None or _model_backend != "omnivoice":
                _release_loaded_model_unlocked()
                if not OMNIVOICE_MODEL_DIR.exists():
                    raise RuntimeError(f"找不到 OmniVoice 模型目录：{OMNIVOICE_MODEL_DIR}")
                _model = OmniVoiceTTS(
                    str(OMNIVOICE_MODEL_DIR),
                    asr_model_dir=str(OMNIVOICE_ASR_MODEL_DIR),
                )
                _model_backend = "omnivoice"
    assert isinstance(_model, OmniVoiceTTS)
    return _model


def get_fish_s2_model() -> FishS2ProTTS:
    """Load Fish Audio S2 Pro 8-bit on demand and release the prior backend."""
    global _model, _model_backend
    if _model is None or _model_backend != "fish-s2-pro":
        with _model_lock:
            if _model is None or _model_backend != "fish-s2-pro":
                _release_loaded_model_unlocked()
                if not FISH_S2_MODEL_DIR.exists():
                    raise RuntimeError(f"找不到 Fish Audio S2 Pro 模型目录：{FISH_S2_MODEL_DIR}")
                _model = FishS2ProTTS(
                    str(FISH_S2_MODEL_DIR),
                    asr_model_dir=str(OMNIVOICE_ASR_MODEL_DIR),
                )
                _model_backend = "fish-s2-pro"
    assert isinstance(_model, FishS2ProTTS)
    return _model


def _read_config_unlocked() -> dict:
    config = {
        **DEFAULT_SETTINGS,
        **DEFAULT_OMNIVOICE_SETTINGS,
        **DEFAULT_FISH_S2_SETTINGS,
        "reference_audio": None,
        "reference_conditioning": None,
        "reference_cache_version": None,
        "optimized_duration": None,
        "voice_name": None,
        "voice_library_id": None,
        "output_format": "wav",
        "output_directory": str(OUTPUT_DIR),
        "model_version": "2.5",
    }
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except (OSError, json.JSONDecodeError):
            pass

    reference = config.get("reference_audio")
    if reference and not Path(str(reference)).exists():
        config["reference_audio"] = None
        config["reference_conditioning"] = None
        config["reference_cache_version"] = None
        config["optimized_duration"] = None
        config["voice_name"] = None
        config["voice_library_id"] = None
    conditioning = config.get("reference_conditioning")
    if conditioning and not Path(str(conditioning)).exists():
        config["reference_conditioning"] = None
        config["reference_cache_version"] = None
        config["optimized_duration"] = None
    if config.get("emotion") not in EMOTIONS:
        config["emotion"] = DEFAULT_SETTINGS["emotion"]
    if config.get("model_backend") not in MODEL_BACKENDS:
        old_version = str(config.get("model_version") or "2.5")
        config["model_backend"] = "IndexTTS 2.0" if old_version.startswith("2.0") else "IndexTTS 2.5"
    if str(config.get("output_format", "")).lower() not in OUTPUT_FORMATS:
        config["output_format"] = "wav"
    if not str(config.get("output_directory") or "").strip():
        config["output_directory"] = str(OUTPUT_DIR)
    try:
        if int(config.get("fish_max_tokens") or 0) < FishS2ProTTS.MIN_AUDIO_TOKENS:
            config["fish_max_tokens"] = FishS2ProTTS.MIN_AUDIO_TOKENS
    except (TypeError, ValueError):
        config["fish_max_tokens"] = FishS2ProTTS.MIN_AUDIO_TOKENS
    return config


def read_user_config() -> dict:
    """Read the persisted voice and generation settings."""
    with _config_lock:
        return _read_config_unlocked()


def update_user_config(**updates: object) -> None:
    """Atomically update the local persistent configuration."""
    with _config_lock:
        config = _read_config_unlocked()
        config.update(updates)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = CONFIG_PATH.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(CONFIG_PATH)


def _normalise_output_format(value: str | None) -> str:
    output_format = str(value or "wav").strip().lower()
    return output_format if output_format in OUTPUT_FORMATS else "wav"


def _resolve_output_directory(value: str | None, *, create: bool = True) -> Path:
    raw_path = str(value or OUTPUT_DIR).strip()
    output_directory = Path(raw_path).expanduser()
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory
    output_directory = output_directory.resolve()
    if create:
        output_directory.mkdir(parents=True, exist_ok=True)
    if not output_directory.is_dir():
        raise ValueError(f"输出地址不是文件夹：{output_directory}")
    if not os.access(output_directory, os.W_OK):
        raise PermissionError(f"输出文件夹不可写：{output_directory}")
    return output_directory


def _default_audio_filename_stem(text: str, limit: int = 15) -> str:
    """Build a readable, filesystem-safe stem from the opening copy."""
    compact = re.sub(r"\s+", "", str(text or ""))
    opening = compact[:limit]
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", opening).strip(" ._")
    return safe or "未命名音频"


def _available_audio_basename(directory: Path, text: str, output_format: str) -> str:
    """Avoid overwriting an earlier result with the same opening copy."""
    stem = _default_audio_filename_stem(text)
    candidate = stem
    sequence = 2
    while True:
        conflicts = (
            directory / f"{candidate}.{output_format}",
            directory / f"{candidate}.wav",
            directory / f"{candidate}.partial.wav",
            directory / f".{candidate}.parts",
        )
        if not any(path.exists() for path in conflicts):
            return candidate
        candidate = f"{stem}_{sequence}"
        sequence += 1


def _convert_output_audio(source_wav: Path, target_path: Path, output_format: str) -> None:
    """Convert a generated WAV to the selected delivery format."""
    if output_format == "wav":
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg，无法输出 MP3 或 FLAC；可先改选 WAV。")
    codec_args = {
        "mp3": ["-codec:a", "libmp3lame", "-q:a", "2"],
        "flac": ["-codec:a", "flac"],
    }[output_format]
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_wav), *codec_args, str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not target_path.exists():
        detail = (result.stderr or "未知转换错误").strip()
        raise RuntimeError(f"{output_format.upper()} 转换失败：{detail}")


def _concatenate_wav_batches(
    batch_paths: list[Path],
    target_path: Path,
    gap_ms: int = 0,
) -> None:
    """Join disk-backed WAV batches without loading the full result into memory."""
    if not batch_paths:
        raise ValueError("没有可合并的音频批次")
    if len(batch_paths) == 1:
        shutil.copy2(batch_paths[0], target_path)
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg，无法合并长文音频批次。")

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for batch_path in batch_paths:
        command.extend(["-i", str(batch_path)])
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    gap_seconds = max(0, int(gap_ms)) / 1000
    for index in range(len(batch_paths)):
        label = f"a{index}"
        if gap_seconds and index < len(batch_paths) - 1:
            filter_parts.append(f"[{index}:a]apad=pad_dur={gap_seconds:.3f}[{label}]")
        else:
            filter_parts.append(f"[{index}:a]anull[{label}]")
        concat_inputs.append(f"[{label}]")
    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={len(batch_paths)}:v=0:a=1[outa]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outa]",
            "-codec:a",
            "pcm_s16le",
            str(target_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not target_path.exists():
        detail = (result.stderr or "未知合并错误").strip()
        raise RuntimeError(f"长文音频合并失败：{detail}")


def _release_mlx_batch_memory() -> None:
    """Finish pending Metal work and clear buffers only when cache is excessive."""
    try:
        import mlx.core as mx

        mx.synchronize()
        if mx.get_cache_memory() > MLX_CACHE_CLEAR_THRESHOLD_BYTES:
            mx.clear_cache()
    except (ImportError, RuntimeError):
        # Generation itself reports real MLX failures. Cache cleanup is best-effort.
        pass


def _start_sleep_prevention() -> subprocess.Popen | None:
    """Prevent idle sleep while a long synthesis job is actively running."""
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        return None
    try:
        return subprocess.Popen(
            [caffeinate, "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None


def _stop_sleep_prevention(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def _schedule_directory_cleanup(directory: Path, delay_seconds: float = 300.0) -> None:
    """Remove live-playback files after the browser has had time to fetch them."""
    cleanup_timer = threading.Timer(
        delay_seconds,
        shutil.rmtree,
        args=(directory,),
        kwargs={"ignore_errors": True},
    )
    cleanup_timer.daemon = True
    cleanup_timer.start()


def open_output_directory(output_directory: str) -> str:
    """Open the configured local output folder in the system file manager."""
    try:
        directory = _resolve_output_directory(output_directory)
        opener = shutil.which("open") or shutil.which("xdg-open")
        if not opener:
            raise RuntimeError("当前系统没有可用的文件夹打开命令。")
        subprocess.Popen(
            [opener, str(directory)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"已打开输出文件夹：{directory}"
    except Exception as exc:
        raise gr.Error(f"无法打开输出文件夹：{exc}") from exc


def render_voice_profile(
    reference_audio: str | None,
    display_name: str | None = None,
    conditioning_ready: bool = False,
    optimized_duration: float | None = None,
) -> str:
    """Render a persistent summary of the voice used for upcoming synthesis."""
    if reference_audio:
        filename = html.escape(display_name or Path(str(reference_audio)).name)
        voice_name = filename
        source = "用户上传或麦克风录音"
        if conditioning_ready:
            duration_note = f" · {optimized_duration:.1f} 秒有效音色" if optimized_duration else ""
            cache_note = f'<div class="voice-profile-meta">加速缓存：已预计算{duration_note}</div>'
        else:
            cache_note = '<div class="voice-profile-meta">加速缓存：将在首次使用时建立</div>'
    else:
        voice_name = "尚未选择音色"
        source = "请先添加并选择一个自定义音色"
        cache_note = '<div class="voice-profile-meta">当前不能开始合成</div>'

    ready_label = "已就绪" if reference_audio else "未就绪"

    return f"""
    <div class="voice-profile">
      <div class="voice-profile-head">
        <span class="voice-profile-dot"></span>
        <span>当前使用音色</span>
        <span class="voice-profile-ready">{ready_label}</span>
      </div>
      <div class="voice-profile-name" title="{voice_name}">{voice_name}</div>
      <div class="voice-profile-meta">来源：{source}</div>
      {cache_note}
      <div class="voice-profile-reminder">{('接下来提交的文本都将按照此音色生成' if reference_audio else '选择音色后才能生成语音')}</div>
    </div>
    """


def _voice_library_root() -> Path:
    return VOICE_DIR / "library"


def _voice_entry_directory(voice_id: str) -> Path:
    return _voice_library_root() / voice_id


def _load_voice_entry(voice_id: str | None) -> dict | None:
    if not voice_id:
        return None
    metadata_path = _voice_entry_directory(str(voice_id)) / "metadata.json"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata_path.parent / str(data["source_filename"])
        preview = metadata_path.parent / "preview.wav"
        if not source.is_file() or not preview.is_file():
            return None
        data.update(
            source_path=str(source),
            preview_path=str(preview),
            conditioning_path=str(metadata_path.parent / "conditioning_v25.npz"),
            conditioning_v2_path=str(metadata_path.parent / "conditioning_v2.npz"),
        )
        return data
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def list_voice_library() -> list[dict]:
    """Return every valid locally persisted voice, newest first."""
    root = _voice_library_root()
    if not root.exists():
        return []
    with _voice_library_lock:
        entries = [
            entry
            for directory in root.iterdir()
            if directory.is_dir()
            for entry in [_load_voice_entry(directory.name)]
            if entry is not None
        ]
    return sorted(entries, key=lambda item: str(item.get("added_at") or ""), reverse=True)


def _voice_library_choices() -> list[tuple[str, str]]:
    choices = []
    for entry in list_voice_library():
        display_name = str(entry.get("name") or entry.get("original_filename") or "未命名音色")
        choices.append((f"{display_name} · {entry['id'][:6]}", str(entry["id"])))
    return choices


def _quick_voice_entries(limit: int = 10) -> list[dict]:
    """Return up to ten voices explicitly marked as favorites by the user."""
    entries = [entry for entry in list_voice_library() if bool(entry.get("is_favorite"))]
    return sorted(
        entries,
        key=lambda item: float(item.get("favorite_added_at") or 0.0),
    )[:limit]


def _update_voice_metadata(voice_id: str, **updates: object) -> dict | None:
    metadata_path = _voice_entry_directory(voice_id) / "metadata.json"
    with _voice_library_lock:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(updates)
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(metadata_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    return _load_voice_entry(voice_id)


def _mark_voice_used(voice_id: str) -> dict | None:
    """Update local usage metadata without touching the source audio."""
    metadata_path = _voice_entry_directory(voice_id) / "metadata.json"
    with _voice_library_lock:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["use_count"] = int(metadata.get("use_count") or 0) + 1
            metadata["last_used_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            temporary = metadata_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(metadata_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return _load_voice_entry(voice_id)
    return _load_voice_entry(voice_id)


def load_quick_voice_state() -> tuple:
    """Populate ten fixed quick-access slots and their hidden IDs."""
    entries = _quick_voice_entries()
    buttons = []
    remove_buttons = []
    voice_ids = []
    for index in range(10):
        entry = entries[index] if index < len(entries) else None
        name = str(entry.get("name") or entry.get("original_filename")) if entry else ""
        buttons.append(gr.Button(value=name, visible=entry is not None))
        remove_buttons.append(gr.Button(value="移出", visible=entry is not None))
        voice_ids.append(str(entry["id"]) if entry else "")
    return (*buttons, *remove_buttons, *voice_ids)


def set_voice_as_favorite(voice_id: str | None) -> str:
    """Add the selected library voice to one of ten manually managed slots."""
    entry = _load_voice_entry(voice_id)
    if entry is None:
        return "请先从音色列表中选择一个音色，再点击“设为常用”。"
    if entry.get("is_favorite"):
        return f"音色“{entry.get('name')}”已经是常用音色。"
    if len(_quick_voice_entries()) >= 10:
        return "常用音色已满 10 个，请先点击某个常用音色旁的“移出”。"
    updated = _update_voice_metadata(
        str(entry["id"]),
        is_favorite=True,
        favorite_added_at=time.time(),
    )
    if updated is None:
        return "设置常用失败：无法更新音色信息。"
    return f"已将音色“{entry.get('name')}”设为常用。"


def remove_voice_from_favorites(voice_id: str | None) -> str:
    """Remove a quick shortcut without deleting the saved voice itself."""
    entry = _load_voice_entry(voice_id)
    if entry is None:
        return "该常用音色已不存在，快捷入口已刷新。"
    updated = _update_voice_metadata(
        str(entry["id"]),
        is_favorite=False,
        favorite_added_at=None,
    )
    if updated is None:
        return "移出常用失败：无法更新音色信息。"
    return f"已将音色“{entry.get('name')}”移出常用；音色文件仍保留在音色库中。"


def render_voice_library_summary() -> str:
    entries = list_voice_library()
    favorite_count = len(_quick_voice_entries())
    return f"""
    <div class="voice-library-summary">
      <strong>已保存 {len(entries)} 个音色 · 常用 {favorite_count}/10</strong>
      <span>文件保存在 {html.escape(str(_voice_library_root()))}</span>
    </div>
    """


def _voice_file_digest(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _write_voice_preview(source: Path, target: Path) -> float:
    """Create the normalized 15-second preview used for listening and inference."""
    import librosa
    import numpy as np
    import soundfile as sf

    audio, _ = librosa.load(
        str(source),
        sr=VOICE_SAMPLE_RATE,
        mono=True,
        duration=30.0,
    )
    audio, _ = librosa.effects.trim(audio, top_db=35)
    if audio.size < VOICE_SAMPLE_RATE // 2:
        raise ValueError("参考音频中的有效人声不足 0.5 秒")

    audio = audio[: int(VOICE_SAMPLE_RATE * VOICE_MAX_DURATION_S)]
    fade_samples = min(int(VOICE_SAMPLE_RATE * 0.01), audio.size // 2)
    if fade_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
        audio[:fade_samples] *= fade
        audio[-fade_samples:] *= fade[::-1]

    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, VOICE_SAMPLE_RATE, subtype="PCM_16")
    return audio.size / VOICE_SAMPLE_RATE


def _store_voice_in_library(
    source: Path,
    *,
    display_name: str | None = None,
    existing_preview: Path | None = None,
    existing_conditioning: Path | None = None,
) -> tuple[dict, bool]:
    """Copy one voice into content-addressed local storage without duplicates."""
    if not source.is_file():
        raise ValueError("音色文件不存在")
    if source.suffix.lower() not in SUPPORTED_VOICE_EXTENSIONS:
        raise ValueError(f"不支持的音频格式：{source.suffix or '未知'}")
    if source.stat().st_size <= 0:
        raise ValueError("音色文件为空")
    voice_id = _voice_file_digest(source)
    root = _voice_entry_directory(voice_id)
    suffix = source.suffix.lower() or ".wav"
    stored_source = root / f"source{suffix}"
    preview_path = root / "preview.wav"
    conditioning_path = root / "conditioning_v25.npz"
    with _voice_library_lock:
        existing = _load_voice_entry(voice_id)
        if existing is not None:
            return existing, False
        root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, stored_source)
            if existing_preview and existing_preview.is_file():
                shutil.copy2(existing_preview, preview_path)
                import soundfile as sf

                duration = float(sf.info(str(preview_path)).duration)
            else:
                duration = _write_voice_preview(stored_source, preview_path)
            if existing_conditioning and existing_conditioning.is_file():
                shutil.copy2(existing_conditioning, conditioning_path)
            metadata = {
                "id": voice_id,
                "name": str(display_name or source.stem).strip() or source.stem,
                "original_filename": source.name,
                "source_filename": stored_source.name,
                "duration": duration,
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "use_count": 0,
                "last_used_at": None,
                "is_favorite": False,
                "favorite_added_at": None,
            }
            temporary = root / "metadata.tmp"
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(root / "metadata.json")
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
    entry = _load_voice_entry(voice_id)
    if entry is None:
        raise RuntimeError("音色已复制，但音色库元数据无法读取")
    return entry, True


def _activate_voice_entry(
    entry: dict | None, *, track_usage: bool = True
) -> tuple[str | None, str, str | None]:
    if entry is None:
        update_user_config(
            reference_audio=None,
            reference_conditioning=None,
            reference_cache_version=None,
            optimized_duration=None,
            voice_name=None,
            voice_library_id=None,
        )
        return None, render_voice_profile(None), None

    if track_usage:
        entry = _mark_voice_used(str(entry["id"])) or entry

    conditioning_path = Path(str(entry["conditioning_path"]))
    conditioning_ready = conditioning_path.is_file()
    update_user_config(
        reference_audio=str(entry["source_path"]),
        reference_conditioning=str(conditioning_path) if conditioning_ready else None,
        reference_cache_version=VOICE_CACHE_VERSION if conditioning_ready else None,
        optimized_duration=float(entry.get("duration") or 0.0),
        voice_name=str(entry.get("name") or entry.get("original_filename")),
        voice_library_id=str(entry["id"]),
    )
    profile = render_voice_profile(
        str(entry["source_path"]),
        str(entry.get("name") or entry.get("original_filename")),
        conditioning_ready,
        float(entry.get("duration") or 0.0),
    )
    return str(entry["source_path"]), profile, str(entry["preview_path"])


def resolve_voice_preview(reference_audio: str | None = None) -> str | None:
    """Return the playable sample that corresponds to the selected voice."""
    if reference_audio:
        config = read_user_config()
        entry = _load_voice_entry(config.get("voice_library_id"))
        if entry and _same_audio_file(Path(str(reference_audio)), Path(str(entry["source_path"]))):
            return str(entry["preview_path"])
        optimized = OPTIMIZED_VOICE_PATH
        if optimized.exists():
            return str(optimized)
        source = Path(str(reference_audio))
        return str(source) if source.exists() else None
    return None


def resolve_saved_voice_preview() -> str | None:
    """Restore the preview player together with the persisted voice."""
    return resolve_voice_preview(read_user_config().get("reference_audio"))


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "计算中"
    total_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}分{remaining_seconds:02d}秒"
    return f"{remaining_seconds}秒"


def _active_elapsed(state: dict, now: float | None = None) -> float:
    if state.get("finished_elapsed") is not None:
        return float(state["finished_elapsed"])
    started_at = state.get("started_at")
    if started_at is None:
        return 0.0
    current_time = now if now is not None else time.perf_counter()
    if state.get("paused_at") is not None:
        current_time = float(state["paused_at"])
    return max(0.0, current_time - float(started_at) - float(state["paused_seconds"]))


def render_generation_progress() -> str:
    """Render authoritative segment progress and a measured dynamic ETA."""
    with _generation_progress_lock:
        state = dict(_generation_progress_state)
    current = max(0, int(state.get("current") or 0))
    total = max(0, int(state.get("total") or 0))
    progress_state = str(state.get("state") or "idle")
    if progress_state == "completed":
        percentage = 100
    elif total:
        percentage = min(99, round(current / total * 100))
    else:
        percentage = 0
    remaining_segments = max(0, total - current)
    elapsed = _active_elapsed(state)
    remaining = None
    if progress_state == "completed":
        remaining = 0.0
    elif current > 0 and total > current:
        remaining = elapsed / current * (total - current)

    state_labels = {
        "idle": "即将生成",
        "preparing": "准备模型",
        "running": "正在生成",
        "paused": "已暂停",
        "cancelling": "正在终止",
        "cancelled": "已终止",
        "completed": "生成完成",
        "failed": "生成失败",
    }
    segment_text = f"共 {total} 个片段" if total else "片段数量将在开始后显示"
    return f"""
    <div class="generation-progress">
      <div class="generation-progress-head">
        <span class="generation-progress-title">{state_labels.get(progress_state, "生成进度")}</span>
        <span class="generation-progress-percent">{percentage}%</span>
      </div>
      <div class="generation-progress-track" role="progressbar" aria-valuemin="0"
           aria-valuemax="100" aria-valuenow="{percentage}">
        <div class="generation-progress-fill" style="width:{percentage}%"></div>
      </div>
      <div class="generation-progress-metrics">
        <div class="generation-progress-metric"><strong>{percentage}%</strong>已完成百分比</div>
        <div class="generation-progress-metric"><strong>{current}</strong>已完成片段</div>
        <div class="generation-progress-metric"><strong>{remaining_segments}</strong>剩余片段</div>
      </div>
      <div class="generation-progress-meta">
        <span>{segment_text}</span>
        <span class="generation-progress-time">已用 {_format_duration(elapsed)} · 预计剩余 {_format_duration(remaining)}</span>
      </div>
      <div class="generation-progress-message">{html.escape(str(state.get("message") or ""))}</div>
    </div>
    """


def _reset_generation_progress() -> None:
    with _generation_progress_lock:
        _generation_progress_state.update(
            state="preparing",
            current=0,
            total=0,
            message="正在准备模型并计算文本片段",
            started_at=time.perf_counter(),
            paused_at=None,
            paused_seconds=0.0,
            finished_elapsed=None,
        )


def _finish_generation_progress(state: str, message: str) -> None:
    with _generation_progress_lock:
        elapsed = _active_elapsed(_generation_progress_state)
        _generation_progress_state.update(
            state=state,
            message=message,
            paused_at=None,
            finished_elapsed=elapsed,
        )


def count_text_characters(text: str | None) -> str:
    """Render the live non-whitespace character count for synthesis text."""
    count = sum(not character.isspace() for character in (text or ""))
    limit_class = " text-counter-over-limit" if count > MAX_SYNTHESIS_CHARACTERS else ""
    return f"""
    <div class="text-entry-header">
      <span class="text-entry-label">合成文字</span>
      <span class="text-counter{limit_class}" title="单次最多 {MAX_SYNTHESIS_CHARACTERS:,} 个有效字符；不统计空格、制表符和换行">
        已输入 <strong>{count}</strong> / {MAX_SYNTHESIS_CHARACTERS:,} 字
      </span>
    </div>
    """


def _validate_synthesis_text(text: str | None) -> str:
    """Validate the complete request before safe internal batching begins."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise gr.Error("请先输入需要合成的文字。")
    character_count = count_effective_characters(cleaned_text)
    if character_count > MAX_SYNTHESIS_CHARACTERS:
        raise gr.Error(
            f"本次文本共 {character_count:,} 个有效字符，超过长文合成上限 "
            f"{MAX_SYNTHESIS_CHARACTERS:,} 字。请删减后再生成。"
        )
    return cleaned_text


def _split_synthesis_batches(text: str, max_characters: int = SYNTHESIS_BATCH_CHARACTERS) -> list[str]:
    """Split long text at natural boundaries without dropping any content."""
    if max_characters < 1:
        raise ValueError("批次字符上限必须大于 0")

    def hard_split(value: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        effective = 0
        for index, character in enumerate(value):
            if not character.isspace():
                effective += 1
            if effective >= max_characters:
                piece = value[start : index + 1]
                if piece.strip():
                    pieces.append(piece)
                start = index + 1
                effective = 0
        remainder = value[start:]
        if remainder.strip():
            pieces.append(remainder)
        return pieces

    units = re.split(r"(?<=[。！？!?；;\n])", text)
    batches: list[str] = []
    current: list[str] = []
    current_count = 0
    for unit in units:
        if not unit:
            continue
        unit_count = count_effective_characters(unit)
        unit_parts = hard_split(unit) if unit_count > max_characters else [unit]
        for part in unit_parts:
            part_count = count_effective_characters(part)
            if current and current_count + part_count > max_characters:
                batches.append("".join(current).strip())
                current = []
                current_count = 0
            current.append(part)
            current_count += part_count
    if current:
        batches.append("".join(current).strip())
    return [batch for batch in batches if batch]


def _prepare_model_batches(
    text: str,
    model: IndexTTSv25 | IndexTTSv2,
    max_text_tokens: int,
) -> list[tuple[str, int]]:
    """Ensure every disk-backed batch also stays below the model segment guard."""
    pending = _split_synthesis_batches(text)
    prepared: list[tuple[str, int]] = []
    while pending:
        batch = pending.pop(0)
        token_ids = model.tokenizer.tokenize(batch)
        segments = model.tokenizer.split_segments(
            token_ids,
            max_tokens_per_segment=max_text_tokens,
        )
        segment_count = len(segments)
        if segment_count <= min(MAX_SYNTHESIS_BATCH_SEGMENTS, MAX_GENERATION_SEGMENTS):
            prepared.append((batch, segment_count))
            continue

        character_count = count_effective_characters(batch)
        smaller_limit = max(1, character_count // 2)
        smaller_batches = _split_synthesis_batches(batch, smaller_limit)
        if len(smaller_batches) < 2:
            raise gr.Error("文本包含无法安全拆分的超长片段，请增加句号或换行后重试。")
        pending = smaller_batches + pending
    # Recursive safety splitting can leave many under-filled batches. Repack
    # adjacent text so long jobs spend less time on batch setup, QA and disk I/O.
    merged: list[tuple[str, int]] = []
    segment_limit = min(MAX_SYNTHESIS_BATCH_SEGMENTS, MAX_GENERATION_SEGMENTS)
    for batch_text, segment_count in prepared:
        if not merged:
            merged.append((batch_text, segment_count))
            continue
        combined_text = merged[-1][0] + batch_text
        combined_segments = model.tokenizer.split_segments(
            model.tokenizer.tokenize(combined_text),
            max_tokens_per_segment=max_text_tokens,
        )
        if len(combined_segments) <= segment_limit:
            merged[-1] = (combined_text, len(combined_segments))
        else:
            merged.append((batch_text, segment_count))
    return merged


def _optimize_reference_audio(source: Path) -> float:
    """Trim silence and keep a compact, useful voice sample for inference."""
    return _write_voice_preview(source, OPTIMIZED_VOICE_PATH)


def _build_voice_conditioning(
    source: Path,
    *,
    reuse_optimized: bool = False,
    optimized_duration: float | None = None,
    optimized_path: Path | None = None,
    conditioning_path: Path | None = None,
    model: IndexTTSv25 | IndexTTSv2 | None = None,
) -> tuple[str, float]:
    """Create the clean reference WAV and reusable IndexTTS-2.5 speaker context."""
    optimized_path = optimized_path or OPTIMIZED_VOICE_PATH
    conditioning_path = conditioning_path or VOICE_CONDITIONING_PATH
    if reuse_optimized and optimized_path.exists():
        import soundfile as sf

        duration = float(sf.info(str(optimized_path)).duration)
        # Earlier builds stored only five seconds.  Rebuild once so 2.5 can use
        # the longer reference window that materially improves timbre matching.
        if duration < 9.5:
            duration = _write_voice_preview(source, optimized_path)
    else:
        duration = _write_voice_preview(source, optimized_path)
    model = model or get_model()
    # Both optimized files use stable paths. Clear the in-memory path cache so
    # uploading a different voice never reuses the previous speaker features.
    model.cache = {}
    model.save_speaker(str(optimized_path), str(conditioning_path))
    if not conditioning_path.exists():
        raise RuntimeError("音色缓存文件未能建立")
    model.cache = {}
    gc.collect()
    return str(conditioning_path), duration


def _same_audio_file(first: Path, second: Path) -> bool:
    """Compare restored Gradio copies without relying on their temporary paths."""
    try:
        if first.resolve() == second.resolve():
            return True
        if first.stat().st_size != second.stat().st_size:
            return False
        digests = []
        for path in (first, second):
            digest = hashlib.sha256()
            with path.open("rb") as audio_file:
                for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests.append(digest.digest())
        return digests[0] == digests[1]
    except OSError:
        return False


def persist_voice(reference_audio: str | None) -> str:
    """Add one uploaded/recorded voice to the library and activate it."""
    if not reference_audio:
        return _activate_voice_entry(None)[1]

    source = Path(str(reference_audio))
    if not source.exists():
        return render_voice_profile(None)

    config = read_user_config()
    stored_reference = config.get("reference_audio")
    stored_path = Path(str(stored_reference)) if stored_reference else None
    cache_is_current = config.get("reference_cache_version") == VOICE_CACHE_VERSION
    conditioning = config.get("reference_conditioning")
    is_saved_voice = bool(stored_path and stored_path.exists() and _same_audio_file(source, stored_path))

    # Page reloads emit an Audio change event. Never re-copy or re-process the
    # already persisted voice during restoration.
    if is_saved_voice:
        voice_name = config.get("voice_name") or source.name
        return render_voice_profile(
            str(stored_path),
            str(voice_name),
            bool(conditioning and Path(str(conditioning)).exists() and cache_is_current),
            config.get("optimized_duration"),
        )

    try:
        entry, _added = _store_voice_in_library(source, display_name=source.stem)
    except Exception as exc:
        raise gr.Error(f"音色读取失败：{exc}") from exc
    return _activate_voice_entry(entry)[1]


def persist_voice_with_preview(reference_audio: str | None) -> tuple[str, str | None]:
    """Persist a voice and update its adjacent preview player atomically."""
    profile = persist_voice(reference_audio)
    return profile, resolve_saved_voice_preview()


def _migrate_legacy_voice_to_library() -> None:
    """Import the former single current_voice slot once without losing its cache."""
    config = read_user_config()
    if _load_voice_entry(config.get("voice_library_id")) is not None:
        return
    reference = config.get("reference_audio")
    if not reference or not Path(str(reference)).is_file():
        return
    try:
        entry, _added = _store_voice_in_library(
            Path(str(reference)),
            display_name=str(config.get("voice_name") or Path(str(reference)).stem),
            existing_preview=OPTIMIZED_VOICE_PATH if OPTIMIZED_VOICE_PATH.is_file() else None,
            existing_conditioning=(
                VOICE_CONDITIONING_PATH if VOICE_CONDITIONING_PATH.is_file() else None
            ),
        )
        _activate_voice_entry(entry)
    except Exception as exc:
        print(f"Warning: legacy voice could not be migrated into the library: {exc}")


def load_voice_library_state() -> tuple:
    """Restore the persistent library selector and its nearby preview."""
    _migrate_legacy_voice_to_library()
    config = read_user_config()
    entry = _load_voice_entry(config.get("voice_library_id"))
    if entry is None:
        entries = list_voice_library()
        entry = entries[0] if entries else None
    reference, profile, preview = _activate_voice_entry(entry, track_usage=False)
    voice_id = str(entry["id"]) if entry else None
    return (
        gr.Dropdown(choices=_voice_library_choices(), value=voice_id),
        preview,
        render_voice_library_summary(),
        reference,
        profile,
        preview,
    )


def select_voice_from_library(
    voice_id: str | None,
) -> tuple[str | None, str, str | None, str | None, str]:
    """Activate one saved voice and expose an immediately playable preview."""
    entry = _load_voice_entry(voice_id)
    if entry is None:
        raise gr.Error("选中的音色文件已不存在，请刷新音色库。")
    reference, profile, preview = _activate_voice_entry(entry)
    name = str(entry.get("name"))
    return reference, profile, preview, preview, f"已选用音色：{name}，可立即试听或生成。"


def select_quick_voice(voice_id: str | None) -> tuple:
    """Select a saved voice from one of the ten quick-access buttons."""
    reference, profile, preview, nearby_preview, status = select_voice_from_library(voice_id)
    return (
        gr.Dropdown(choices=_voice_library_choices(), value=voice_id),
        reference,
        profile,
        preview,
        nearby_preview,
        status,
    )


def delete_voice_from_library(voice_id: str | None) -> tuple:
    """Remove one voice from the active library by moving it to recoverable trash."""
    entry = _load_voice_entry(voice_id)
    if entry is None:
        config = read_user_config()
        selected = config.get("voice_library_id")
        return (
            gr.Dropdown(choices=_voice_library_choices(), value=selected),
            config.get("reference_audio"),
            render_voice_profile(
                config.get("reference_audio"),
                config.get("voice_name"),
                bool(config.get("reference_conditioning")),
                config.get("optimized_duration"),
            ),
            resolve_saved_voice_preview(),
            resolve_saved_voice_preview(),
            render_voice_library_summary(),
            "请先选择需要删除的音色。",
        )

    source_directory = _voice_entry_directory(str(entry["id"]))
    trash_root = VOICE_DIR / "trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    trash_target = trash_root / f"{entry['id']}-{time.strftime('%Y%m%d-%H%M%S')}"
    with _voice_library_lock:
        source_directory.replace(trash_target)

    remaining = list_voice_library()
    next_entry = remaining[0] if remaining else None
    reference, profile, preview = _activate_voice_entry(next_entry, track_usage=False)
    selected = str(next_entry["id"]) if next_entry else None
    name = str(entry.get("name") or entry.get("original_filename"))
    return (
        gr.Dropdown(choices=_voice_library_choices(), value=selected),
        reference,
        profile,
        preview,
        preview,
        render_voice_library_summary(),
        f"已删除音色“{name}”。原文件已移到可恢复目录：{trash_target}",
    )


def import_voice_files(
    uploaded_files: list[str] | str | None,
    fallback_audio: str | None = None,
    progress=gr.Progress(),
) -> tuple[gr.Dropdown, None, str, str]:
    """Persist many audio files locally while isolating per-file failures."""
    files = [uploaded_files] if isinstance(uploaded_files, str) else list(uploaded_files or [])
    used_single_fallback = False
    if not files and fallback_audio:
        files = [fallback_audio]
        used_single_fallback = True
    if not files:
        config = read_user_config()
        selected = config.get("voice_library_id")
        return (
            gr.Dropdown(choices=_voice_library_choices(), value=selected),
            None,
            render_voice_library_summary(),
            "尚未选择音色。请点击上方蓝色虚线框；在 Finder 中可按住 Command 或 Shift 一次选择多个文件。",
        )

    added = 0
    duplicates = 0
    errors = []
    for index, raw_path in enumerate(files):
        source = Path(str(raw_path))
        progress((index, len(files)), desc=f"正在保存 {index + 1}/{len(files)}：{source.name}")
        try:
            _entry, was_added = _store_voice_in_library(source, display_name=source.stem)
            added += int(was_added)
            duplicates += int(not was_added)
        except Exception as exc:
            errors.append(f"{source.name}：{exc}")

    config = read_user_config()
    selected = config.get("voice_library_id")
    if _load_voice_entry(selected) is None:
        selected = None
    if used_single_fallback:
        message = (
            "单个音色已保存到本机音色库。"
            if added
            else "这个音色已经保存在本机音色库中，无需重复保存。"
        )
    else:
        message = f"批量导入完成：新增 {added} 个，已存在 {duplicates} 个"
    if errors:
        message += f"，失败 {len(errors)} 个｜" + "；".join(errors[:5])
        if len(errors) > 5:
            message += "；其余失败项已省略"
    return (
        gr.Dropdown(choices=_voice_library_choices(), value=selected),
        None,
        render_voice_library_summary(),
        message,
    )


def persist_uploaded_voice_with_library(
    reference_audio: str | None,
) -> tuple[str, str | None, gr.Dropdown, str | None, str, str]:
    """Save one upload/recording, activate it, and refresh all library UI."""
    # Entering Gradio's microphone mode temporarily clears the Audio value.
    # Treat that transition as UI-only: it must never clear the active voice.
    if not reference_audio:
        config = read_user_config()
        selected = config.get("voice_library_id")
        preview = resolve_saved_voice_preview()
        return (
            render_voice_profile(
                config.get("reference_audio"),
                config.get("voice_name"),
                bool(config.get("reference_conditioning")),
                config.get("optimized_duration"),
            ),
            preview,
            gr.Dropdown(choices=_voice_library_choices(), value=selected),
            preview,
            render_voice_library_summary(),
            "录音尚未完成；当前音色保持不变。",
        )
    profile, preview = persist_voice_with_preview(reference_audio)
    config = read_user_config()
    selected = config.get("voice_library_id")
    name = str(config.get("voice_name") or "未命名音色")
    return (
        profile,
        preview,
        gr.Dropdown(choices=_voice_library_choices(), value=selected),
        preview,
        render_voice_library_summary(),
        f"音色“{name}”已保存到本机音色库并选中。",
    )


def save_user_settings(
    model_backend: str,
    emotion: str,
    emotion_strength: float,
    speed: float,
    seed: float,
    interval_silence: float,
    segment_overlap_ms: float,
    max_text_tokens: float,
    temperature: float,
    diffusion_steps: float,
    max_mel_tokens: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
    cfg_rate: float,
    fast_vocoder: bool,
    output_format: str,
    output_directory: str,
) -> None:
    """Persist all adjustable synthesis settings."""
    update_user_config(
        model_backend=model_backend if model_backend in MODEL_BACKENDS else "IndexTTS 2.5",
        model_version=(
            "2.0"
            if model_backend == "IndexTTS 2.0"
            else "OmniVoice"
            if model_backend == "OmniVoice"
            else "Fish Audio S2 Pro"
            if model_backend == "Fish Audio S2 Pro"
            else "2.5"
        ),
        emotion=emotion,
        emotion_strength=float(emotion_strength),
        speed=float(speed),
        seed=int(seed),
        interval_silence=int(interval_silence),
        segment_overlap_ms=int(segment_overlap_ms),
        max_text_tokens=int(max_text_tokens),
        temperature=float(temperature),
        diffusion_steps=int(diffusion_steps),
        max_mel_tokens=int(max_mel_tokens),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        cfg_rate=float(cfg_rate),
        fast_vocoder=bool(fast_vocoder),
        output_format=_normalise_output_format(output_format),
        output_directory=str(output_directory or OUTPUT_DIR).strip(),
    )


def load_saved_state() -> tuple:
    """Restore the last voice and all settings whenever the page loads."""
    config = read_user_config()
    reference = config.get("reference_audio")
    conditioning = config.get("reference_conditioning")
    cache_is_current = config.get("reference_cache_version") == VOICE_CACHE_VERSION
    profile = render_voice_profile(
        reference,
        config.get("voice_name"),
        bool(conditioning) and cache_is_current,
        config.get("optimized_duration"),
    )
    return (
        reference,
        profile,
        config["model_backend"],
        config["emotion"],
        config["emotion_strength"],
        config["speed"],
        config["seed"],
        config["interval_silence"],
        config["segment_overlap_ms"],
        config["max_text_tokens"],
        config["temperature"],
        config["diffusion_steps"],
        config["max_mel_tokens"],
        config["top_p"],
        config["top_k"],
        config["repetition_penalty"],
        config["cfg_rate"],
        config["fast_vocoder"],
        config["output_format"],
        config["output_directory"],
    )


def save_omnivoice_settings(
    mode: str,
    language: str,
    ref_text: str,
    instruct: str,
    duration_s: float,
    num_steps: float,
    guidance_scale: float,
    class_temperature: float,
    position_temperature: float,
    layer_penalty_factor: float,
    t_shift: float,
    ref_audio_max_duration_s: float,
) -> None:
    update_user_config(
        omnivoice_mode=str(mode),
        omnivoice_language=str(language or "None"),
        omnivoice_instruct=str(instruct or ""),
        omnivoice_duration_s=max(0.0, float(duration_s)),
        omnivoice_num_steps=int(num_steps),
        omnivoice_guidance_scale=float(guidance_scale),
        omnivoice_class_temperature=float(class_temperature),
        omnivoice_position_temperature=float(position_temperature),
        omnivoice_layer_penalty_factor=float(layer_penalty_factor),
        omnivoice_t_shift=float(t_shift),
        omnivoice_ref_audio_max_duration_s=float(ref_audio_max_duration_s),
    )


def load_omnivoice_settings() -> tuple:
    config = read_user_config()
    entry = _load_voice_entry(config.get("voice_library_id"))
    values = dict(config)
    values["omnivoice_ref_text"] = str(entry.get("omnivoice_ref_text") or "") if entry else ""
    return tuple(values[key] for key in DEFAULT_OMNIVOICE_SETTINGS)


def load_voice_omnivoice_transcript(voice_id: str | None) -> str:
    """Keep the alignment transcript attached to its matching voice."""
    entry = _load_voice_entry(voice_id)
    return str(entry.get("omnivoice_ref_text") or "") if entry else ""


def save_fish_s2_settings(
    mode: str,
    ref_text: str,
    instruct: str,
    temperature: float,
    top_p: float,
    top_k: float,
    max_tokens: float,
    chunk_length: float,
    ref_audio_max_duration_s: float,
) -> None:
    update_user_config(
        fish_mode=str(mode or "clone"),
        fish_instruct=str(instruct or ""),
        fish_temperature=float(temperature),
        fish_top_p=float(top_p),
        fish_top_k=int(top_k),
        fish_max_tokens=max(FishS2ProTTS.MIN_AUDIO_TOKENS, int(max_tokens)),
        fish_chunk_length=int(chunk_length),
        fish_ref_audio_max_duration_s=float(ref_audio_max_duration_s),
    )


def load_voice_fish_transcript(voice_id: str | None) -> str:
    """Keep Fish S2 reference text attached to the matching voice."""
    entry = _load_voice_entry(voice_id)
    return str(entry.get("fish_ref_text") or "") if entry else ""


def update_model_controls(model_backend: str) -> tuple:
    """Show only controls that affect the selected backend."""
    if model_backend == "IndexTTS 2.0":
        return gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)
    if model_backend == "OmniVoice":
        return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=False)
    if model_backend == "Fish Audio S2 Pro":
        return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="跟随参考音频", interactive=False),
    )


def reset_advanced_settings() -> tuple:
    """Return the recommended IndexTTS2 generation settings."""
    defaults = DEFAULT_SETTINGS
    update_user_config(**{key: value for key, value in defaults.items() if key != "model_backend"})
    return (
        defaults["emotion"],
        defaults["emotion_strength"],
        defaults["speed"],
        defaults["seed"],
        defaults["interval_silence"],
        defaults["segment_overlap_ms"],
        defaults["max_text_tokens"],
        defaults["temperature"],
        defaults["diffusion_steps"],
        defaults["max_mel_tokens"],
        defaults["top_p"],
        defaults["top_k"],
        defaults["repetition_penalty"],
        defaults["cfg_rate"],
        defaults["fast_vocoder"],
    )


SPEED_PRESETS = {
    "极速预览": {
        "interval_silence": 200,
        "segment_overlap_ms": 40,
        "max_text_tokens": 140,
        "temperature": 0.75,
        "diffusion_steps": 10,
        "max_mel_tokens": 1000,
        "top_p": 1.0,
        "top_k": 30,
        "repetition_penalty": 1.0,
        "cfg_rate": 0.55,
        "fast_vocoder": False,
    },
    "平衡模式": {
        "interval_silence": 230,
        "segment_overlap_ms": 50,
        "max_text_tokens": 130,
        "temperature": 0.8,
        "diffusion_steps": 16,
        "max_mel_tokens": 1300,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 8.0,
        "cfg_rate": 0.65,
        "fast_vocoder": False,
    },
    "高质量": {
        key: DEFAULT_SETTINGS[key]
        for key in (
            "interval_silence",
            "segment_overlap_ms",
            "max_text_tokens",
            "temperature",
            "diffusion_steps",
            "max_mel_tokens",
            "top_p",
            "top_k",
            "repetition_penalty",
            "cfg_rate",
            "fast_vocoder",
        )
    },
}


def apply_speed_preset(name: str) -> tuple:
    """Apply and persist a tested speed/quality parameter group."""
    preset = SPEED_PRESETS[name]
    update_user_config(**preset)
    return tuple(preset[key] for key in preset)


def apply_fast_preset() -> tuple:
    return apply_speed_preset("极速预览")


def apply_balanced_preset() -> tuple:
    return apply_speed_preset("平衡模式")


def apply_quality_preset() -> tuple:
    return apply_speed_preset("高质量")


def render_empty_document_summary() -> str:
    return """
    <div class="document-summary">
      尚未导入文档。支持 TXT、MD、DOC、DOCX、PDF、EPUB 和 MOBI；
      扫描版 PDF 会自动使用本机中文 OCR。
    </div>
    """


QUEUE_STATUS_LABELS = {
    "pending": "待确认",
    "confirmed": "已确认",
    "running": "转换中",
    "completed": "已完成",
    "failed": "失败",
    "stopped": "已终止",
}


def _normalise_document_queue(queue_data: list[dict] | None) -> list[dict]:
    queue = []
    for raw_item in queue_data or []:
        item = dict(raw_item)
        item.setdefault("id", uuid.uuid4().hex)
        item.setdefault("title", Path(str(item.get("filename") or "未命名文档")).stem)
        item.setdefault("filename", str(item["title"]))
        item.setdefault("text", "")
        item.setdefault("status", "pending")
        item.setdefault("confirmed", item.get("status") == "confirmed")
        item.setdefault("output", "")
        item.setdefault("error", "")
        queue.append(item)
    return queue


def _document_queue_choices(queue_data: list[dict] | None) -> list[tuple[str, str]]:
    queue = _normalise_document_queue(queue_data)
    choices = []
    for index, item in enumerate(queue, start=1):
        status = QUEUE_STATUS_LABELS.get(str(item.get("status")), "待确认")
        count = count_effective_characters(str(item.get("text") or ""))
        choices.append(
            (f"{index:02d}. [{status}] {item['title']} · {count:,} 字", str(item["id"]))
        )
    return choices


def render_document_queue(queue_data: list[dict] | None) -> str:
    queue = _normalise_document_queue(queue_data)
    if not queue:
        return """
        <div class="queue-empty">
          队列为空。一次选择多个文档并加入队列，然后逐份预览、编辑和确认。
        </div>
        """
    confirmed = sum(bool(item.get("confirmed")) for item in queue)
    completed = sum(item.get("status") == "completed" for item in queue)
    rows = []
    for index, item in enumerate(queue, start=1):
        status_key = str(item.get("status") or "pending")
        status_label = QUEUE_STATUS_LABELS.get(status_key, "待确认")
        count = count_effective_characters(str(item.get("text") or ""))
        detail = ""
        if item.get("output"):
            detail = f'<div class="queue-item-detail">{html.escape(Path(str(item["output"])).name)}</div>'
        elif item.get("error"):
            detail = f'<div class="queue-item-detail queue-item-error">{html.escape(str(item["error"]))}</div>'
        rows.append(
            f"""
            <div class="queue-item queue-status-{html.escape(status_key)}">
              <span class="queue-order">{index:02d}</span>
              <span class="queue-item-main">
                <strong>{html.escape(str(item['title']))}</strong>
                <small>{count:,} 字 · {html.escape(str(item.get('filename') or ''))}</small>
                {detail}
              </span>
              <span class="queue-state">{html.escape(status_label)}</span>
            </div>
            """
        )
    return f"""
    <div class="queue-overview">
      <div class="queue-overview-head">
        <strong>排队文档 {len(queue)} 份</strong>
        <span>已确认 {confirmed}/{len(queue)} · 已完成 {completed}/{len(queue)}</span>
      </div>
      <div class="queue-list">{''.join(rows)}</div>
    </div>
    """


def add_documents_to_queue(
    uploaded_files: list[str] | str | None,
    queue_data: list[dict] | None,
    progress=gr.Progress(),
) -> tuple[list[dict], gr.Dropdown, str, None, str]:
    files = [uploaded_files] if isinstance(uploaded_files, str) else list(uploaded_files or [])
    if not files:
        raise gr.Error("请先选择一个或多个文档。")
    queue = _normalise_document_queue(queue_data)
    added = 0
    failures = []
    for index, uploaded_file in enumerate(files, start=1):
        progress((index - 1, len(files)), desc=f"正在解析第 {index}/{len(files)} 份文档")
        try:
            document = import_document(uploaded_file, cache_dir=DOCUMENT_CACHE_DIR)
        except Exception as exc:
            failures.append(f"{Path(str(uploaded_file)).name}：{exc}")
            continue
        queue.append(
            {
                "id": uuid.uuid4().hex,
                "filename": document.filename,
                "title": document.title,
                "text": document.text,
                "document": document.to_dict(),
                "status": "pending",
                "confirmed": False,
                "output": "",
                "error": "",
            }
        )
        added += 1
    progress((len(files), len(files)), desc="队列文档解析完成")
    if not added:
        raise gr.Error("所选文档均未能加入队列：" + "；".join(failures))
    selected_id = str(queue[-added]["id"])
    failure_note = f"｜失败 {len(failures)} 份：{'；'.join(failures)}" if failures else ""
    return (
        queue,
        gr.Dropdown(choices=_document_queue_choices(queue), value=selected_id),
        render_document_queue(queue),
        None,
        f"已加入 {added} 份文档｜请逐份预览并确认{failure_note}",
    )


def preview_queue_document(
    queue_data: list[dict] | None,
    selected_id: str | None,
) -> tuple[str, str]:
    queue = _normalise_document_queue(queue_data)
    item = next((entry for entry in queue if entry["id"] == selected_id), None)
    if item is None:
        return "", "请从队列中选择一份文档。"
    count = count_effective_characters(str(item["text"]))
    status = QUEUE_STATUS_LABELS.get(str(item.get("status")), "待确认")
    return str(item["text"]), f"正在预览：{item['title']}｜{count:,} 字｜{status}"


def confirm_queue_document(
    queue_data: list[dict] | None,
    selected_id: str | None,
    edited_text: str,
) -> tuple[list[dict], gr.Dropdown, str, str]:
    queue = _normalise_document_queue(queue_data)
    item = next((entry for entry in queue if entry["id"] == selected_id), None)
    if item is None:
        raise gr.Error("请先选择需要确认的队列文档。")
    cleaned = _validate_synthesis_text(edited_text)
    item.update(text=cleaned, confirmed=True, status="confirmed", error="", output="")
    return (
        queue,
        gr.Dropdown(choices=_document_queue_choices(queue), value=selected_id),
        render_document_queue(queue),
        f"已确认：{item['title']}｜{count_effective_characters(cleaned):,} 字",
    )


def mark_queue_document_edited(
    queue_data: list[dict] | None,
    selected_id: str | None,
    edited_text: str,
) -> tuple[list[dict], str]:
    queue = _normalise_document_queue(queue_data)
    item = next((entry for entry in queue if entry["id"] == selected_id), None)
    if item is None or str(item.get("text") or "") == str(edited_text or ""):
        return queue, render_document_queue(queue)
    item.update(text=str(edited_text or ""), confirmed=False, status="pending", error="", output="")
    return queue, render_document_queue(queue)


def _move_queue_document(
    queue_data: list[dict] | None,
    selected_id: str | None,
    direction: int,
) -> tuple[list[dict], gr.Dropdown, str, str]:
    queue = _normalise_document_queue(queue_data)
    current = next((index for index, item in enumerate(queue) if item["id"] == selected_id), None)
    if current is None:
        raise gr.Error("请先选择队列文档。")
    target = max(0, min(len(queue) - 1, current + direction))
    if target != current:
        queue[current], queue[target] = queue[target], queue[current]
    return (
        queue,
        gr.Dropdown(choices=_document_queue_choices(queue), value=selected_id),
        render_document_queue(queue),
        "已调整队列顺序。" if target != current else "当前文档已经位于队列边界。",
    )


def move_queue_document_up(queue_data, selected_id):
    return _move_queue_document(queue_data, selected_id, -1)


def move_queue_document_down(queue_data, selected_id):
    return _move_queue_document(queue_data, selected_id, 1)


def remove_queue_document(
    queue_data: list[dict] | None,
    selected_id: str | None,
) -> tuple[list[dict], gr.Dropdown, str, str, str]:
    queue = _normalise_document_queue(queue_data)
    removed = next((item for item in queue if item["id"] == selected_id), None)
    if removed is None:
        raise gr.Error("请先选择需要移除的文档。")
    queue = [item for item in queue if item["id"] != selected_id]
    next_id = str(queue[0]["id"]) if queue else None
    next_text = str(queue[0]["text"]) if queue else ""
    return (
        queue,
        gr.Dropdown(choices=_document_queue_choices(queue), value=next_id),
        render_document_queue(queue),
        next_text,
        f"已从队列移除：{removed['title']}",
    )


def render_document_summary(document: ImportedDocument) -> str:
    audio_minutes = estimate_audio_minutes(document.character_count)
    minimum_conversion = audio_minutes * 1.2
    maximum_conversion = audio_minutes * 2.0
    conversion_note = (
        "少于 1 分钟"
        if maximum_conversion < 1
        else f"{max(1, round(minimum_conversion))}–{max(1, round(maximum_conversion))} 分钟"
    )
    warnings = list(document.warnings)
    if document.character_count > MAX_SYNTHESIS_CHARACTERS:
        warnings.append(
            f"超过长文合成上限 {MAX_SYNTHESIS_CHARACTERS:,} 字，请减少章节。"
        )
    warning_html = ""
    if warnings:
        warning_html = (
            '<div class="document-warning">' + "<br>".join(html.escape(warning) for warning in warnings) + "</div>"
        )
    ocr_note = f"· OCR {len(document.ocr_pages)} 页" if document.used_ocr else "· 文本层读取"
    return f"""
    <div class="document-summary">
      <div class="document-summary-head">
        <span class="document-summary-name" title="{html.escape(document.filename)}">
          {html.escape(document.title)} · {html.escape(document.file_type)}
        </span>
        <span class="document-summary-ready">解析完成</span>
      </div>
      <div class="document-metrics">
        <div class="document-metric"><strong>{document.character_count:,}</strong>有效字符</div>
        <div class="document-metric"><strong>{document.chinese_character_count:,}</strong>中文字</div>
        <div class="document-metric"><strong>{len(document.chapters):,}</strong>章 / 页</div>
        <div class="document-metric"><strong>{audio_minutes:.1f}</strong>预计音频分钟</div>
      </div>
      <div style="margin-top:6px">读取方式：{ocr_note}；预计本机转换约
        {conversion_note}（以当前高质量模式估算）。
      </div>
      {warning_html}
    </div>
    """


def parse_uploaded_document(
    uploaded_file: str | None,
    progress=gr.Progress(),
) -> tuple[dict | None, str, gr.Dropdown, str]:
    """Parse an uploaded document and populate chapter choices."""
    if not uploaded_file:
        return None, render_empty_document_summary(), gr.Dropdown(choices=[], value=[]), ""

    def report_progress(current: int, total: int, message: str) -> None:
        if total:
            progress((current, total), desc=message)

    try:
        document = import_document(
            uploaded_file,
            cache_dir=DOCUMENT_CACHE_DIR,
            progress_callback=report_progress,
        )
    except DocumentImportError as exc:
        raise gr.Error(f"文档解析失败：{exc}") from exc
    except Exception as exc:
        raise gr.Error(f"文档解析出现未预期错误：{exc}") from exc

    choices = [
        (
            f"{index + 1}. {chapter.title} · {chapter.character_count:,} 字",
            f"{index + 1}. {chapter.title}",
        )
        for index, chapter in enumerate(document.chapters)
    ]
    default_values = [choice[1] for choice in choices[: min(3, len(choices))]]
    preview = document.text[:1200]
    if len(document.text) > 1200:
        preview += "\n\n……（预览仅显示前 1,200 字）"
    return (
        document.to_dict(),
        render_document_summary(document),
        gr.Dropdown(choices=choices, value=default_values, multiselect=True),
        preview,
    )


def load_document_chapters(
    document_data: dict | None,
    selected_chapters: list[str] | None,
) -> tuple[str, str]:
    if not document_data:
        raise gr.Error("请先上传并完成文档解析。")
    document = ImportedDocument.from_dict(document_data)
    if not selected_chapters:
        raise gr.Error("请至少选择一个章节或页面。")
    try:
        text = select_document_text(document, selected_chapters)
    except DocumentImportError as exc:
        raise gr.Error(str(exc)) from exc
    character_count = count_effective_characters(text)
    warning = (
        "｜超过长文合成上限，请减少所选章节"
        if character_count > MAX_SYNTHESIS_CHARACTERS
        else "｜可继续编辑后生成"
    )
    return text, f"已载入 {len(selected_chapters)} 个章节｜{character_count:,} 有效字符{warning}。"


def load_entire_document(document_data: dict | None) -> tuple[str, str]:
    if not document_data:
        raise gr.Error("请先上传并完成文档解析。")
    document = ImportedDocument.from_dict(document_data)
    text = document.text
    warning = ""
    if document.character_count > MAX_SYNTHESIS_CHARACTERS:
        warning = "｜超过长文合成上限，请分章生成"
    return text, f"已载入全文｜{document.character_count:,} 有效字符{warning}"


def clear_imported_document() -> tuple[None, None, str, gr.Dropdown, str, str]:
    return (
        None,
        None,
        render_empty_document_summary(),
        gr.Dropdown(choices=[], value=[], multiselect=True),
        "",
        "",
    )


def toggle_generation_pause() -> tuple[str, str]:
    """Pause or resume the single active synthesis task."""
    with _generation_control_lock:
        if not _generation_active.is_set():
            _generation_paused.clear()
            return "暂停转换", "当前没有正在转换的任务。"
        if _generation_paused.is_set():
            _generation_paused.clear()
            with _generation_progress_lock:
                paused_at = _generation_progress_state.get("paused_at")
                if paused_at is not None and not _generation_system_paused.is_set():
                    _generation_progress_state["paused_seconds"] += time.perf_counter() - float(paused_at)
                if _generation_system_paused.is_set():
                    _generation_progress_state.update(
                        state="paused",
                        message="电脑休眠暂停仍在生效，唤醒后将自动继续",
                    )
                else:
                    _generation_progress_state.update(
                        state="running",
                        paused_at=None,
                        message="已继续转换，正从暂停位置接着运行",
                    )
            return "暂停转换", "已继续转换，正从暂停位置接着运行。"
        _generation_paused.set()
        with _generation_progress_lock:
            _generation_progress_state.update(
                state="paused",
                paused_at=time.perf_counter(),
                message="转换已暂停；已完成片段和计时状态均已保留",
            )
        return "继续转换", "已暂停。已完成的片段会保留，点击“继续转换”即可恢复。"


def _system_will_sleep() -> None:
    """Pause an active task before macOS powers down for sleep."""
    with _generation_control_lock:
        if not _generation_active.is_set() or _generation_cancelled.is_set():
            return
        _generation_system_paused.set()
        with _generation_progress_lock:
            if _generation_progress_state.get("paused_at") is None:
                _generation_progress_state["paused_at"] = time.perf_counter()
            _generation_progress_state.update(
                state="paused",
                message="电脑即将休眠：转换已自动暂停，已完成音频已保留",
            )


def _system_did_wake() -> None:
    """Resume only the system-imposed pause after macOS has fully awakened."""
    with _generation_control_lock:
        if not _generation_system_paused.is_set():
            return
        _generation_system_paused.clear()
        if not _generation_active.is_set() or _generation_cancelled.is_set():
            return
        with _generation_progress_lock:
            if _generation_paused.is_set():
                _generation_progress_state.update(
                    state="paused",
                    message="电脑已唤醒；任务仍保持手动暂停",
                )
                return
            paused_at = _generation_progress_state.get("paused_at")
            if paused_at is not None:
                _generation_progress_state["paused_seconds"] += (
                    time.perf_counter() - float(paused_at)
                )
            _generation_progress_state.update(
                state="running",
                paused_at=None,
                message="电脑已唤醒，正在自动继续原生成任务",
            )


def terminate_generation() -> tuple[str, str]:
    """Request cooperative termination of the active synthesis task."""
    with _generation_control_lock:
        if not _generation_active.is_set() and not _document_queue_active.is_set():
            return "暂停转换", "当前没有正在转换的任务。"
        if _document_queue_active.is_set():
            _document_queue_cancelled.set()
        _generation_cancelled.set()
        _generation_paused.clear()
        _generation_system_paused.clear()
        with _generation_progress_lock:
            _generation_progress_state.update(
                state="cancelling",
                paused_at=None,
                message="已收到终止请求：当前短片段结束后立即保存并停止",
            )
        if _document_queue_active.is_set():
            return "暂停转换", "正在终止文档队列……当前文档已完成的片段将保留，后续文档不再启动。"
        return "暂停转换", "正在终止任务：当前短片段完成后立即停止，已生成音频将自动保留。"


def _wait_for_generation_control() -> None:
    """Honor pause/terminate requests between non-segment preprocessing stages."""
    while _generation_paused.is_set() or _generation_system_paused.is_set():
        if _generation_cancelled.is_set():
            raise GenerationCancelled("用户已终止当前任务")
        time.sleep(0.1)
    if _generation_cancelled.is_set():
        raise GenerationCancelled("用户已终止当前任务")


def _resolve_emotion_backend(
    emotion_label: str,
) -> tuple[IndexTTSv25 | IndexTTSv2, str | None, str, str]:
    """Choose the matching model and per-model voice cache for one expression."""
    emotion = EMOTIONS.get(emotion_label, "calm")
    if emotion is None:
        return (
            get_model(),
            None,
            "conditioning_path",
            "IndexTTS 2.5 · 跟随参考音频",
        )
    return (
        get_legacy_emotion_model(),
        emotion,
        "conditioning_v2_path",
        f"IndexTTS 2.0 · {emotion_label}情绪控制",
    )


def _resolve_model_backend(
    model_backend: str,
    emotion_label: str,
) -> tuple[IndexTTSv25 | IndexTTSv2 | OmniVoiceTTS | FishS2ProTTS, str | None, str | None, str]:
    """Resolve an explicitly selected backend; never switch models implicitly."""
    if model_backend == "OmniVoice":
        return get_omnivoice_model(), None, None, "OmniVoice · 本地 MLX"
    if model_backend == "Fish Audio S2 Pro":
        return get_fish_s2_model(), None, None, "Fish Audio S2 Pro · MLX 8-bit"
    if model_backend == "IndexTTS 2.0":
        emotion = EMOTIONS.get(emotion_label)
        if emotion is None:
            emotion = "calm"
        return get_legacy_emotion_model(), emotion, "conditioning_v2_path", f"IndexTTS 2.0 · {emotion_label}"
    return get_model(), None, "conditioning_path", "IndexTTS 2.5 · 跟随参考音频"


def _analyze_backend_audio(audio, sample_rate: int, model_backend: str) -> dict:
    """Apply the spectral guard calibrated for each backend's codec/sample rate."""
    report = analyze_audio_quality(audio, sample_rate)
    if model_backend not in {"OmniVoice", "Fish Audio S2 Pro"}:
        return report
    high_frequency_issue = "检测到异常高频能量，可能存在啸叫或金属音"
    # OmniVoice's 24 kHz audio tokenizer naturally retains more 7–12 kHz
    # energy than IndexTTS's 22.05 kHz vocoder. Keep the guard for severe
    # failures while avoiding false rejection of normal codec detail.
    if report["high_frequency_mean"] <= 0.25 and high_frequency_issue in report["issues"]:
        report["issues"] = [issue for issue in report["issues"] if issue != high_frequency_issue]
        report["passed"] = not report["issues"]
    return report


def _synthesize_unlocked(
    text: str,
    voice_library_id: str | None,
    model_backend: str,
    emotion_label: str,
    emotion_strength: float,
    speed: float,
    seed: float,
    interval_silence: float,
    segment_overlap_ms: float,
    max_text_tokens: float,
    temperature: float,
    diffusion_steps: float,
    max_mel_tokens: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
    cfg_rate: float,
    fast_vocoder: bool,
    omnivoice_mode: str,
    omnivoice_language: str,
    omnivoice_ref_text: str,
    omnivoice_instruct: str,
    omnivoice_duration_s: float,
    omnivoice_num_steps: float,
    omnivoice_guidance_scale: float,
    omnivoice_class_temperature: float,
    omnivoice_position_temperature: float,
    omnivoice_layer_penalty_factor: float,
    omnivoice_t_shift: float,
    omnivoice_ref_audio_max_duration_s: float,
    output_format: str,
    output_directory: str,
    fish_mode: str = "clone",
    fish_ref_text: str = "",
    fish_instruct: str = "",
    fish_temperature: float = 0.7,
    fish_top_p: float = 0.7,
    fish_top_k: float = 30,
    fish_max_tokens: float = 1024,
    fish_chunk_length: float = 300,
    fish_ref_audio_max_duration_s: float = 15.0,
    progress=gr.Progress(),
    batch_ready_callback: Callable[[str], None] | None = None,
) -> tuple[str | None, str, str]:
    """Generate audio in the selected format and output directory."""
    cleaned_text = _validate_synthesis_text(text)

    # The selector value is the single source of truth. Never fall back to the
    # former global current_voice cache, which may belong to a previously used voice.
    library_entry = _load_voice_entry(voice_library_id)
    voice_required = (
        model_backend not in {"OmniVoice", "Fish Audio S2 Pro"}
        or (model_backend == "OmniVoice" and omnivoice_mode == "clone")
        or (model_backend == "Fish Audio S2 Pro" and fish_mode == "clone")
    )
    if library_entry is None and voice_required:
        raise gr.Error("当前选择的音色不存在，请重新选择音色后再生成。")
    reference_audio = str(library_entry["source_path"]) if library_entry else ""
    optimized_path = Path(str(library_entry["preview_path"])) if library_entry else Path()
    if library_entry:
        _activate_voice_entry(library_entry, track_usage=False)

    selected_format = _normalise_output_format(output_format)
    try:
        selected_directory = _resolve_output_directory(output_directory)
    except (OSError, ValueError) as exc:
        raise gr.Error(f"输出地址不可用：{exc}") from exc
    basename = _available_audio_basename(selected_directory, cleaned_text, selected_format)
    final_path = selected_directory / f"{basename}.{selected_format}"
    output_path = final_path if selected_format == "wav" else selected_directory / f"{basename}.wav"
    update_user_config(
        output_format=selected_format,
        output_directory=str(selected_directory),
    )

    with _generation_control_lock:
        _generation_cancelled.clear()
        _generation_paused.clear()
        _generation_system_paused.clear()
        _generation_active.set()
    _reset_generation_progress()

    quality_reports: list[dict] = []
    quality_fallback_used = False
    speed_optimization_used = False
    completed_batches: list[Path] = []
    batch_directory = selected_directory / f".{basename}.parts"

    def preserve_partial_audio() -> Path | None:
        partial_path = output_path.with_name(f"{output_path.stem}.partial.wav")
        try:
            if completed_batches:
                _concatenate_wav_batches(completed_batches, partial_path, int(interval_silence))
                return partial_path
            partial_batches = sorted(batch_directory.glob("batch_*.partial.wav"))
            if partial_batches:
                shutil.copy2(partial_batches[-1], partial_path)
                return partial_path
        except Exception:
            return None
        return partial_path if partial_path.exists() else None

    config = read_user_config()
    try:
        model, emotion, conditioning_key, backend_label = _resolve_model_backend(
            model_backend, emotion_label
        )
        use_v25_backend = model_backend == "IndexTTS 2.5"
        if model_backend in {"OmniVoice", "Fish Audio S2 Pro"}:
            # Both external MLX backends own their reference resampling. Feeding
            # the normalized 22.05 kHz IndexTTS preview loses speaker detail.
            clone_mode = omnivoice_mode if model_backend == "OmniVoice" else fish_mode
            reference = Path(reference_audio) if library_entry and clone_mode == "clone" else None
        else:
            assert library_entry is not None and conditioning_key is not None
            conditioning_path = Path(str(library_entry[conditioning_key]))
            if conditioning_path.exists():
                reference = conditioning_path
            else:
                source = Path(str(library_entry["source_path"]))
                with _generation_progress_lock:
                    _generation_progress_state.update(
                        state="preparing",
                        message=(
                            f"已锁定音色“{library_entry.get('name')}”："
                            f"正在建立 {backend_label} 专属缓存"
                        ),
                    )
                conditioning, duration = _build_voice_conditioning(
                    source,
                    reuse_optimized=optimized_path.exists(),
                    optimized_duration=config.get("optimized_duration"),
                    optimized_path=optimized_path,
                    conditioning_path=conditioning_path,
                    model=model,
                )
                _wait_for_generation_control()
                if use_v25_backend:
                    update_user_config(
                        reference_conditioning=conditioning,
                        reference_cache_version=VOICE_CACHE_VERSION,
                        optimized_duration=duration,
                    )
                    config.update(
                        reference_conditioning=conditioning,
                        reference_cache_version=VOICE_CACHE_VERSION,
                        optimized_duration=duration,
                    )
                reference = Path(conditioning)

        if reference is not None and not reference.exists():
            raise gr.Error("所选音色文件不存在，请重新选择或重新添加。")
        _wait_for_generation_control()

        save_user_settings(
            model_backend,
            emotion_label,
            emotion_strength,
            speed,
            seed,
            interval_silence,
            segment_overlap_ms,
            max_text_tokens,
            temperature,
            diffusion_steps,
            max_mel_tokens,
            top_p,
            top_k,
            repetition_penalty,
            cfg_rate,
            fast_vocoder,
            selected_format,
            str(selected_directory),
        )
        save_omnivoice_settings(
            omnivoice_mode,
            omnivoice_language,
            omnivoice_ref_text,
            omnivoice_instruct,
            omnivoice_duration_s,
            omnivoice_num_steps,
            omnivoice_guidance_scale,
            omnivoice_class_temperature,
            omnivoice_position_temperature,
            omnivoice_layer_penalty_factor,
            omnivoice_t_shift,
            omnivoice_ref_audio_max_duration_s,
        )
        save_fish_s2_settings(
            fish_mode,
            fish_ref_text,
            fish_instruct,
            fish_temperature,
            fish_top_p,
            fish_top_k,
            fish_max_tokens,
            fish_chunk_length,
            fish_ref_audio_max_duration_s,
        )
        if model_backend not in {"OmniVoice", "Fish Audio S2 Pro"}:
            # IndexTTS caches are keyed by per-voice conditioning files. Keep
            # external-model prompt caches so reference encoding/ASR runs once.
            model.cache = {}
        requested_max_text_tokens = int(max_text_tokens)
        effective_max_text_tokens = requested_max_text_tokens
        prepared_batches = _prepare_model_batches(
            cleaned_text,
            model,
            effective_max_text_tokens,
        )
        total_segments = sum(segment_count for _, segment_count in prepared_batches)
        batch_directory.mkdir(parents=True, exist_ok=False)
        completed_segment_count = 0
        live_chunk_count = 0

        for batch_index, (batch_text, batch_segment_count) in enumerate(prepared_batches, start=1):
            _wait_for_generation_control()
            batch_path = batch_directory / f"batch_{batch_index:04d}.wav"

            def report_progress(current: int, _total: int, message: str) -> None:
                global_current = completed_segment_count + max(0, int(current))
                batch_message = (
                    f"{backend_label}｜长文批次 "
                    f"{batch_index}/{len(prepared_batches)}｜{message}"
                )
                with _generation_progress_lock:
                    if _generation_system_paused.is_set():
                        progress_state = "paused"
                        progress_message = "电脑休眠中：转换已自动暂停，已完成音频已保留"
                    elif _generation_paused.is_set():
                        progress_state = "paused"
                        progress_message = "转换已暂停；已完成片段和计时状态均已保留"
                    else:
                        progress_state = "running"
                        progress_message = batch_message
                    _generation_progress_state.update(
                        state=progress_state,
                        current=global_current,
                        total=total_segments,
                        message=progress_message,
                    )
                if total_segments > 0:
                    progress((global_current, total_segments), desc=batch_message)

            def publish_audio_chunk(audio, sample_rate: int) -> None:
                nonlocal live_chunk_count
                if batch_ready_callback is None:
                    return
                quality = _analyze_backend_audio(audio, int(sample_rate), model_backend)
                if not quality["passed"]:
                    return
                import soundfile as sf

                live_chunk_count += 1
                chunk_path = batch_directory / f"live_{live_chunk_count:04d}.wav"
                sf.write(str(chunk_path), audio, int(sample_rate), subtype="PCM_16")
                batch_ready_callback(str(chunk_path))

            generate_kwargs = dict(
                text=batch_text,
                reference_audio=str(reference) if reference is not None else None,
                output_path=str(batch_path),
                emotion=emotion,
                emo_alpha=float(emotion_strength),
                speed=float(speed),
                seed=int(seed) + batch_index - 1,
                max_text_tokens_per_segment=effective_max_text_tokens,
                interval_silence=int(interval_silence),
                segment_overlap_ms=int(segment_overlap_ms),
                temperature=float(temperature),
                diffusion_steps=int(diffusion_steps),
                max_mel_tokens=int(max_mel_tokens),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                cfg_rate=float(cfg_rate),
                fast_vocoder=bool(fast_vocoder),
                progress_callback=report_progress,
                audio_chunk_callback=(publish_audio_chunk if batch_ready_callback else None),
                cancel_requested=_generation_cancelled.is_set,
                pause_requested=lambda: (
                    _generation_paused.is_set() or _generation_system_paused.is_set()
                ),
                verbose=True,
            )
            if model_backend == "OmniVoice":
                generate_kwargs.update(
                    omnivoice_mode=omnivoice_mode,
                    language=omnivoice_language,
                    ref_text=(
                        omnivoice_ref_text
                        or (str(library_entry.get("omnivoice_ref_text") or "") if library_entry else "")
                    ),
                    instruct=omnivoice_instruct,
                    duration_s=float(omnivoice_duration_s),
                    num_steps=int(omnivoice_num_steps),
                    guidance_scale=float(omnivoice_guidance_scale),
                    class_temperature=float(omnivoice_class_temperature),
                    position_temperature=float(omnivoice_position_temperature),
                    layer_penalty_factor=float(omnivoice_layer_penalty_factor),
                    t_shift=float(omnivoice_t_shift),
                    ref_audio_max_duration_s=float(omnivoice_ref_audio_max_duration_s),
                )
            elif model_backend == "Fish Audio S2 Pro":
                generate_kwargs.update(
                    fish_mode=fish_mode,
                    ref_text=(
                        fish_ref_text
                        or (str(library_entry.get("fish_ref_text") or "") if library_entry else "")
                    ),
                    instruct=fish_instruct,
                    temperature=float(fish_temperature),
                    top_p=float(fish_top_p),
                    top_k=int(fish_top_k),
                    max_tokens=int(fish_max_tokens),
                    chunk_length=int(fish_chunk_length),
                    ref_audio_max_duration_s=float(fish_ref_audio_max_duration_s),
                )
            generated_audio = model.generate(**generate_kwargs)
            if (
                model_backend == "OmniVoice"
                and omnivoice_mode == "clone"
                and library_entry
                and getattr(model, "last_reference_transcript", "")
                and str(library_entry.get("omnivoice_ref_text") or "")
                != str(model.last_reference_transcript)
            ):
                library_entry = _update_voice_metadata(
                    str(library_entry["id"]),
                    omnivoice_ref_text=str(model.last_reference_transcript),
                ) or library_entry
            if (
                model_backend == "Fish Audio S2 Pro"
                and fish_mode == "clone"
                and library_entry
                and getattr(model, "last_reference_transcript", "")
                and str(library_entry.get("fish_ref_text") or "")
                != str(model.last_reference_transcript)
            ):
                library_entry = _update_voice_metadata(
                    str(library_entry["id"]),
                    fish_ref_text=str(model.last_reference_transcript),
                ) or library_entry
            job_sample_rate = int(getattr(model, "sample_rate", VOICE_SAMPLE_RATE))
            quality_report = _analyze_backend_audio(generated_audio, job_sample_rate, model_backend)
            quality_reports.append(quality_report)
            quality_fallback_used = quality_fallback_used or model.last_quality_fallback_used
            speed_optimization_used = speed_optimization_used or bool(
                getattr(model, "last_speed_optimization_used", False)
            )
            del generated_audio
            gc.collect()
            _release_mlx_batch_memory()
            if not quality_report["passed"]:
                rejected_path = batch_path.with_name(f"{batch_path.stem}.rejected.wav")
                if batch_path.exists():
                    batch_path.replace(rejected_path)
                issue_text = "；".join(quality_report["issues"])
                raise RuntimeError(
                    f"第 {batch_index}/{len(prepared_batches)} 批音质检查未通过：{issue_text}"
                )
            completed_batches.append(batch_path)
            completed_segment_count += batch_segment_count

        with _generation_progress_lock:
            _generation_progress_state.update(
                state="running",
                current=total_segments,
                total=total_segments,
                message=f"全部 {len(prepared_batches)} 个批次已通过音质检查，正在合并音频",
            )
        _concatenate_wav_batches(completed_batches, output_path, int(interval_silence))
    except GenerationCancelled:
        _finish_generation_progress("cancelled", "任务已终止，已完成片段已尽量保留")
        partial_path = preserve_partial_audio()
        if partial_path:
            return (
                str(partial_path),
                f"任务已终止｜已保留部分音频｜{partial_path.name}",
                str(partial_path),
            )
        return None, "任务已终止｜尚未完成可保留的音频片段。", ""
    except Exception as exc:
        _finish_generation_progress("failed", f"生成失败：{exc}")
        partial_path = preserve_partial_audio()
        if partial_path:
            raise gr.Error(f"生成中断：{exc}。已保留可播放的部分音频：{partial_path}") from exc
        raise gr.Error(f"生成失败：{exc}") from exc
    finally:
        with _generation_control_lock:
            _generation_active.clear()
            _generation_paused.clear()
            _generation_system_paused.clear()
            _generation_cancelled.clear()

    if selected_format != "wav":
        with _generation_progress_lock:
            _generation_progress_state.update(
                state="running",
                message=f"音质检查通过，正在转换为 {selected_format.upper()}",
            )
        try:
            _convert_output_audio(output_path, final_path, selected_format)
        except Exception as exc:
            _finish_generation_progress("failed", f"格式转换失败：{exc}")
            raise gr.Error(f"格式转换失败。原始 WAV 已保留在：{output_path}。{exc}") from exc
        output_path.unlink()

    # Live playback needs the completed batch files until Gradio has published
    # the final preview update. The stream wrapper removes them immediately after.
    if batch_ready_callback is None:
        shutil.rmtree(batch_directory, ignore_errors=True)

    with _generation_progress_lock:
        elapsed = _active_elapsed(_generation_progress_state)
    if model_backend == "OmniVoice" and omnivoice_mode == "design":
        reference_note = "OmniVoice 音色设计"
    elif model_backend == "OmniVoice" and omnivoice_mode == "auto":
        reference_note = "OmniVoice 自动音色"
    elif model_backend == "Fish Audio S2 Pro" and fish_mode == "auto":
        reference_note = "Fish S2 Pro 自动音色"
    else:
        assert library_entry is not None
        reference_note = f"自定义音色：{library_entry.get('name') or Path(reference_audio).name}"
    if quality_fallback_used and use_v25_backend:
        fallback_note = "｜已自动拆分并重生成异常拉长片段"
    elif quality_fallback_used:
        fallback_note = "｜已自动启用高质量声码器"
    else:
        fallback_note = ""
    speed_note = "｜2.5 长文已自动使用平衡扩散步数" if speed_optimization_used else ""
    quality_note = ""
    if quality_reports:
        high_frequency_mean = sum(
            report["high_frequency_mean"] for report in quality_reports
        ) / len(quality_reports)
        quality_note = f"｜音质检查通过（高频均值 {high_frequency_mean * 100:.1f}%）"
    status = (
        f"生成完成｜{backend_label}｜{len(cleaned_text)} 字符｜耗时 {elapsed:.1f} 秒｜"
        f"{reference_note}{fallback_note}{speed_note}{quality_note}｜格式 {selected_format.upper()}｜"
        f"已保存至 {final_path}"
    )
    with _generation_progress_lock:
        total_segments = int(_generation_progress_state.get("total") or 0)
        _generation_progress_state.update(
            state="completed",
            current=total_segments,
            message=(
                f"{backend_label}｜全部 {total_segments} 个片段"
                "已生成并通过音质检查"
            ),
            finished_elapsed=elapsed,
        )
    return str(final_path), status, str(final_path)


def synthesize(*args, **kwargs) -> tuple[str | None, str, str]:
    """Run exactly one model job at a time, including after browser reconnects."""
    if not _synthesis_job_lock.acquire(blocking=False):
        raise gr.Error(
            "已有音频合成任务正在后台运行，请勿重复提交。"
            "即使页面刷新或短暂重连，原任务也会继续生成。"
        )
    sleep_prevention = _start_sleep_prevention()
    try:
        return _synthesize_unlocked(*args, **kwargs)
    finally:
        _stop_sleep_prevention(sleep_prevention)
        _synthesis_job_lock.release()


def _write_document_queue_manifest(
    queue_data: list[dict],
    output_directory: str,
    run_id: str,
    state: str,
) -> str:
    directory = _resolve_output_directory(output_directory)
    manifest_path = directory / f"文档转换队列_{run_id}.json"
    payload = {
        "run_id": run_id,
        "state": state,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": [
            {
                "order": index,
                "id": item.get("id"),
                "filename": item.get("filename"),
                "title": item.get("title"),
                "characters": count_effective_characters(str(item.get("text") or "")),
                "status": item.get("status"),
                "output": item.get("output") or "",
                "error": item.get("error") or "",
            }
            for index, item in enumerate(queue_data, start=1)
        ],
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return str(manifest_path)


def synthesize_document_queue_stream(
    queue_data: list[dict] | None,
    voice_library_id: str | None,
    model_backend: str,
    emotion_label: str,
    emotion_strength: float,
    speed: float,
    seed: float,
    interval_silence: float,
    segment_overlap_ms: float,
    max_text_tokens: float,
    temperature: float,
    diffusion_steps: float,
    max_mel_tokens: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
    cfg_rate: float,
    fast_vocoder: bool,
    omnivoice_mode: str,
    omnivoice_language: str,
    omnivoice_ref_text: str,
    omnivoice_instruct: str,
    omnivoice_duration_s: float,
    omnivoice_num_steps: float,
    omnivoice_guidance_scale: float,
    omnivoice_class_temperature: float,
    omnivoice_position_temperature: float,
    omnivoice_layer_penalty_factor: float,
    omnivoice_t_shift: float,
    omnivoice_ref_audio_max_duration_s: float,
    output_format: str,
    output_directory: str,
    fish_mode: str = "clone",
    fish_ref_text: str = "",
    fish_instruct: str = "",
    fish_temperature: float = 0.7,
    fish_top_p: float = 0.7,
    fish_top_k: float = 30,
    fish_max_tokens: float = 1024,
    fish_chunk_length: float = 300,
    fish_ref_audio_max_duration_s: float = 15.0,
):
    """Generate every confirmed document sequentially in one protected job."""
    queue = _normalise_document_queue(queue_data)
    if not queue:
        raise gr.Error("文档队列为空，请先添加文档。")
    unconfirmed = [str(item["title"]) for item in queue if not item.get("confirmed")]
    if unconfirmed:
        raise gr.Error("以下文档尚未确认：" + "、".join(unconfirmed))
    for item in queue:
        _validate_synthesis_text(str(item.get("text") or ""))
        item.update(status="confirmed", output="", error="")

    finished = threading.Event()
    state_lock = threading.RLock()
    runtime: dict[str, object] = {
        "queue": queue,
        "message": f"队列准备启动｜共 {len(queue)} 份文档",
        "audio": None,
        "location": "",
        "error": None,
        "version": 0,
    }

    synthesis_args = (
        voice_library_id,
        model_backend,
        emotion_label,
        emotion_strength,
        speed,
        seed,
        interval_silence,
        segment_overlap_ms,
        max_text_tokens,
        temperature,
        diffusion_steps,
        max_mel_tokens,
        top_p,
        top_k,
        repetition_penalty,
        cfg_rate,
        fast_vocoder,
        omnivoice_mode,
        omnivoice_language,
        omnivoice_ref_text,
        omnivoice_instruct,
        omnivoice_duration_s,
        omnivoice_num_steps,
        omnivoice_guidance_scale,
        omnivoice_class_temperature,
        omnivoice_position_temperature,
        omnivoice_layer_penalty_factor,
        omnivoice_t_shift,
        omnivoice_ref_audio_max_duration_s,
        output_format,
        output_directory,
        fish_mode,
        fish_ref_text,
        fish_instruct,
        fish_temperature,
        fish_top_p,
        fish_top_k,
        fish_max_tokens,
        fish_chunk_length,
        fish_ref_audio_max_duration_s,
    )

    def publish(message: str, *, audio: str | None = None, location: str | None = None) -> None:
        with state_lock:
            runtime["message"] = message
            if audio is not None:
                runtime["audio"] = audio
            if location is not None:
                runtime["location"] = location
            runtime["version"] = int(runtime["version"]) + 1

    def run_queue() -> None:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        if not _synthesis_job_lock.acquire(blocking=False):
            with state_lock:
                runtime["error"] = gr.Error("已有音频合成任务正在后台运行，请先等待或终止当前任务。")
            finished.set()
            return
        sleep_prevention = _start_sleep_prevention()
        _document_queue_active.set()
        _document_queue_cancelled.clear()
        try:
            _write_document_queue_manifest(queue, output_directory, run_id, "running")
            for index, item in enumerate(queue, start=1):
                if _document_queue_cancelled.is_set():
                    break
                item.update(status="running", error="", output="")
                publish(f"队列 {index}/{len(queue)}｜正在转换：{item['title']}")
                _write_document_queue_manifest(queue, output_directory, run_id, "running")
                try:
                    audio_path, item_status, location = _synthesize_unlocked(
                        str(item["text"]),
                        *synthesis_args,
                        progress=lambda *_args, **_kwargs: None,
                    )
                except BaseException as exc:
                    if _document_queue_cancelled.is_set():
                        item.update(status="stopped", error="用户终止队列")
                        publish(f"队列已终止｜当前文档：{item['title']}")
                        break
                    item.update(status="failed", error=str(exc))
                    publish(f"队列 {index}/{len(queue)}｜失败并继续下一份：{item['title']}｜{exc}")
                    _write_document_queue_manifest(queue, output_directory, run_id, "running_with_errors")
                    continue
                if _document_queue_cancelled.is_set():
                    item.update(
                        status="stopped",
                        output=audio_path or location or "",
                        error="用户终止队列；当前文档仅保留已完成片段",
                    )
                    publish(
                        f"队列已终止｜当前文档已保留部分音频：{item['title']}",
                        audio=audio_path,
                        location=location,
                    )
                    break
                item.update(status="completed", output=audio_path or location or "", error="")
                publish(
                    f"队列 {index}/{len(queue)}｜已完成：{item['title']}｜{item_status}",
                    audio=audio_path,
                    location=audio_path or location,
                )
                _write_document_queue_manifest(queue, output_directory, run_id, "running")

            if _document_queue_cancelled.is_set():
                for item in queue:
                    if item.get("status") in {"confirmed", "pending"}:
                        item["status"] = "stopped"
                final_state = "stopped"
                publish("文档队列已安全终止，后续文档未启动。")
            else:
                failed_count = sum(item.get("status") == "failed" for item in queue)
                completed_count = sum(item.get("status") == "completed" for item in queue)
                final_state = "completed_with_errors" if failed_count else "completed"
                publish(
                    f"文档队列执行完成｜成功 {completed_count} 份｜失败 {failed_count} 份"
                )
            manifest = _write_document_queue_manifest(queue, output_directory, run_id, final_state)
            publish(f"{runtime['message']}｜队列记录：{manifest}")
        except BaseException as exc:
            with state_lock:
                runtime["error"] = exc
        finally:
            _document_queue_active.clear()
            _document_queue_cancelled.clear()
            _stop_sleep_prevention(sleep_prevention)
            _synthesis_job_lock.release()
            finished.set()

    worker = threading.Thread(target=run_queue, name="indextts-document-queue", daemon=True)
    worker.start()
    last_version = -1
    while not finished.wait(0.5):
        with state_lock:
            version = int(runtime["version"])
            snapshot = [dict(item) for item in queue]
            message = str(runtime["message"])
            audio = runtime["audio"]
            location = str(runtime["location"])
        if version == last_version:
            yield gr.skip(), render_document_queue(snapshot), gr.skip(), message, render_generation_progress(), gr.skip()
        else:
            yield snapshot, render_document_queue(snapshot), audio or gr.skip(), message, render_generation_progress(), location or gr.skip()
            last_version = version

    with state_lock:
        error = runtime["error"]
        snapshot = [dict(item) for item in queue]
        message = str(runtime["message"])
        audio = runtime["audio"]
        location = str(runtime["location"])
    if error is not None:
        raise error
    yield snapshot, render_document_queue(snapshot), audio or gr.skip(), message, render_generation_progress(), location or gr.skip()


def synthesize_stream(
    text: str,
    voice_library_id: str | None,
    model_backend: str,
    emotion_label: str,
    emotion_strength: float,
    speed: float,
    seed: float,
    interval_silence: float,
    segment_overlap_ms: float,
    max_text_tokens: float,
    temperature: float,
    diffusion_steps: float,
    max_mel_tokens: float,
    top_p: float,
    top_k: float,
    repetition_penalty: float,
    cfg_rate: float,
    fast_vocoder: bool,
    omnivoice_mode: str,
    omnivoice_language: str,
    omnivoice_ref_text: str,
    omnivoice_instruct: str,
    omnivoice_duration_s: float,
    omnivoice_num_steps: float,
    omnivoice_guidance_scale: float,
    omnivoice_class_temperature: float,
    omnivoice_position_temperature: float,
    omnivoice_layer_penalty_factor: float,
    omnivoice_t_shift: float,
    omnivoice_ref_audio_max_duration_s: float,
    output_format: str,
    output_directory: str,
    live_playback: bool = False,
    fish_mode: str = "clone",
    fish_ref_text: str = "",
    fish_instruct: str = "",
    fish_temperature: float = 0.7,
    fish_top_p: float = 0.7,
    fish_top_k: float = 30,
    fish_max_tokens: float = 1024,
    fish_chunk_length: float = 300,
    fish_ref_audio_max_duration_s: float = 15.0,
):
    """Stream progress and optionally autoplay each completed audio batch."""
    finished = threading.Event()
    result: dict[str, object] = {}
    preview_lock = threading.Lock()
    preview_state: dict[str, object] = {"path": None, "version": 0}

    def publish_completed_batch(batch_path: str) -> None:
        with preview_lock:
            preview_state["path"] = batch_path
            preview_state["version"] = int(preview_state["version"]) + 1

    def run_generation() -> None:
        try:
            result["value"] = synthesize(
                text,
                voice_library_id,
                model_backend,
                emotion_label,
                emotion_strength,
                speed,
                seed,
                interval_silence,
                segment_overlap_ms,
                max_text_tokens,
                temperature,
                diffusion_steps,
                max_mel_tokens,
                top_p,
                top_k,
                repetition_penalty,
                cfg_rate,
                fast_vocoder,
                omnivoice_mode,
                omnivoice_language,
                omnivoice_ref_text,
                omnivoice_instruct,
                omnivoice_duration_s,
                omnivoice_num_steps,
                omnivoice_guidance_scale,
                omnivoice_class_temperature,
                omnivoice_position_temperature,
                omnivoice_layer_penalty_factor,
                omnivoice_t_shift,
                omnivoice_ref_audio_max_duration_s,
                output_format,
                output_directory,
                fish_mode,
                fish_ref_text,
                fish_instruct,
                fish_temperature,
                fish_top_p,
                fish_top_k,
                fish_max_tokens,
                fish_chunk_length,
                fish_ref_audio_max_duration_s,
                progress=lambda *_args, **_kwargs: None,
                batch_ready_callback=(publish_completed_batch if live_playback else None),
            )
        except BaseException as exc:
            result["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(
        target=run_generation,
        name="indextts-generation",
        daemon=True,
    )
    worker.start()

    published_preview_version = 0
    preview_reset_sent = False
    live_batch_directory: Path | None = None
    try:
        while not finished.wait(0.5):
            with _generation_progress_lock:
                message = str(_generation_progress_state.get("message") or "正在生成")
            preview_update = gr.skip()
            if not preview_reset_sent:
                preview_update = gr.Audio(
                    value=None,
                    visible=bool(live_playback),
                    autoplay=bool(live_playback),
                )
                preview_reset_sent = True
            if live_playback:
                with preview_lock:
                    preview_path = preview_state["path"]
                    preview_version = int(preview_state["version"])
                if preview_path and preview_version > published_preview_version:
                    live_batch_directory = Path(str(preview_path)).parent
                    preview_update = gr.Audio(
                        value=str(preview_path),
                        visible=True,
                        autoplay=True,
                    )
                    published_preview_version = preview_version
            yield gr.skip(), message, render_generation_progress(), gr.skip(), preview_update

        error = result.get("error")
        if error is not None:
            raise error
        audio_path, final_status, output_location = result["value"]
        preview_update = gr.skip()
        if live_playback:
            with preview_lock:
                preview_path = preview_state["path"]
                preview_version = int(preview_state["version"])
            if preview_path:
                live_batch_directory = Path(str(preview_path)).parent
            if preview_path and preview_version > published_preview_version:
                preview_update = gr.Audio(
                    value=str(preview_path),
                    visible=True,
                    autoplay=True,
                )
        else:
            preview_update = gr.Audio(value=None, visible=False, autoplay=False)
        yield (
            audio_path,
            final_status,
            render_generation_progress(),
            output_location,
            preview_update,
        )
    finally:
        if live_batch_directory is not None:
            _schedule_directory_cleanup(live_batch_directory)


def build_ui() -> gr.Blocks:
    initial_config = read_user_config()
    initial_omnivoice_ref_text = load_voice_omnivoice_transcript(
        initial_config.get("voice_library_id")
    )
    initial_fish_ref_text = load_voice_fish_transcript(initial_config.get("voice_library_id"))
    with gr.Blocks(title=f"IndexTTS 2.5 专业语音工作台 · v{APP_VERSION}") as demo:
        document_state = gr.State(value=None)
        document_queue_state = gr.State(value=[])
        with gr.Sidebar(
            label="设置",
            open=False,
            width=370,
            position="right",
            elem_classes=["settings-sidebar"],
        ):
            gr.HTML('<div class="settings-title">合成设置</div>')
            gr.HTML(
                '<div class="settings-description">IndexTTS 2.5 默认跟随参考人声的音色和表达。先选择速度档位，再按需微调。</div>'
            )
            with gr.Row(elem_classes=["preset-row"]):
                fast_preset = gr.Button("极速预览", elem_classes=["preset-button"])
                balanced_preset = gr.Button("平衡模式", elem_classes=["preset-button"])
                quality_preset = gr.Button("高质量", elem_classes=["preset-button"])
            interval_silence = gr.Number(
                label="分段间停顿（毫秒）",
                info="长文本分段之间插入的静音时长。数值越大，段落停顿越明显。",
                value=250,
                minimum=0,
                maximum=1500,
                step=10,
                precision=0,
            )
            segment_overlap_ms = gr.Number(
                label="片段交叉淡化（毫秒）",
                info="相邻音频片段的平滑衔接长度。推荐 30–80 毫秒。",
                value=50,
                minimum=0,
                maximum=300,
                step=10,
                precision=0,
            )
            max_text_tokens = gr.Number(
                label="单段最大文本 Token",
                info="控制长文本切分长度。较小更稳定，较大可能让语气更连贯。",
                value=120,
                minimum=40,
                maximum=200,
                step=10,
                precision=0,
            )
            temperature = gr.Slider(
                label="采样温度",
                info="较低更稳定，较高会增加语调变化。推荐值为 0.8。",
                minimum=0.1,
                maximum=1.2,
                value=0.8,
                step=0.05,
            )
            diffusion_steps = gr.Number(
                label="扩散步数",
                info="步数越高细节可能越丰富，但生成速度会更慢。",
                value=25,
                minimum=10,
                maximum=50,
                step=1,
                precision=0,
            )
            max_mel_tokens = gr.Number(
                label="单段最大 Mel Token",
                info="限制单个片段的最长音频长度。一般保持 1500，过小可能截断声音。",
                value=1500,
                minimum=300,
                maximum=3000,
                step=100,
                precision=0,
            )
            top_p = gr.Slider(
                label="Top-P 采样范围",
                info="控制候选声音范围。数值越低越保守，推荐值为 0.8。",
                minimum=0.1,
                maximum=1.0,
                value=0.8,
                step=0.05,
            )
            top_k = gr.Number(
                label="Top-K 候选数量",
                info="每一步保留的候选数量。较低更稳定，较高变化更多。",
                value=30,
                minimum=1,
                maximum=100,
                step=1,
                precision=0,
            )
            repetition_penalty = gr.Number(
                label="重复惩罚",
                info="降低重复音节或卡顿的概率。推荐保持 10.0。",
                value=10.0,
                minimum=1.0,
                maximum=20.0,
                step=0.5,
            )
            cfg_rate = gr.Slider(
                label="CFG 引导强度",
                info="控制模型遵循条件的强度。过高可能导致声音不自然，推荐 0.7。",
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.05,
            )
            fast_vocoder = gr.Checkbox(
                label="旧版快速声码器（2.5 不使用）",
                info="IndexTTS 2.5 始终使用高保真 BigVGAN，此项仅为旧配置兼容保留。",
                value=False,
                interactive=False,
            )
            reset_settings = gr.Button(
                "恢复推荐设置",
                elem_classes=["settings-reset"],
            )

        gr.HTML(
            """
            <header class="app-header">
              <div class="app-brand">
                <div class="app-mark">IX</div>
                <div>
                  <h1 class="app-title">IndexTTS 2.5 专业语音工作台</h1>
                  <div class="app-subtitle">真实音色克隆 · 参考表现跟随 · 全程本地处理</div>
                </div>
              </div>
              <div class="app-badges">
                <span class="app-badge">Apple MLX</span>
                <span class="app-badge">离线可用</span>
                <span class="app-badge">22.05 kHz</span>
                <button id="about-open" class="about-trigger" type="button">关于 / v0.3.3</button>
              </div>
            </header>

            <div id="about-modal" class="about-modal" aria-hidden="true">
              <section class="about-card" role="dialog" aria-modal="true" aria-labelledby="about-title">
                <div class="about-card-head">
                  <h2 id="about-title">IndexTTS WebUI · v0.3.3</h2>
                  <button id="about-close" class="about-close" type="button" aria-label="关闭">×</button>
                </div>
                <div class="about-card-body">
                  <div class="about-current">
                    <strong>当前应用版本：v0.3.3</strong><br>
                    默认使用 IndexTTS 2.5，可切换 IndexTTS 2.0、OmniVoice 与 Fish Audio S2 Pro。
                    四个大模型按需分时加载，避免同时占用统一内存。
                  </div>
                  <table class="about-table">
                    <thead><tr><th>组件</th><th>当前版本 / 规格</th><th>状态与作用</th></tr></thead>
                    <tbody>
                      <tr><td>IndexTTS 主模型</td><td><strong>2.5</strong></td><td><span class="version-installed">已安装、正在使用</span>；负责音色克隆、多语种语音生成和语速控制。</td></tr>
                      <tr><td>IndexTTS 2.0</td><td><strong>2.0</strong></td><td>在模型选择器中切换后，可使用平静、高兴、悲伤等具体情绪。</td></tr>
                      <tr><td>OmniVoice</td><td><strong>MLX bfloat16</strong></td><td>本地 24 kHz 多语种合成、音色克隆与文字音色设计；CC-BY-NC，非小米官方 MiMo。</td></tr>
                      <tr><td>Fish Audio S2 Pro</td><td><strong>MLX 8-bit · 44.1 kHz</strong></td><td>Built with Fish Audio。约 5B 参数，支持音色克隆、自动音色、多说话人和文本内情绪标签；Fish Audio Research License，仅限研究及非商业使用。</td></tr>
                      <tr><td>GPT 声学 Token 模型</td><td>GPT 2.5、持久化 8-bit</td><td>自回归解码加速；音色与声码器等保真关键模块保持 FP32。</td></tr>
                      <tr><td>S2Mel</td><td>IndexTTS 2.5 CFM / DiT</td><td>将语音 Token 转换为 Mel 频谱，保持 FP32 以避免细节损失。</td></tr>
                      <tr><td>BigVGAN 声码器</td><td>2.5 内置高保真权重、22.05 kHz</td><td>将 Mel 频谱转成 WAV 波形；保持 FP32，不使用旧版快速降质路径。</td></tr>
                      <tr><td>MLX 推理引擎</td><td>0.31.1</td><td>运行于 Apple Silicon 统一内存和 GPU。</td></tr>
                      <tr><td>PyTorch</td><td>2.10.0（仅旧 2.0 回退）</td><td>2.5 主路径为 Torch-free MLX，不调用 PyTorch。</td></tr>
                      <tr><td>文档导入 / OCR</td><td>Calibre 9.13.0 / Tesseract 5</td><td>本机读取 TXT、MD、DOC、DOCX、PDF、EPUB、MOBI；扫描 PDF 使用本机中文 OCR。</td></tr>
                      <tr><td>WebUI</td><td><strong>mlx-indextts 0.3.3</strong> + IndexTTS-2.5 MLX 0.1.1</td><td>本地网页界面；支持四模型切换、独立参数、队列、长文分段、暂停、终止、实时试听与音质检查。</td></tr>
                    </tbody>
                  </table>
                  <div class="about-changelog-title">版本变更日志</div>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.3.3</strong><span>2026-09-10 · Fish 长文进度与内存修复</span></div>
                    <ul>
                      <li>修复普通单人长文未被 MLX 内部分块、长期停在 0% 并最终触发 Metal 内存溢出的问题。</li>
                      <li>改为最多 60 字的标点优先安全段，每完成一段立即刷新百分比与剩余时间。</li>
                      <li>生成过程持续写入部分音频，安全终止时保留已完成段落。</li>
                      <li>每段后释放 MLX 临时缓存，避免长任务内存持续增长。</li>
                    </ul>
                  </section>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.3.2</strong><span>2026-09-10 · Fish S2 Pro 突停修复</span></div>
                    <ul>
                      <li>修复 256 音频 Token 造成约每 12 秒硬截断一次的问题，安全下限调整为 1024。</li>
                      <li>触及 Token 上限时自动扩大并重新生成；达到 4096 仍未结束则拒绝保存残缺音频。</li>
                      <li>长文交由 Fish 原生分块器连续生成，保留上下文并减少段落重置停顿。</li>
                      <li>已有低 Token 旧配置自动迁移，无需手工恢复默认参数。</li>
                    </ul>
                  </section>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.3.1</strong><span>2026-09-10 · 音频按文案开头命名</span></div>
                    <ul>
                      <li>生成文件默认采用文案开头前 15 个非空白字符作为文件名。</li>
                      <li>自动处理文件名非法字符；遇到同名文件时追加序号，避免覆盖已有音频。</li>
                      <li>单次合成和多文档队列遵循同一命名规则。</li>
                    </ul>
                  </section>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.3.0</strong><span>2026-09-10 · Fish Audio S2 Pro 接入</span></div>
                    <ul>
                      <li>新增 Fish Audio S2 Pro 8-bit MLX 后端，输出 44.1 kHz 音频。</li>
                      <li>支持克隆已选音色、自动音色、多说话人标签和文本内情绪控制标签。</li>
                      <li>开放温度、Top-P、Top-K、最大 Token、长文分块、风格指令及参考音频时长参数。</li>
                      <li>参考原文可由本地 Qwen3-ASR 自动识别并按音色独立缓存。</li>
                      <li>标明 Fish Audio Research License：研究和非商业使用免费，商业用途需单独授权。</li>
                    </ul>
                  </section>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.2.1</strong><span>2026-09-10 · OmniVoice 克隆质量修复</span></div>
                    <ul>
                      <li>克隆改用原始参考音频，避免二次处理造成声纹细节损失。</li>
                      <li>新增本地 Qwen3-ASR，自动转写预处理后的参考音频，完成声音与文本对齐。</li>
                      <li>对齐原文按音色独立缓存；mlx-audio 升级至 0.4.6。</li>
                      <li>明确区分“克隆已选音色”、“音色设计”和“自动音色”。</li>
                    </ul>
                  </section>
                  <section class="about-release">
                    <div class="about-release-head"><strong>v0.2.0</strong><span>2026-09-10 · OmniVoice 首次接入</span></div>
                    <ul>
                      <li>新增 IndexTTS 2.5、IndexTTS 2.0、OmniVoice 三模型显式切换。</li>
                      <li>接入本地 OmniVoice bfloat16 权重及音色克隆、设计、自动三种模式。</li>
                      <li>开放语言、扩散步数、引导强度、温度、T-Shift 等参数。</li>
                      <li>实现大模型按需加载和 OmniVoice 24 kHz 音质检查。</li>
                    </ul>
                  </section>
                  <p class="about-note">版本判定以本机实际加载的配置和权重为准，而不是以页面名称为准。更换主模型后，还需同步检查 GPT、S2Mel、声码器和音色缓存格式的兼容性。</p>
                </div>
              </section>
            </div>
            """
        )

        with gr.Row(elem_classes=["workspace"]):
            with gr.Column(scale=2, min_width=0, elem_classes=["panel", "workspace-main"]):
                gr.HTML('<div class="panel-heading">01 · 文本与音色</div>')
                with gr.Accordion(
                    "多文档排队转换 · 夜间自动执行",
                    open=False,
                    elem_classes=["document-queue"],
                ):
                    gr.HTML(
                        """
                        <div class="queue-guide">
                          <strong>使用顺序：</strong>
                          ① 一次选择多份文档并加入队列；② 在下方主文本框逐份预览、编辑；
                          ③ 点击“确认当前文档”；④ 调整顺序后启动队列。全部确认后才允许自动执行。
                        </div>
                        """
                    )
                    queue_document_files = gr.File(
                        label="选择排队文档（可一次选择多份）",
                        file_count="multiple",
                        file_types=[
                            ".txt",
                            ".md",
                            ".markdown",
                            ".doc",
                            ".docx",
                            ".pdf",
                            ".epub",
                            ".mobi",
                        ],
                        type="filepath",
                        height=94,
                        elem_classes=["queue-file-picker"],
                    )
                    add_queue_documents_button = gr.Button(
                        "解析并加入队列",
                        variant="secondary",
                    )
                    document_queue_summary = gr.HTML(render_document_queue([]))
                    queue_document_selector = gr.Dropdown(
                        label="选择一份文档进行预览与确认",
                        choices=[],
                        value=None,
                        filterable=True,
                    )
                    with gr.Row(elem_classes=["queue-order-actions"]):
                        queue_move_up_button = gr.Button("上移")
                        queue_move_down_button = gr.Button("下移")
                        queue_remove_button = gr.Button("移除")
                    with gr.Row(elem_classes=["queue-actions"]):
                        queue_confirm_button = gr.Button(
                            "确认当前文档",
                            scale=1,
                            elem_classes=["queue-confirm-action"],
                        )
                        queue_start_button = gr.Button(
                            "按顺序自动转换",
                            variant="primary",
                            scale=2,
                            elem_classes=["queue-start-action"],
                        )
                with gr.Accordion(
                    "单个文档导入 / 章节选择",
                    open=False,
                    elem_classes=["document-import"],
                ):
                    document_file = gr.File(
                        label="上传 TXT、MD、Word、PDF、EPUB 或 MOBI",
                        file_types=[
                            ".txt",
                            ".md",
                            ".markdown",
                            ".doc",
                            ".docx",
                            ".pdf",
                            ".epub",
                            ".mobi",
                        ],
                        type="filepath",
                    )
                    document_summary = gr.HTML(render_empty_document_summary())
                    chapter_selector = gr.Dropdown(
                        label="选择需要载入的章节 / 页面",
                        choices=[],
                        value=[],
                        multiselect=True,
                        filterable=True,
                    )
                    with gr.Row(elem_classes=["document-actions"]):
                        load_selected_button = gr.Button(
                            "载入选中章节",
                            elem_classes=["document-action"],
                        )
                        load_all_button = gr.Button(
                            "载入全文",
                            elem_classes=["document-action"],
                        )
                        clear_document_button = gr.Button(
                            "清空导入",
                            elem_classes=["document-action"],
                        )
                    document_preview = gr.Textbox(
                        label="解析预览",
                        interactive=False,
                        lines=3,
                        elem_classes=["document-preview"],
                    )
                text_counter = gr.HTML(count_text_characters(None))
                text = gr.Textbox(
                    label="合成文字",
                    show_label=False,
                    placeholder="请输入或粘贴中文、英文内容。编辑区会自动换行，建议按自然段检查口播节奏；长文本会自动分段生成……",
                    lines=16,
                    max_lines=28,
                    elem_classes=["text-entry"],
                )
                with gr.Accordion(
                    "本机音色库 · 批量上传与试听",
                    open=True,
                    elem_classes=["voice-library"],
                ):
                    gr.HTML(
                        """
                        <div class="voice-library-guide">
                          <strong>批量添加方法</strong>
                          ① 点击下面的蓝色虚线框；② 在 Finder 中按住 Command 或 Shift 选择多个音频；
                          ③ 点击“批量导入并永久保存”。也可以直接把多个文件一起拖进虚线框。
                        </div>
                        """
                    )
                    voice_batch_files = gr.File(
                        label="第 1 步｜点击这里，一次选择多个音色文件",
                        file_count="multiple",
                        file_types=sorted(SUPPORTED_VOICE_EXTENSIONS),
                        type="filepath",
                        height=142,
                        elem_classes=["voice-batch-files"],
                    )
                    save_voice_batch_button = gr.Button(
                        "第 2 步｜批量导入并永久保存",
                        variant="primary",
                        elem_classes=["voice-library-import"],
                    )
                    voice_library_summary = gr.HTML(render_voice_library_summary())
                    gr.HTML('<div class="quick-voice-title">常用音色快捷入口（最多 10 个）｜点击音色即可选用，点击“移出”可腾出位置</div>')
                    quick_voice_buttons = []
                    quick_voice_remove_buttons = []
                    quick_voice_states = []
                    initial_quick_voices = _quick_voice_entries()
                    for row_start in range(0, 10, 2):
                        with gr.Row(elem_classes=["quick-voice-grid"]):
                            for quick_index in range(row_start, row_start + 2):
                                quick_entry = (
                                    initial_quick_voices[quick_index]
                                    if quick_index < len(initial_quick_voices)
                                    else None
                                )
                                quick_name = (
                                    str(
                                        quick_entry.get("name")
                                        or quick_entry.get("original_filename")
                                    )
                                    if quick_entry
                                    else ""
                                )
                                quick_voice_buttons.append(
                                    gr.Button(
                                        quick_name,
                                        visible=quick_entry is not None,
                                        size="sm",
                                        scale=4,
                                        elem_classes=["quick-voice-button"],
                                    )
                                )
                                quick_voice_remove_buttons.append(
                                    gr.Button(
                                        "移出",
                                        visible=quick_entry is not None,
                                        size="sm",
                                        scale=1,
                                        elem_classes=["quick-voice-remove"],
                                    )
                                )
                                quick_voice_states.append(
                                    gr.State(str(quick_entry["id"]) if quick_entry else "")
                                )
                    gr.HTML('<div class="voice-library-divider">已保存音色｜选择后立即试听和使用</div>')
                    with gr.Row(elem_classes=["saved-voice-actions"]):
                        voice_library_selector = gr.Dropdown(
                            label="选择已保存音色",
                            choices=_voice_library_choices(),
                            value=read_user_config().get("voice_library_id"),
                            filterable=True,
                            info="选中后立即成为当前合成音色，并可直接试听。",
                            scale=85,
                            min_width=0,
                            elem_classes=["saved-voice-selector"],
                        )
                        favorite_voice_button = gr.Button(
                            "常用",
                            scale=7,
                            min_width=0,
                            elem_classes=["favorite-voice-button"],
                        )
                        delete_voice_button = gr.Button(
                            "删除",
                            variant="stop",
                            scale=8,
                            min_width=0,
                            elem_classes=["voice-delete-button"],
                        )
                    library_voice_preview = gr.Audio(
                        label="选中音色试听",
                        value=resolve_saved_voice_preview(),
                        type="filepath",
                        interactive=False,
                        elem_classes=["voice-library-preview"],
                    )
                gr.HTML(
                    '<div class="reference-intro"><strong>只添加一个音色：</strong>在下方上传或录音后会自动保存。为提高真人还原度，建议使用 10–15 秒、无背景音乐、无混响的自然人声。</div>'
                )
                reference_audio = gr.Audio(
                    label="单个音色上传 / 麦克风录音（自动保存）",
                    type="filepath",
                    sources=["upload", "microphone"],
                    format="wav",
                    editable=True,
                    elem_classes=["reference-audio"],
                )

            with gr.Column(
                scale=1,
                min_width=340,
                elem_classes=["workspace-sidebar"],
            ):
                with gr.Column(
                    elem_classes=["panel", "control-panel", "compact-control-panel"]
                ):
                    with gr.Accordion(
                        "02 · 合成参数｜点击展开设置",
                        open=False,
                        elem_classes=["compact-parameter-accordion"],
                    ):
                        model_backend = gr.Dropdown(
                            label="合成模型",
                            choices=list(MODEL_BACKENDS),
                            value=initial_config["model_backend"],
                            info="2.5 高保真克隆；2.0 可控情绪；OmniVoice 与 Fish S2 Pro 为本地 MLX 模型。",
                        )
                        emotion = gr.Dropdown(
                            label="表达方式 / 情绪",
                            choices=list(EMOTIONS),
                            value=initial_config["emotion"],
                            interactive=initial_config["model_backend"] == "IndexTTS 2.0",
                            info=(
                                "仅 IndexTTS 2.0 使用具体情绪；2.5 跟随参考音频，"
                                "OmniVoice 使用下方的音色设计参数。"
                            ),
                        )
                        with gr.Row(elem_classes=["compact-parameter-grid"]):
                            emotion_strength = gr.Number(
                                label="情绪强度",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.05,
                                info="具体情绪推荐 0.6；跟随参考音频时此值不参与计算。",
                            )
                            speed = gr.Number(
                                label="语速倍率",
                                minimum=0.75,
                                maximum=1.5,
                                value=1.0,
                                step=0.05,
                            )
                            seed = gr.Number(
                                label="随机种子",
                                value=42,
                                precision=0,
                                info="相同参数和种子便于复现相近结果。",
                            )
                        with gr.Group(visible=initial_config["model_backend"] == "OmniVoice") as omnivoice_controls:
                            gr.Markdown(
                                "**OmniVoice 本地参数**  模型权重为 CC-BY-NC（非商业），"
                                "并非小米官方 MiMo。克隆时强烈建议填写参考音频原文。"
                            )
                            omnivoice_mode = gr.Dropdown(
                                label="OmniVoice 工作模式",
                                choices=[("克隆已选音色", "clone"), ("根据文字设计音色", "design"), ("自动音色", "auto")],
                                value=initial_config["omnivoice_mode"],
                            )
                            omnivoice_language = gr.Dropdown(
                                label="语言",
                                choices=["chinese", "cantonese", "english", "japanese", "korean", "french", "german", "spanish", "None"],
                                value=initial_config["omnivoice_language"],
                                allow_custom_value=True,
                            )
                            omnivoice_ref_text = gr.Textbox(
                                label="参考音频原文（留空将自动识别并按音色缓存）",
                                value=initial_omnivoice_ref_text,
                                lines=2,
                            )
                            omnivoice_instruct = gr.Textbox(
                                label="音色设计描述（仅设计模式）",
                                value=initial_config["omnivoice_instruct"],
                                placeholder="例如：温暖、成熟的女声，语速自然，带轻微微笑",
                                lines=2,
                            )
                            with gr.Row():
                                omnivoice_duration_s = gr.Number(label="固定总时长（秒，0=自动）", value=initial_config["omnivoice_duration_s"], minimum=0, maximum=600, step=0.5)
                                omnivoice_num_steps = gr.Number(label="扩散步数", value=initial_config["omnivoice_num_steps"], minimum=8, maximum=64, step=1, precision=0)
                                omnivoice_guidance_scale = gr.Number(label="引导强度", value=initial_config["omnivoice_guidance_scale"], minimum=0, maximum=8, step=0.1)
                            with gr.Row():
                                omnivoice_class_temperature = gr.Number(label="类别温度", value=initial_config["omnivoice_class_temperature"], minimum=0, maximum=2, step=0.05)
                                omnivoice_position_temperature = gr.Number(label="位置温度", value=initial_config["omnivoice_position_temperature"], minimum=0, maximum=10, step=0.1)
                                omnivoice_layer_penalty_factor = gr.Number(label="层惩罚系数", value=initial_config["omnivoice_layer_penalty_factor"], minimum=0, maximum=10, step=0.1)
                            with gr.Row():
                                omnivoice_t_shift = gr.Number(label="T-Shift", value=initial_config["omnivoice_t_shift"], minimum=0, maximum=1, step=0.01)
                                omnivoice_ref_audio_max_duration_s = gr.Number(label="参考音频最长（秒）", value=initial_config["omnivoice_ref_audio_max_duration_s"], minimum=3, maximum=30, step=0.5)
                        with gr.Group(visible=initial_config["model_backend"] == "Fish Audio S2 Pro") as fish_s2_controls:
                            gr.Markdown(
                                "**Fish Audio S2 Pro 参数**  本机使用 8-bit MLX 权重，输出 44.1 kHz。"
                                "权重仅限研究及非商业使用；商业用途需要 Fish Audio 单独授权。"
                            )
                            fish_mode = gr.Dropdown(
                                label="Fish S2 Pro 工作模式",
                                choices=[("克隆已选音色", "clone"), ("自动音色 / 多说话人", "auto")],
                                value=initial_config["fish_mode"],
                            )
                            fish_ref_text = gr.Textbox(
                                label="参考音频原文（留空将自动识别并按音色缓存）",
                                value=initial_fish_ref_text,
                                lines=2,
                            )
                            fish_instruct = gr.Textbox(
                                label="全局风格指令（可留空）",
                                value=initial_config["fish_instruct"],
                                placeholder="例如：professional broadcast tone；文本内还可加入 [whisper]、[laughing] 等标签",
                                lines=2,
                            )
                            with gr.Row():
                                fish_temperature = gr.Number(label="采样温度", value=initial_config["fish_temperature"], minimum=0, maximum=2, step=0.05)
                                fish_top_p = gr.Number(label="Top-P", value=initial_config["fish_top_p"], minimum=0.05, maximum=1, step=0.05)
                                fish_top_k = gr.Number(label="Top-K", value=initial_config["fish_top_k"], minimum=1, maximum=200, step=1, precision=0)
                            with gr.Row():
                                fish_max_tokens = gr.Number(
                                    label="最大音频 Token",
                                    info="最低 1024；过小会在一句话尚未读完时硬截断。达到上限时程序会自动扩大并重试。",
                                    value=initial_config["fish_max_tokens"],
                                    minimum=1024,
                                    maximum=4096,
                                    step=256,
                                    precision=0,
                                )
                                fish_chunk_length = gr.Number(
                                    label="多说话人批次字节数",
                                    info="用于含 <|speaker:n|> 标签的 Fish 内部分组；普通单人长文由程序按标点安全分段。",
                                    value=initial_config["fish_chunk_length"],
                                    minimum=100,
                                    maximum=1000,
                                    step=50,
                                    precision=0,
                                )
                                fish_ref_audio_max_duration_s = gr.Number(label="参考音频最长（秒）", value=initial_config["fish_ref_audio_max_duration_s"], minimum=3, maximum=30, step=0.5)
                    with gr.Row(elem_classes=["compact-generation-controls"]):
                        generate_button = gr.Button(
                            "开始生成",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )
                        pause_button = gr.Button(
                            "暂停转换",
                            elem_classes=["pause-action"],
                        )
                        stop_button = gr.Button(
                            "终止任务",
                            elem_classes=["stop-action"],
                        )
                    live_playback = gr.Checkbox(
                        label="边生成边播放",
                        value=False,
                        info="默认关闭。开启后，每完成一个音频批次便自动播放；本次任务开始后设置保持不变。",
                        elem_classes=["live-playback-toggle"],
                    )

                with gr.Column(elem_classes=["panel", "result-panel"]):
                    gr.HTML('<div class="panel-heading">03 · 输出与状态</div>')
                    with gr.Row(elem_classes=["output-settings"]):
                        output_format = gr.Dropdown(
                            label="输出格式",
                            choices=[
                                ("WAV（无损）", "wav"),
                                ("MP3（通用）", "mp3"),
                                ("FLAC（无损压缩）", "flac"),
                            ],
                            value="wav",
                            scale=2,
                        )
                        open_output_button = gr.Button(
                            "打开文件夹",
                            scale=1,
                            elem_classes=["output-folder-action"],
                        )
                    output_directory = gr.Textbox(
                        label="输出文件夹",
                        value=str(OUTPUT_DIR),
                        placeholder="可填写绝对路径，或相对于项目目录的路径",
                        info="修改后自动保存；不存在的文件夹会在生成时创建。",
                        lines=1,
                    )
                    gr.HTML('<div class="result-section-label">本次生成</div>')
                    generation_progress = gr.HTML(render_generation_progress())
                    generation_progress_timer = gr.Timer(value=1.0, active=True)
                    output_audio = gr.Audio(
                        label="播放或下载生成结果",
                        type="filepath",
                        elem_classes=["primary-result-audio"],
                    )
                    output_location = gr.Textbox(
                        label="本次文件保存位置",
                        interactive=False,
                        lines=1,
                        elem_classes=["output-location"],
                    )
                    live_preview_audio = gr.Audio(
                        label="实时试听（已完成批次）",
                        type="filepath",
                        interactive=False,
                        autoplay=True,
                        visible=False,
                        elem_classes=["live-preview-audio"],
                    )
                    gr.HTML('<div class="result-section-label voice-details">音色与记录</div>')
                    voice_profile = gr.HTML(render_voice_profile(None))
                    voice_preview = gr.Audio(
                        label="当前音色试听",
                        value=None,
                        type="filepath",
                        interactive=False,
                        elem_classes=["voice-preview"],
                    )
                    status = gr.Textbox(
                        label="处理信息",
                        interactive=False,
                        lines=2,
                        elem_classes=["status-box"],
                    )

        document_file.change(
            fn=parse_uploaded_document,
            inputs=[document_file],
            outputs=[document_state, document_summary, chapter_selector, document_preview],
            concurrency_limit=1,
        )
        load_selected_button.click(
            fn=load_document_chapters,
            inputs=[document_state, chapter_selector],
            outputs=[text, status],
            queue=False,
        )
        load_all_button.click(
            fn=load_entire_document,
            inputs=[document_state],
            outputs=[text, status],
            queue=False,
        )
        clear_document_button.click(
            fn=clear_imported_document,
            outputs=[
                document_file,
                document_state,
                document_summary,
                chapter_selector,
                document_preview,
                text,
            ],
            queue=False,
        )

        add_queue_documents_button.click(
            fn=add_documents_to_queue,
            inputs=[queue_document_files, document_queue_state],
            outputs=[
                document_queue_state,
                queue_document_selector,
                document_queue_summary,
                queue_document_files,
                status,
            ],
            concurrency_limit=1,
        )
        queue_document_selector.change(
            fn=preview_queue_document,
            inputs=[document_queue_state, queue_document_selector],
            outputs=[text, status],
            queue=False,
        )
        text.input(
            fn=mark_queue_document_edited,
            inputs=[document_queue_state, queue_document_selector, text],
            outputs=[document_queue_state, document_queue_summary],
            queue=False,
        )
        queue_confirm_button.click(
            fn=confirm_queue_document,
            inputs=[document_queue_state, queue_document_selector, text],
            outputs=[
                document_queue_state,
                queue_document_selector,
                document_queue_summary,
                status,
            ],
            queue=False,
        )
        for queue_button, queue_function in (
            (queue_move_up_button, move_queue_document_up),
            (queue_move_down_button, move_queue_document_down),
        ):
            queue_button.click(
                fn=queue_function,
                inputs=[document_queue_state, queue_document_selector],
                outputs=[
                    document_queue_state,
                    queue_document_selector,
                    document_queue_summary,
                    status,
                ],
                queue=False,
            )
        queue_remove_button.click(
            fn=remove_queue_document,
            inputs=[document_queue_state, queue_document_selector],
            outputs=[
                document_queue_state,
                queue_document_selector,
                document_queue_summary,
                text,
                status,
            ],
            queue=False,
        )

        save_voice_batch_event = save_voice_batch_button.click(
            fn=import_voice_files,
            inputs=[voice_batch_files, reference_audio],
            outputs=[
                voice_library_selector,
                voice_batch_files,
                voice_library_summary,
                status,
            ],
            concurrency_limit=1,
        )
        save_voice_batch_event.then(
            fn=load_quick_voice_state,
            outputs=[
                *quick_voice_buttons,
                *quick_voice_remove_buttons,
                *quick_voice_states,
            ],
            queue=False,
        )

        voice_library_selector.change(
            fn=select_voice_from_library,
            inputs=[voice_library_selector],
            outputs=[
                reference_audio,
                voice_profile,
                voice_preview,
                library_voice_preview,
                status,
            ],
            queue=False,
        )
        voice_library_selector.change(
            fn=load_voice_omnivoice_transcript,
            inputs=[voice_library_selector],
            outputs=[omnivoice_ref_text],
            queue=False,
        )
        voice_library_selector.change(
            fn=load_voice_fish_transcript,
            inputs=[voice_library_selector],
            outputs=[fish_ref_text],
            queue=False,
        )

        for quick_button, quick_remove_button, quick_state in zip(
            quick_voice_buttons, quick_voice_remove_buttons, quick_voice_states
        ):
            quick_select_event = quick_button.click(
                fn=select_quick_voice,
                inputs=[quick_state],
                outputs=[
                    voice_library_selector,
                    reference_audio,
                    voice_profile,
                    voice_preview,
                    library_voice_preview,
                    status,
                ],
                queue=False,
            )
            quick_select_event.then(
                fn=load_voice_omnivoice_transcript,
                inputs=[voice_library_selector],
                outputs=[omnivoice_ref_text],
                queue=False,
            )
            quick_select_event.then(
                fn=load_voice_fish_transcript,
                inputs=[voice_library_selector],
                outputs=[fish_ref_text],
                queue=False,
            )
            remove_favorite_event = quick_remove_button.click(
                fn=remove_voice_from_favorites,
                inputs=[quick_state],
                outputs=[status],
                queue=False,
            )
            remove_favorite_event.then(
                fn=render_voice_library_summary,
                outputs=[voice_library_summary],
                queue=False,
            ).then(
                fn=load_quick_voice_state,
                outputs=[
                    *quick_voice_buttons,
                    *quick_voice_remove_buttons,
                    *quick_voice_states,
                ],
                queue=False,
            )

        favorite_voice_event = favorite_voice_button.click(
            fn=set_voice_as_favorite,
            inputs=[voice_library_selector],
            outputs=[status],
            queue=False,
        )
        favorite_voice_event.then(
            fn=render_voice_library_summary,
            outputs=[voice_library_summary],
            queue=False,
        ).then(
            fn=load_quick_voice_state,
            outputs=[
                *quick_voice_buttons,
                *quick_voice_remove_buttons,
                *quick_voice_states,
            ],
            queue=False,
        )

        delete_voice_event = delete_voice_button.click(
            fn=delete_voice_from_library,
            inputs=[voice_library_selector],
            outputs=[
                voice_library_selector,
                reference_audio,
                voice_profile,
                voice_preview,
                library_voice_preview,
                voice_library_summary,
                status,
            ],
            queue=False,
        )
        delete_voice_event.then(
            fn=load_quick_voice_state,
            outputs=[
                *quick_voice_buttons,
                *quick_voice_remove_buttons,
                *quick_voice_states,
            ],
            queue=False,
        )

        # Audio.input also fires when the user merely enters microphone mode,
        # before a recording exists. Persist only completed uploads/recordings.
        for register_completed_audio in (
            reference_audio.upload,
            reference_audio.stop_recording,
        ):
            reference_audio_event = register_completed_audio(
                fn=persist_uploaded_voice_with_library,
                inputs=[reference_audio],
                outputs=[
                    voice_profile,
                    voice_preview,
                    voice_library_selector,
                    library_voice_preview,
                    voice_library_summary,
                    status,
                ],
                concurrency_limit=1,
            )
            reference_audio_event.then(
                fn=load_quick_voice_state,
                outputs=[
                    *quick_voice_buttons,
                    *quick_voice_remove_buttons,
                    *quick_voice_states,
                ],
                queue=False,
            )
            reference_audio_event.then(
                fn=load_voice_omnivoice_transcript,
                inputs=[voice_library_selector],
                outputs=[omnivoice_ref_text],
                queue=False,
            )
            reference_audio_event.then(
                fn=load_voice_fish_transcript,
                inputs=[voice_library_selector],
                outputs=[fish_ref_text],
                queue=False,
            )

        advanced_inputs = [
            interval_silence,
            segment_overlap_ms,
            max_text_tokens,
            temperature,
            diffusion_steps,
            max_mel_tokens,
            top_p,
            top_k,
            repetition_penalty,
            cfg_rate,
            fast_vocoder,
        ]
        for preset_button, preset_function in (
            (fast_preset, apply_fast_preset),
            (balanced_preset, apply_balanced_preset),
            (quality_preset, apply_quality_preset),
        ):
            preset_button.click(
                fn=preset_function,
                outputs=advanced_inputs,
                queue=False,
            )

        reset_settings.click(
            fn=reset_advanced_settings,
            outputs=[
                emotion,
                emotion_strength,
                speed,
                seed,
                interval_silence,
                segment_overlap_ms,
                max_text_tokens,
                temperature,
                diffusion_steps,
                max_mel_tokens,
                top_p,
                top_k,
                repetition_penalty,
                cfg_rate,
                fast_vocoder,
            ],
            queue=False,
        )

        settings_inputs = [
            model_backend,
            emotion,
            emotion_strength,
            speed,
            seed,
            interval_silence,
            segment_overlap_ms,
            max_text_tokens,
            temperature,
            diffusion_steps,
            max_mel_tokens,
            top_p,
            top_k,
            repetition_penalty,
            cfg_rate,
            fast_vocoder,
            output_format,
            output_directory,
        ]
        for setting_component in settings_inputs:
            setting_component.change(
                fn=save_user_settings,
                inputs=settings_inputs,
                queue=False,
            )

        omnivoice_inputs = [
            omnivoice_mode,
            omnivoice_language,
            omnivoice_ref_text,
            omnivoice_instruct,
            omnivoice_duration_s,
            omnivoice_num_steps,
            omnivoice_guidance_scale,
            omnivoice_class_temperature,
            omnivoice_position_temperature,
            omnivoice_layer_penalty_factor,
            omnivoice_t_shift,
            omnivoice_ref_audio_max_duration_s,
        ]
        for omnivoice_component in omnivoice_inputs:
            omnivoice_component.change(
                fn=save_omnivoice_settings,
                inputs=omnivoice_inputs,
                queue=False,
            )
        fish_s2_inputs = [
            fish_mode,
            fish_ref_text,
            fish_instruct,
            fish_temperature,
            fish_top_p,
            fish_top_k,
            fish_max_tokens,
            fish_chunk_length,
            fish_ref_audio_max_duration_s,
        ]
        for fish_component in fish_s2_inputs:
            fish_component.change(
                fn=save_fish_s2_settings,
                inputs=fish_s2_inputs,
                queue=False,
            )
        model_backend.change(
            fn=update_model_controls,
            inputs=[model_backend],
            outputs=[omnivoice_controls, fish_s2_controls, emotion],
            queue=False,
        )

        demo.load(
            fn=load_saved_state,
            outputs=[reference_audio, voice_profile, *settings_inputs],
            queue=False,
        )
        demo.load(
            fn=resolve_saved_voice_preview,
            outputs=[voice_preview],
            queue=False,
        )
        demo.load(
            fn=load_voice_library_state,
            outputs=[
                voice_library_selector,
                library_voice_preview,
                voice_library_summary,
                reference_audio,
                voice_profile,
                voice_preview,
            ],
            queue=False,
        )
        demo.load(
            fn=load_quick_voice_state,
            outputs=[
                *quick_voice_buttons,
                *quick_voice_remove_buttons,
                *quick_voice_states,
            ],
            queue=False,
        )
        demo.load(
            fn=render_generation_progress,
            outputs=[generation_progress],
            queue=False,
        )
        generation_progress_timer.tick(
            fn=render_generation_progress,
            outputs=[generation_progress],
            queue=False,
        )

        open_output_button.click(
            fn=open_output_directory,
            inputs=[output_directory],
            outputs=[status],
            queue=False,
        )

        demo.load(
            fn=None,
            js=r"""
            () => {
              const bindCounter = () => {
                const textarea = document.querySelector(".text-entry textarea");
                const number = document.querySelector(".text-counter strong");
                const counter = document.querySelector(".text-counter");
                if (!textarea || !number || !counter) return;

                const updateCounter = () => {
                  const count = Array.from(textarea.value || "").filter(
                    (character) => !/\s/u.test(character)
                  ).length;
                  number.textContent = count.toString();
                  counter.classList.toggle("text-counter-over-limit", count > 100000);
                };

                if (textarea.dataset.charCounterBound !== "1") {
                  textarea.addEventListener("input", updateCounter);
                  textarea.dataset.charCounterBound = "1";
                }
                updateCounter();
              };

              const bindAbout = () => {
                const openButton = document.getElementById("about-open");
                const closeButton = document.getElementById("about-close");
                const modal = document.getElementById("about-modal");
                if (!openButton || !closeButton || !modal) return;

                const openModal = () => {
                  modal.classList.add("is-open");
                  modal.setAttribute("aria-hidden", "false");
                };
                const closeModal = () => {
                  modal.classList.remove("is-open");
                  modal.setAttribute("aria-hidden", "true");
                };

                if (openButton.dataset.aboutBound !== "1") {
                  openButton.addEventListener("click", openModal);
                  closeButton.addEventListener("click", closeModal);
                  modal.addEventListener("click", (event) => {
                    if (event.target === modal) closeModal();
                  });
                  document.addEventListener("keydown", (event) => {
                    if (event.key === "Escape") closeModal();
                  });
                  openButton.dataset.aboutBound = "1";
                }
              };

              const bindPresets = () => {
                const buttons = Array.from(document.querySelectorAll(".preset-button"));
                if (buttons.length !== 3) return;
                const storageKey = "indextts2-selected-preset-v2";
                const validNames = ["极速预览", "平衡模式", "高质量"];
                let selected = window.localStorage.getItem(storageKey) || "高质量";
                if (!validNames.includes(selected)) selected = "高质量";

                const selectButton = (name) => {
                  buttons.forEach((button) => {
                    button.classList.toggle(
                      "preset-selected",
                      (button.textContent || "").trim() === name
                    );
                  });
                  window.localStorage.setItem(storageKey, name);
                };

                buttons.forEach((button) => {
                  if (button.dataset.presetBound === "1") return;
                  button.addEventListener("click", () => {
                    selectButton((button.textContent || "").trim());
                  });
                  button.dataset.presetBound = "1";
                });

                const resetButton = document.querySelector(".settings-reset");
                if (resetButton && resetButton.dataset.presetBound !== "1") {
                  resetButton.addEventListener("click", () => selectButton("高质量"));
                  resetButton.dataset.presetBound = "1";
                }
                selectButton(selected);
              };

              bindCounter();
              bindAbout();
              bindPresets();
              window.setInterval(() => {
                bindCounter();
                bindAbout();
                bindPresets();
              }, 500);
            }
            """,
            queue=False,
        )

        pause_button.click(
            fn=toggle_generation_pause,
            outputs=[pause_button, status],
            queue=False,
        )
        stop_button.click(
            fn=terminate_generation,
            outputs=[pause_button, status],
            queue=False,
        )

        generate_button.click(
            fn=synthesize_stream,
            inputs=[
                text,
                voice_library_selector,
                model_backend,
                emotion,
                emotion_strength,
                speed,
                seed,
                interval_silence,
                segment_overlap_ms,
                max_text_tokens,
                temperature,
                diffusion_steps,
                max_mel_tokens,
                top_p,
                top_k,
                repetition_penalty,
                cfg_rate,
                fast_vocoder,
                *omnivoice_inputs,
                output_format,
                output_directory,
                live_playback,
                *fish_s2_inputs,
            ],
            outputs=[
                output_audio,
                status,
                generation_progress,
                output_location,
                live_preview_audio,
            ],
            concurrency_limit=1,
            concurrency_id="tts_generation",
        )

        queue_start_button.click(
            fn=synthesize_document_queue_stream,
            inputs=[
                document_queue_state,
                voice_library_selector,
                model_backend,
                emotion,
                emotion_strength,
                speed,
                seed,
                interval_silence,
                segment_overlap_ms,
                max_text_tokens,
                temperature,
                diffusion_steps,
                max_mel_tokens,
                top_p,
                top_k,
                repetition_penalty,
                cfg_rate,
                fast_vocoder,
                *omnivoice_inputs,
                output_format,
                output_directory,
                *fish_s2_inputs,
            ],
            outputs=[
                document_queue_state,
                document_queue_summary,
                output_audio,
                status,
                generation_progress,
                output_location,
            ],
            concurrency_limit=1,
            concurrency_id="tts_generation",
        )

    return demo


def main() -> None:
    global _power_monitor
    _power_monitor = start_macos_power_monitor(_system_will_sleep, _system_did_wake)
    if _power_monitor is None:
        print("Warning: macOS sleep/wake monitoring is unavailable.")
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
