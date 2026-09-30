"""Small, persistent JSON-line bridge to the local CosyVoice3 macOS runtime."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path


def reply(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main() -> None:
    from tts_worker import ModelHost

    config = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        host = ModelHost(config)
        host.load()
    reply({"ok": True, "load_seconds": host.load_seconds, "sample_rate": host.sample_rate})
    speakers = {(config["prompt_wav"], config["prompt_text"]): host.spk_id}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            key = (request["prompt_wav"], request["prompt_text"])
            speaker = speakers.get(key)
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                if speaker is None:
                    speaker = "voice_" + hashlib.sha256("\0".join(key).encode()).hexdigest()[:20]
                    host.cv.add_zero_shot_spk(key[1], key[0], speaker)
                    speakers[key] = speaker
                host.spk_id = speaker
                wav, duration = host.synthesize_wav(
                    {"text": request["text"], "mode": "zero_shot",
                     "speed": request["speed"], "instruct_text": ""},
                    request.get("seed"),
                )
            Path(request["output_path"]).write_bytes(wav)
            from mac_memory import self_footprint_bytes

            reply({"ok": True, "duration": duration,
                   "elapsed": time.monotonic() - started,
                   "footprint_bytes": self_footprint_bytes()})
        except Exception as exc:
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)
        reply({"ok": False, "error": str(exc)})
