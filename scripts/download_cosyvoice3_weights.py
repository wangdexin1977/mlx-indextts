"""Resume and verify the four large official CosyVoice3 files on flaky links."""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Fun-CosyVoice3-0.5B-2512"
REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
CHUNK_SIZE = 8 * 1024 * 1024
FILES = {
    "CosyVoice-BlankEN/model.safetensors": (988097824, "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96"),
    "flow.pt": (1329116148, "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b"),
    "llm.pt": (2024669519, "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970"),
    "speech_tokenizer_v3.onnx": (969451503, "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d"),
}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_chunk(relative: str, index: int, size: int) -> tuple[str, int]:
    first = index * CHUNK_SIZE
    last = min(size, first + CHUNK_SIZE) - 1
    expected = last - first + 1
    part_dir = MODEL_DIR / ".download-parts" / relative
    part_dir.mkdir(parents=True, exist_ok=True)
    part = part_dir / f"{index:05d}.part"
    if part.is_file() and part.stat().st_size == expected:
        return relative, index
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{relative}"
    temporary = part.with_suffix(".tmp")
    for attempt in range(30):
        received = temporary.stat().st_size if temporary.is_file() else 0
        if received > expected:
            temporary.unlink()
            received = 0
        if received == expected:
            temporary.replace(part)
            return relative, index
        request_start = first + received
        try:
            with requests.get(
                url, headers={"Range": f"bytes={request_start}-{last}"},
                stream=True, allow_redirects=True, timeout=(20, 45),
            ) as response:
                response.raise_for_status()
                wanted_range = f"bytes {request_start}-{last}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != wanted_range:
                    raise RuntimeError(f"unexpected HTTP range: {response.status_code} {response.headers.get('Content-Range')}")
                with temporary.open("ab") as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
                if temporary.stat().st_size != expected:
                    raise RuntimeError(f"short chunk: {temporary.stat().st_size}/{expected}")
            temporary.replace(part)
            return relative, index
        except (OSError, requests.RequestException, RuntimeError) as exc:
            if attempt == 29:
                raise RuntimeError(f"{relative} chunk {index} failed after retries: {exc}") from exc
            time.sleep(min(10, 1 + attempt))
    raise AssertionError("unreachable")


def assemble(relative: str, size: int, sha256: str) -> None:
    target = MODEL_DIR / relative
    if target.is_file() and target.stat().st_size == size and digest_file(target) == sha256:
        print(f"verified {relative}", flush=True)
        return
    part_dir = MODEL_DIR / ".download-parts" / relative
    temporary = target.with_suffix(target.suffix + ".verified-tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with temporary.open("wb") as output:
        for index in range((size + CHUNK_SIZE - 1) // CHUNK_SIZE):
            part = part_dir / f"{index:05d}.part"
            with part.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    written += len(block)
    if written != size or digest.hexdigest() != sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 verification failed for {relative}")
    temporary.replace(target)
    for part in part_dir.glob("*.part"):
        part.unlink()
    part_dir.rmdir()
    print(f"verified {relative} ({size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [
        (relative, index, size)
        for relative, (size, sha256) in FILES.items()
        if not (MODEL_DIR / relative).is_file()
        or (MODEL_DIR / relative).stat().st_size != size
        or digest_file(MODEL_DIR / relative) != sha256
        for index in range((size + CHUNK_SIZE - 1) // CHUNK_SIZE)
    ]
    print(f"Downloading {len(jobs)} chunks with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(download_chunk, *job) for job in jobs]
        for completed, future in enumerate(as_completed(pending), 1):
            future.result()
            if completed % 10 == 0 or completed == len(jobs):
                print(f"Downloaded {completed}/{len(jobs)} chunks", flush=True)
    for relative, (size, sha256) in FILES.items():
        assemble(relative, size, sha256)


if __name__ == "__main__":
    main()
