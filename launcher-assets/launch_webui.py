"""Reliable Finder launcher for the local IndexTTS2 WebUI."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
APP_URL = "http://127.0.0.1:7860/"
HEALTH_URL = f"{APP_URL}gradio_api/info"
LOG_DIR = APP_DIR / "outputs" / "webui" / "logs"
LOG_FILE = LOG_DIR / "server.log"
PID_FILE = LOG_DIR / "server.pid"
SUPERVISOR = APP_DIR / "launcher-assets" / "server_supervisor.py"
PYTHON_BIN = APP_DIR / ".venv" / "bin" / "python"
MODEL_DIR = APP_DIR / "models" / "mlx-IndexTTS-2"
HF_CACHE = Path.home() / ".cache" / "huggingface"


def show_error(message: str) -> None:
    script = 'display alert "IndexTTS2 启动失败" message {} as critical'.format(
        repr(message)
    )
    subprocess.run(["/usr/bin/osascript", "-e", script], check=False)


def is_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            return b"IndexTTS2" in response.read()
    except (OSError, urllib.error.URLError):
        return False


def port_is_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", 7860)) == 0


def open_webui() -> None:
    subprocess.run(["/usr/bin/open", APP_URL], check=False)


def process_is_running(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def wait_until_ready(seconds: int) -> bool:
    for _attempt in range(seconds):
        if is_ready():
            return True
        time.sleep(1)
    return False


def main() -> int:
    if is_ready():
        open_webui()
        return 0

    if process_is_running(PID_FILE):
        if wait_until_ready(90):
            open_webui()
            return 0
        show_error("IndexTTS2 正在恢复，但 90 秒内尚未就绪。请查看 outputs/webui/logs/server.log。")
        return 1

    if port_is_in_use():
        show_error("端口 7860 已被其他程序占用，请关闭该程序后重试。")
        return 1

    if not PYTHON_BIN.is_file() or not os.access(PYTHON_BIN, os.X_OK):
        show_error("IndexTTS2 的 Python 环境不存在或已经损坏。")
        return 1
    if not SUPERVISOR.is_file():
        show_error("IndexTTS2 的服务守护组件缺失。")
        return 1
    if not MODEL_DIR.is_dir():
        show_error("没有找到 IndexTTS2 模型文件，无法启动。")
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(HF_CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    with LOG_FILE.open("ab", buffering=0) as log_file:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n===== IndexTTS2 launcher: {stamp} =====\n".encode())
        process = subprocess.Popen(
            [str(PYTHON_BIN), str(SUPERVISOR)],
            cwd=APP_DIR,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        PID_FILE.write_text(str(process.pid), encoding="utf-8")

    for _attempt in range(90):
        if is_ready():
            open_webui()
            return 0
        if process.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            show_error(
                "服务进程提前退出。详细原因已记录到 "
                "outputs/webui/logs/server.log。"
            )
            return 1
        time.sleep(1)

    process.terminate()
    show_error("等待服务超过 90 秒，请查看 outputs/webui/logs/server.log。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
