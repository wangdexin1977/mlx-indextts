"""Keep the local IndexTTS2 WebUI available after a native MLX/Metal crash."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = APP_DIR / "outputs" / "webui" / "logs"
PID_FILE = LOG_DIR / "server.pid"
CHILD_PID_FILE = LOG_DIR / "server-child.pid"
MAX_CRASHES = 3
CRASH_WINDOW_SECONDS = 10 * 60
RESTART_DELAY_SECONDS = 2

_stopping = False
_child: subprocess.Popen | None = None


def log(message: str) -> None:
    print(f"[supervisor {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def stop(_signum: int, _frame: object) -> None:
    global _stopping
    _stopping = True
    if _child is not None and _child.poll() is None:
        _child.terminate()


def remove_own_pid_file() -> None:
    try:
        if int(PID_FILE.read_text(encoding="utf-8").strip()) == os.getpid():
            PID_FILE.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    CHILD_PID_FILE.unlink(missing_ok=True)


def main() -> int:
    global _child
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    crash_times: list[float] = []

    try:
        while not _stopping:
            log("正在启动 WebUI 服务")
            _child = subprocess.Popen(
                [sys.executable, "-m", "mlx_indextts.webui"],
                cwd=APP_DIR,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
            )
            CHILD_PID_FILE.write_text(str(_child.pid), encoding="utf-8")
            return_code = _child.wait()
            CHILD_PID_FILE.unlink(missing_ok=True)
            if _stopping or return_code == 0:
                log(f"WebUI 已停止（退出码 {return_code}）")
                return return_code

            now = time.monotonic()
            crash_times = [stamp for stamp in crash_times if now - stamp < CRASH_WINDOW_SECONDS]
            crash_times.append(now)
            log(f"检测到 WebUI 异常退出（退出码 {return_code}），准备自动恢复")
            if len(crash_times) > MAX_CRASHES:
                log("10 分钟内异常退出次数过多，停止自动重启；请检查日志")
                return return_code or 1
            time.sleep(RESTART_DELAY_SECONDS)
        return 0
    finally:
        remove_own_pid_file()


if __name__ == "__main__":
    raise SystemExit(main())
