#!/bin/zsh

set -u

APP_DIR="${0:A:h}"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
LAUNCH_HELPER="$APP_DIR/launcher-assets/launch_webui.py"

if [[ ! -x "$PYTHON_BIN" || ! -f "$LAUNCH_HELPER" ]]; then
  /usr/bin/osascript -e 'display alert "找不到 IndexTTS2" message "程序目录、Python 环境或启动组件缺失。" as critical'
  exit 1
fi

exec "$PYTHON_BIN" "$LAUNCH_HELPER"
