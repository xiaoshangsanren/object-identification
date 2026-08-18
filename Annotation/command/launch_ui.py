from __future__ import annotations

import multiprocessing
import os
import runpy
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _open_when_ready(url: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(1)


def main() -> None:
    root = Path(os.environ["PACKAGE_ROOT"]).resolve()
    port_text = os.environ.get("PORT", "7860")
    if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise SystemExit("PORT必须是1到65535之间的整数。")
    gpu_text = os.environ.get("GPU_ID", "0")
    if not gpu_text.isdigit():
        raise SystemExit("GPU_ID必须是非负整数。")

    url = f"http://127.0.0.1:{port_text}/"
    print(f"Package: {root}")
    print(f"GPU: {gpu_text}")
    print(f"Local URL: {url}")
    print("LAN URL: http://<本机IPv4地址>:" + port_text + "/")
    print("按 Ctrl+C 或关闭此窗口可停止服务。")

    if os.environ.get("TVS_NO_BROWSER") != "1":
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    app_path = root / "app" / "app.py"
    sys.path.insert(0, str(app_path.parent))
    sys.argv = [str(app_path), "--host", "0.0.0.0", "--port", port_text]
    runpy.run_path(str(app_path), run_name="__main__")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
