"""Streamlit entrypoint.

Use:
    streamlit run app.py
"""

from pathlib import Path
import runpy
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

API_HEALTH_URL = "http://127.0.0.1:8000/table-metadata"


def api_is_up(timeout: float = 1.0) -> bool:
    try:
        with urlopen(API_HEALTH_URL, timeout=timeout) as resp:
            return resp.status == 200
    except URLError:
        return False
    except Exception:
        return False


def ensure_api_running() -> None:
    if api_is_up():
        return

    app_dir = Path(__file__).resolve().parent
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(app_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    for _ in range(20):
        if api_is_up():
            break
        time.sleep(0.25)


ensure_api_running()

# Run frontend.py on every Streamlit rerun (button clicks included).
runpy.run_path(str(Path(__file__).with_name("frontend.py")), run_name="__main__")
