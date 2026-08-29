from pathlib import Path
import runpy

from scripts.vaequantumhugface import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "vaequantumhugface.py"),
        run_name="__main__",
    )
