from pathlib import Path
import runpy

from scripts.vaequantumhugface_imagenet import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "vaequantumhugface_imagenet.py"),
        run_name="__main__",
    )
