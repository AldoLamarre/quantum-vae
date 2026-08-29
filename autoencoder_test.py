from pathlib import Path
import runpy

from scripts.autoencoder_test import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "autoencoder_test.py"),
        run_name="__main__",
    )
