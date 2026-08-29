from pathlib import Path
import runpy

from scripts.variationalautoencodertestpennylane import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "variationalautoencodertestpennylane.py"),
        run_name="__main__",
    )
