from pathlib import Path
import runpy

from scripts.MNIST3v5classicalquantumvae import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "MNIST3v5classicalquantumvae.py"),
        run_name="__main__",
    )
