from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parent
            / "scripts"
            / "vaequantumhugface_mnist_pretraining_data_reupload11.py"
        ),
        run_name="__main__",
    )
