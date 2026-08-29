from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parent
            / "scripts"
            / "vaequantumhugface_cifar_pretraining_data_reupload.py"
        ),
        run_name="__main__",
    )
