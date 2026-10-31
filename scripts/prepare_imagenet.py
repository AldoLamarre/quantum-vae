"""Resize ImageNet-1k to a fixed short side and save it as an Arrow dataset.

Reads the Parquet shards downloaded from the Hugging Face hub
(ILSVRC/imagenet-1k), resizes every image so its short side equals --size
(aspect ratio kept, small images upscaled), re-encodes it as JPEG and
writes a DatasetDict with the splits train / validation / test. Training
loads it with data.imagenet_path.

Usage:
  python scripts/prepare_imagenet.py --src /mnt/hdd/datasets/imagenet-1k --out ~/datasets/imagenet256
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from datasets import DatasetDict, Image, load_dataset
from PIL import Image as PILImage

SPLITS = ("train", "validation", "test")


def find_shards(src: Path) -> dict[str, list[str]]:
    """Parquet shards per split, matched on the file name prefix."""
    files = sorted(src.rglob("*.parquet"))
    shards = {split: [str(f) for f in files if f.name.startswith(split)] for split in SPLITS}
    if not any(shards.values()):
        found = sorted({f.suffix or f.name for f in src.rglob("*") if f.is_file()})
        raise FileNotFoundError(f"No train/validation/test Parquet shards under {src}; file types found: {found}")
    return {split: files for split, files in shards.items() if files}


def resize_jpeg(data: bytes, size: int, quality: int) -> bytes:
    """Short side to `size`, RGB, JPEG-encoded."""
    image = PILImage.open(io.BytesIO(data))
    width, height = image.size
    scale = size / min(width, height)
    target = (max(size, round(width * scale)), max(size, round(height * scale)))
    image.draft("RGB", target)  # JPEG decodes directly at a reduced scale when possible
    image = image.convert("RGB")
    if image.size != target:
        image = image.resize(target, PILImage.BICUBIC, reducing_gap=3.0)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="folder with the downloaded Parquet shards")
    parser.add_argument("--out", required=True, help="output folder for the prepared dataset")
    parser.add_argument("--size", type=int, default=256, help="short side in pixels")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    shards = find_shards(src)
    for split, files in shards.items():
        print(f"{split}: {len(files)} shards")

    dataset = load_dataset("parquet", data_files=shards)
    dataset = dataset.cast_column("image", Image(decode=False))

    def process(batch):
        return {"image": [{"bytes": resize_jpeg(item["bytes"], args.size, args.quality), "path": None}
                          for item in batch["image"]]}

    dataset = DatasetDict({
        split: ds.map(process, batched=True, batch_size=256, num_proc=args.num_proc, desc=f"resize {split}")
        for split, ds in dataset.items()
    })
    dataset = dataset.cast_column("image", Image())
    dataset.save_to_disk(str(out), num_proc=args.num_proc)
    print(f"saved to {out}: " + ", ".join(f"{k} {len(v)}" for k, v in dataset.items()))


if __name__ == "__main__":
    main()
