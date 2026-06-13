"""Download a Kaggle dataset through the Kaggle CLI.

Prerequisites:
1. Install `kaggle`.
2. Place `kaggle.json` in `~/.kaggle/kaggle.json`.
3. Accept the dataset terms on Kaggle if required.

Example:
    python scripts/download_kaggle_dataset.py --dataset owner/dataset-slug
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and unzip a Kaggle dataset into data/raw.")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, for example owner/dataset-name")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(output_dir), "--unzip"]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Downloaded dataset to {output_dir}")


if __name__ == "__main__":
    main()
