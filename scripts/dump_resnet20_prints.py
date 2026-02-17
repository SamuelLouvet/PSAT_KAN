from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from converted_KAN.convert import convert_to_kan


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump exact __repr__ printouts for ResNet20 before/after KAN conversion."
    )
    parser.add_argument(
        "--out-dir",
        default="report_artifacts",
        help="Output directory for text files (default: report_artifacts)",
    )
    parser.add_argument(
        "--repo",
        default="chenyaofo/pytorch-cifar-models",
        help="torch.hub repo (default: chenyaofo/pytorch-cifar-models)",
    )
    parser.add_argument(
        "--model",
        default="cifar10_resnet20",
        help="torch.hub model name (default: cifar10_resnet20)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights via torch.hub (downloads if needed).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    device = torch.device("cpu")
    model = torch.hub.load(args.repo, args.model, pretrained=args.pretrained, verbose=False)
    model = model.to(device).eval()

    before = repr(model)
    _write_text(out_dir / "resnet20_before.txt", before + "\n")

    kan_model = convert_to_kan(model, inplace=False).to(device).eval()
    after = repr(kan_model)
    _write_text(out_dir / "resnet20_after_convert_to_kan.txt", after + "\n")

    print(f"Wrote {out_dir / 'resnet20_before.txt'}")
    print(f"Wrote {out_dir / 'resnet20_after_convert_to_kan.txt'}")


if __name__ == "__main__":
    main()
