"""
Thin entrypoint for the scoring/reranking pipeline.

This keeps the active training implementation in train_baseline.py while
providing an explicit command surface for scoring-focused experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Train PeptidQuantum scoring/reranking model")
    parser.add_argument("--config", type=Path, required=True, help="Path to scoring config YAML")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    import train_baseline  # noqa: E402

    train_baseline.main(str(args.config))


if __name__ == "__main__":
    main()
