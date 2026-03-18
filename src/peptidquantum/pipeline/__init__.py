"""Pipeline orchestration module."""

from .config import PipelineConfig
from .pipeline import PeptidQuantumPipeline


def cli_main():
    """Import CLI lazily so `python -m peptidquantum.pipeline.cli` stays warning-free."""
    from .cli import main

    return main()


__all__ = ["PeptidQuantumPipeline", "PipelineConfig", "cli_main"]
