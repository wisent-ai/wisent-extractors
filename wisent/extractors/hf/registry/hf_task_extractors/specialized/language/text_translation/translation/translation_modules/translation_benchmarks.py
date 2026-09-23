"""Extractors for machine translation benchmarks."""
from __future__ import annotations

import random
from typing import Any, Optional

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "TranslationExtractor",
    "WMT14Extractor",
    "WMT16Extractor",
]

log = setup_logger(__name__)


from .translation_benchmarks_parts.translation_extractor import TranslationExtractor  # noqa: F401
from .translation_benchmarks_parts.wmt14_extractor import WMT14Extractor  # noqa: F401


# Re-export from split module
from wisent.extractors.hf.hf_task_extractors.translation_benchmarks_wmt16 import (
    WMT16Extractor,
)
