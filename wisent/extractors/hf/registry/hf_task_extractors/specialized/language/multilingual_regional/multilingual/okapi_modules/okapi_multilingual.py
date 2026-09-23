"""Extractors for Okapi multilingual benchmarks (MMLU, HellaSwag, TruthfulQA)."""
from __future__ import annotations

import json
import os
import random
from typing import Any

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.config_tools.constants import INDEX_FIRST, SENSOR_LAST_OFFSET
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "OkapiMMLUExtractor",
    "OkapiHellaswagExtractor",
    "OkapiTruthfulQAExtractor",
]

log = setup_logger(__name__)


from .okapi_multilingual_parts.okapi_mmlu_extractor import _OKAPI_LANGS, _hf_headers, _fetch_okapi_json, _fetch_okapi_parquet, OkapiMMLUExtractor  # noqa: F401
from .okapi_multilingual_parts.okapi_hellaswag_extractor import OkapiHellaswagExtractor  # noqa: F401


# Re-export from split module
from wisent.extractors.hf.hf_task_extractors.okapi_multilingual_truthfulqa import (
    OkapiTruthfulQAExtractor,
)
