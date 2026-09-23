"""AraDiCE extractor for Arabic dialect multiple-choice tasks."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AradiceExtractor"]
_LOG = setup_logger(__name__)


from .aradice_parts.task_names import task_names  # noqa: F401
from .aradice_parts.aradice_extractor import AradiceExtractor  # noqa: F401
