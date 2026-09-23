from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_LONG

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["NorevalGenerationExtractor"]
_LOG = setup_logger(__name__)


from .noreval_gen_parts.task_names import task_names, NorevalGenerationExtractor  # noqa: F401
