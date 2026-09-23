from __future__ import annotations

import random
from typing import TYPE_CHECKING

from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor, is_rate_limit_exc
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair

__all__ = ["build_contrastive_pairs", "lm_build_contrastive_pairs"]
_LOG = setup_logger(__name__)


from .lm_task_pairs_generation_parts.flatten_task_dict import _flatten_task_dict, _add_evaluator_to_pairs, _load_subtask_from_parent, _pairs_from_lazy_group  # noqa: F401
from .lm_task_pairs_generation_parts.build_contrastive_pairs import _pairs_from_group, build_contrastive_pairs  # noqa: F401
from .lm_task_pairs_generation_parts.lm_build_contrastive_pairs import lm_build_contrastive_pairs  # noqa: F401
