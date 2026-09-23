from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PileExtractor"]
_LOG = setup_logger(__name__)


from .pile_parts.task_names import _PILE_CDN_CACHE, _hf_pile_headers, _fetch_monology_pile_cdn, task_names  # noqa: F401
from .pile_parts.pile_extractor import PileExtractor  # noqa: F401
