from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AfroBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)


from .afrobench_mc_parts.task_names import task_names, _SCHEMA_DISPATCHERS, _get_task_choices, _make_pair, _extract_belebele, _extract_abcd_choices, _extract_classification, _extract_choices_list, _extract_options_abcd_choices  # noqa: F401
from .afrobench_mc_parts.afro_bench_multiple_choice_extractor import _extract_choices_dict, AfroBenchMultipleChoiceExtractor  # noqa: F401
