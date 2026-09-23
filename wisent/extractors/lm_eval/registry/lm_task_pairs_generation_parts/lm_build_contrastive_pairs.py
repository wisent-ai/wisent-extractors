"""Parts of lm_task_pairs_generation.py, split by the tama size splitter; lm_task_pairs_generation.py imports every name back."""

from __future__ import annotations
from typing import TYPE_CHECKING
from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from ..lm_task_pairs_generation import _LOG
from .flatten_task_dict import _add_evaluator_to_pairs


def lm_build_contrastive_pairs(
    task_name: str,
    lm_eval_task: "ConfigurableTask | None",
    limit: int | None = None,
    *,
    train_ratio: float,
) -> list["ContrastivePair"]:
    """
    Legacy function - resolve the task's extractor and return contrastive pairs.
    
    For new code, prefer using build_contrastive_pairs() which handles
    task loading automatically.

    arguments:
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
        lm_eval_task:
            An lm-eval task instance. Can be None for HuggingFace-only tasks
            like livecodebench that don't use lm-eval.
        limit:
            Optional upper bound on the number of pairs to return.
            Values <= 0 are treated as "no limit".

    returns:
        A list of ContrastivePair objects.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs", extra={"limit": limit})

    # 1) Get extractor instance by name (exact or longest-prefix)
    extractor = get_extractor(task_name)

    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})

    # 2) Normalize limit (<=0 → None)
    max_items = None if (limit is None or limit <= 0) else int(limit)

    log.info("Extracting contrastive pairs", extra={"max_items": max_items})
    
    # Get evaluator_name from extractor
    evaluator_name = getattr(extractor, 'evaluator_name', None)

    # 3) Delegate: extractor loads docs and builds pairs
    # HuggingFace extractors don't need lm_eval_task - they load data directly from HuggingFace
    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        pairs = extractor.extract_contrastive_pairs(limit=max_items)
    else:
        pairs = extractor.extract_contrastive_pairs(lm_eval_task, limit=max_items, train_ratio=train_ratio)

    return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
