"""Parts of lm_task_pairs_generation.py, split by the tama size splitter; lm_task_pairs_generation.py imports every name back."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING
from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor, is_rate_limit_exc
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from ..lm_task_pairs_generation import _LOG
from .flatten_task_dict import _add_evaluator_to_pairs, _flatten_task_dict, _load_subtask_from_parent, _pairs_from_lazy_group


def _pairs_from_group(task_name, task_obj, extractor, evaluator_name, max_items, train_ratio, log,


                      upload_pairs_to_hf) -> list["ContrastivePair"]:


    """Sample pairs across every leaf subtask of a group task dict, `max_items` in all."""
    leaf_tasks = _flatten_task_dict(task_obj)
    log.info(f"Group task with {len(leaf_tasks)} leaf subtasks")

    if not leaf_tasks:
        log.warning("No leaf tasks found in group")
        return []

    # Shuffle to get random sampling across subtasks
    random.shuffle(leaf_tasks)

    # Calculate pairs per subtask
    if max_items is None:
        pairs_per_task = None
    else:
        # Distribute limit across subtasks, minimum 1 per task
        pairs_per_task = max(1, max_items // len(leaf_tasks))

    all_pairs = []
    for subtask_name, subtask in leaf_tasks:
        try:
            # Get the leaf task name (last part after /)
            leaf_name = subtask_name.split("/")[-1] if "/" in subtask_name else subtask_name

            # Try to get extractor for the specific subtask first
            try:
                subtask_extractor = get_extractor(leaf_name)
            except Exception:
                # Fall back to parent extractor
                subtask_extractor = extractor

            subtask_evaluator = getattr(subtask_extractor, 'evaluator_name', evaluator_name)

            # HF extractors take only `limit` (no lm-eval task object, no train_ratio).
            # LM extractors take (task, limit, train_ratio).  Without this branch a leaf
            # that happens to resolve to an HF extractor would crash with
            # "got multiple values for argument 'limit'" because the positional `subtask`
            # would collide with the keyword `limit` in HF's signature.
            if isinstance(subtask_extractor, HuggingFaceBenchmarkExtractor):
                # Make the HF extractor route to the correct leaf dataset by setting task_name.
                try:
                    subtask_extractor.task_name = leaf_name
                except Exception:
                    pass
                subtask_pairs = subtask_extractor.extract_contrastive_pairs(limit=pairs_per_task)
            else:
                subtask_pairs = subtask_extractor.extract_contrastive_pairs(subtask, limit=pairs_per_task, train_ratio=train_ratio)
            subtask_pairs = _add_evaluator_to_pairs(subtask_pairs, subtask_evaluator, subtask_name)
            all_pairs.extend(subtask_pairs)

            # Stop if we have enough
            if max_items is not None and len(all_pairs) >= max_items:
                break
        except Exception as e:
            log.warning(f"Failed to extract from subtask {subtask_name}: {e}")
            continue

    # Shuffle final result and trim to limit
    random.shuffle(all_pairs)
    if max_items is not None:
        all_pairs = all_pairs[:max_items]

    log.info(f"Extracted {len(all_pairs)} pairs from group task")
    upload_pairs_to_hf(task_name, all_pairs)
    return all_pairs


def build_contrastive_pairs(
    task_name: str,
    limit: int | None = None,
    *,
    train_ratio: float,
) -> list["ContrastivePair"]:
    """
    Unified loader for contrastive pairs - handles both HuggingFace and lm-eval tasks.

    Loads from storage first (cache -> HF -> Supabase). If not found in any
    storage, generates via extractors and uploads to HF for future reuse.

    arguments:
        task_name:
            Name of the benchmark/task (e.g., "winogrande", "mmlu", "humaneval").
        limit:
            Optional upper bound on the number of pairs to return.
            Non-positive values are treated as no limit.

    returns:
        A list of ContrastivePair objects, each with metadata containing
        'evaluator_name' and 'source_task'.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs (unified)", extra={"limit": limit})
    from wisent.core.utils.services.benchmarks import validate_benchmark
    validate_benchmark(task_name)
    
    # Normalize limit
    max_items = None if (limit is None or limit <= 0) else int(limit)
    
    # Try loading from storage first (cache -> HF -> Supabase)
    from wisent.extractors.lm_eval.registry.lm_task_pairs_storage import (
        try_load_from_storage, upload_pairs_to_hf,
    )
    stored_pairs = try_load_from_storage(task_name, max_items)
    if stored_pairs:
        extractor = get_extractor(task_name)
        evaluator_name = getattr(extractor, 'evaluator_name', None)
        return _add_evaluator_to_pairs(stored_pairs, evaluator_name, task_name)

    log.info("No stored pairs found, generating from extractors")

    # Get extractor
    extractor = get_extractor(task_name)
    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})

    # Get evaluator_name from extractor
    evaluator_name = getattr(extractor, 'evaluator_name', None)

    # HuggingFace extractor - load directly
    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        log.info("HuggingFace task - loading directly")
        pairs = extractor.extract_contrastive_pairs(limit=max_items)
        upload_pairs_to_hf(task_name, pairs)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
    
    # lm-eval extractor - need to load task
    log.info("lm-eval task - loading via LMEvalDataLoader")
    from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
        GROUP_TASK_EXPANSIONS,
    )

    loader = LMEvalDataLoader()

    # Check if task is a known GROUP with subtasks listed in GROUP_TASK_EXPANSIONS.
    # For such tasks, load subtasks LAZILY (one at a time) rather than all at once to
    # avoid the long initialisation time of get_task_dict with 50+ task names.
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    task_normalized = _normalize(task_name)
    lazy_subtask_names: list[str] | None = None
    for group_key, expansion_subtasks in GROUP_TASK_EXPANSIONS.items():
        if _normalize(group_key) == task_normalized:
            # Filter out the group key itself (e.g. "advanced_ai_risk" is in its own expansion)
            lazy_subtask_names = [s for s in expansion_subtasks if _normalize(s) != task_normalized]
            log.info(
                f"Known GROUP task '{task_name}' with {len(lazy_subtask_names)} expansion subtasks "
                f"— will load lazily"
            )
            break

    from lm_eval.api.task import ConfigurableTask

    def _to_lm_eval_subtask_name(name: str) -> str:
        """Convert a GROUP_TASK_EXPANSIONS underscore subtask name to its lm-eval dash form.

        For advanced_ai_risk subtasks, lm-eval uses dashes after the variant prefix:
          advanced_ai_risk_fewshot_coordinate_itself  ->  advanced_ai_risk_fewshot-coordinate-itself
          advanced_ai_risk_human_corrigible_less_HHH  ->  advanced_ai_risk_human-corrigible-less-HHH
          advanced_ai_risk_lm_self_awareness_training_nn_architecture
              -> advanced_ai_risk_lm-self-awareness-training-nn-architecture

        Other task names are returned unchanged.
        """
        import re
        # Pattern: advanced_ai_risk_(fewshot|human|lm)_(rest)
        m = re.match(r'^(advanced_ai_risk_(?:fewshot|human|lm))_(.+)$', name)
        if m:
            prefix, rest = m.group(1), m.group(2)
            # Replace underscores with dashes in the suffix, but preserve uppercase HHH
            dash_rest = rest.replace("_", "-")
            return f"{prefix}-{dash_rest}"
        return name

    if lazy_subtask_names is not None and len(lazy_subtask_names) > 0:
        # Lazy group loading: load subtasks one-by-one and stop once we have enough pairs.
        # Convert underscore names (from GROUP_TASK_EXPANSIONS) to lm-eval dash names so
        # each subtask loads in ~5 s directly without triggering a full parent-group reload.
        lm_eval_subtask_names = [_to_lm_eval_subtask_name(s) for s in lazy_subtask_names]
        return _pairs_from_lazy_group(
            task_name, lm_eval_subtask_names, loader, extractor, evaluator_name, max_items, train_ratio, log,
            upload_pairs_to_hf,
        )

    BYPASS_LM_EVAL_LOAD = ("scrolls_", "mediqa_qa2019")
    if any(task_name.startswith(p) for p in BYPASS_LM_EVAL_LOAD):
        try:
            pairs = extractor.extract_contrastive_pairs(None, limit=max_items, train_ratio=train_ratio)
        except TypeError:
            pairs = extractor.extract_contrastive_pairs(limit=max_items)
        upload_pairs_to_hf(task_name, pairs)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)

    try:
        task_obj = loader.load_lm_eval_task(task_name)
    except Exception as _e:
        if is_rate_limit_exc(_e):
            raise
        log.warning(f"load_lm_eval_task({task_name!r}) raised {type(_e).__name__}: {_e}", exc_info=_e)
        # Subtask not loadable directly — try loading parent group and finding subtask
        task_obj, _ = _load_subtask_from_parent(task_name, loader, log)
        # advanced_ai_risk_<variant>_<subtopic> uses dashes after <variant> in lm-eval.
        # If the parent-group fallback failed, retry with the dash-converted name
        # directly. Confirmed live on 2026-05-07: many advanced_ai_risk_human_*
        # jobs failed with "task: NoneType" because the parent-group leaf-match
        # didn't pick them up.
        if task_obj is None:
            import re as _re
            _m = _re.match(r"^(advanced_ai_risk_(?:fewshot|human|lm))_(.+)$", task_name)
            if _m:
                _dash = f"{_m.group(1)}-{_m.group(2).replace('_','-')}"
                try:
                    task_obj = loader.load_lm_eval_task(_dash)
                except Exception:
                    task_obj = None
        if task_obj is None:
            # Last resort: some extractors (storycloze, multipl_e) can produce pairs
            # without an lm-eval task object. Try the extractor directly.
            try:
                pairs = extractor.extract_contrastive_pairs(None, limit=max_items, train_ratio=train_ratio)
            except TypeError:
                try:
                    pairs = extractor.extract_contrastive_pairs(limit=max_items)
                except Exception:
                    raise
            if pairs:
                upload_pairs_to_hf(task_name, pairs)
                return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
            raise

    # If the loader returned a dict but the caller asked for a specific leaf,
    # narrow down to that leaf so we do not aggregate unrelated subtasks.
    if isinstance(task_obj, dict) and task_name in task_obj:
        task_obj = task_obj[task_name]

    # Single task (ConfigurableTask)
    if isinstance(task_obj, ConfigurableTask):
        log.info("Single task")
        pairs = extractor.extract_contrastive_pairs(task_obj, limit=max_items, train_ratio=train_ratio)
        upload_pairs_to_hf(task_name, pairs)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)

    # Group task (dict) - flatten and sample from all subtasks
    if isinstance(task_obj, dict):
        return _pairs_from_group(
            task_name, task_obj, extractor, evaluator_name, max_items, train_ratio, log, upload_pairs_to_hf,
        )

    log.error(f"Unexpected task_obj type: {type(task_obj)}")
    return []
