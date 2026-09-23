"""Parts of lm_task_pairs_generation.py, split by the tama size splitter; lm_task_pairs_generation.py imports every name back."""

from __future__ import annotations
import random
from typing import TYPE_CHECKING
from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor, is_rate_limit_exc
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair


def _flatten_task_dict(task_dict: dict, prefix: str = "") -> list[tuple[str, "ConfigurableTask"]]:
    """
    Recursively flatten nested group tasks into a list of (name, ConfigurableTask) tuples.

    Handles both string keys and Task object keys.

    arguments:
        task_dict: Dict of task_name -> ConfigurableTask or nested dict
        prefix: Prefix for nested task names

    returns:
        List of (full_task_name, ConfigurableTask) tuples (leaf tasks only)
    """
    from lm_eval.api.task import ConfigurableTask

    result = []
    for name, task in task_dict.items():
        # Handle both string keys and Task object keys
        if isinstance(name, str):
            task_name = name
        else:
            # Task object as key - try to get its name
            task_name = getattr(name, 'name', None) or getattr(name, 'NAME', None) or str(name)

        full_name = f"{prefix}/{task_name}" if prefix else task_name
        if isinstance(task, ConfigurableTask):
            result.append((full_name, task))
        elif isinstance(task, dict):
            # Nested group - recurse
            result.extend(_flatten_task_dict(task, full_name))
    return result


def _add_evaluator_to_pairs(
    pairs: list["ContrastivePair"],
    evaluator_name: str | None,
    task_name: str,
) -> list["ContrastivePair"]:
    """Add evaluator_name and task_name to each pair's metadata."""
    from dataclasses import replace
    
    result = []
    for pair in pairs:
        metadata = dict(pair.metadata) if pair.metadata else {}
        metadata["evaluator_name"] = evaluator_name
        metadata["source_task"] = task_name
        result.append(replace(pair, metadata=metadata))
    return result


def _load_subtask_from_parent(task_name: str, loader, log):
    """Try to load a subtask by loading its parent group and finding the subtask within.

    For example, 'aclue_ancient_chinese_culture' -> load 'aclue' group -> find subtask.
    Tries progressively shorter prefixes as the parent name.

    Also handles tasks with variant suffixes (e.g. '_light') by constructing
    suffixed parent candidates.  For example:
      'arabic_leaderboard_acva_arabic_literature_light'
      -> try parent 'arabic_leaderboard_acva_light' (strip the middle subtopic,
         keep the variant suffix) in addition to the plain prefix approach.
    """
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
        GROUP_TASK_EXPANSIONS,
    )

    def _normalize_name(name: str) -> str:
        """Normalize name for comparison by converting dashes to underscores."""
        return name.replace("-", "_").lower()

    def _match(leaf_name: str) -> bool:
        """Case-insensitive match between a leaf task name and the requested task name.

        Requires an underscore word-boundary before any suffix match so that a
        short single-word leaf like "flores" does not falsely match
        "african_flores".  Additionally, single-word (no underscore) leaf names
        are only accepted on exact match to prevent spurious hits.
        """
        clean_leaf = leaf_name.split("/")[-1] if "/" in leaf_name else leaf_name
        clean_leaf_normalized = _normalize_name(clean_leaf)
        task_normalized = _normalize_name(task_name)
        if clean_leaf_normalized == task_normalized:
            return True
        # Single-word leaf names only match exactly (handled above).
        if "_" not in clean_leaf_normalized:
            return False
        # leaf ends with task_name (task is a suffix of leaf): require underscore boundary
        if clean_leaf_normalized.endswith(f"_{task_normalized}"):
            return True
        # task_name ends with leaf (leaf is a suffix of task): require underscore boundary
        # Only apply if the leaf itself is multi-word to avoid short-name false positives.
        if task_normalized.endswith(f"_{clean_leaf_normalized}"):
            return True
        return False

    def _try_parent(parent_name: str):
        """Load parent group and search for task_name among its leaf tasks.

        Returns (task_obj, parent_name) tuple on success, (None, None) on failure.
        """
        try:
            parent_obj = loader.load_lm_eval_task(parent_name)
        except Exception as _e:
            if is_rate_limit_exc(_e):
                raise
            return None, None
        if not isinstance(parent_obj, dict):
            return None, None
        for leaf_name, leaf_task in _flatten_task_dict(parent_obj):
            if _match(leaf_name):
                log.info(f"Found subtask '{task_name}' in parent group '{parent_name}'")
                return leaf_task, parent_name
        return None, None

    task_normalized = _normalize_name(task_name)

    # Strategy 0a: task_name is itself a GROUP key in GROUP_TASK_EXPANSIONS.
    # This handles cases where lm-eval registers the group under a different name
    # (e.g. 'afrimgsm-irokobench' instead of 'afrimgsm'), so direct loading of the
    # group fails.  Fall back to loading each known expansion subtask individually
    # and returning a synthetic dict so the caller can iterate over all subtasks.
    for group_key, expansion_subtasks in GROUP_TASK_EXPANSIONS.items():
        if _normalize_name(group_key) == task_normalized:
            log.info(
                f"'{task_name}' is a GROUP_TASK_EXPANSIONS key — "
                f"attempting to load {len(expansion_subtasks)} expansion subtasks individually"
            )
            synthetic_dict: dict = {}
            for subtask_name in expansion_subtasks:
                try:
                    subtask_obj = loader.load_lm_eval_task(subtask_name)
                    if isinstance(subtask_obj, ConfigurableTask):
                        synthetic_dict[subtask_name] = subtask_obj
                    elif isinstance(subtask_obj, dict):
                        synthetic_dict.update(subtask_obj)
                except Exception:
                    pass
            if synthetic_dict:
                log.info(
                    f"Built synthetic group dict with {len(synthetic_dict)} subtasks "
                    f"for '{task_name}'"
                )
                return synthetic_dict, group_key
            break  # found the group key but couldn't load any subtask

    # Strategy 0b: task_name is a subtask listed in GROUP_TASK_EXPANSIONS values.
    # Try loading its parent group and finding the subtask within.
    for parent_name, subtasks in GROUP_TASK_EXPANSIONS.items():
        # Check case-insensitively with dash/underscore normalization
        if any(_normalize_name(s) == task_normalized for s in subtasks):
            log.info(f"Found '{task_name}' in GROUP_TASK_EXPANSIONS under parent '{parent_name}'")
            result, parent = _try_parent(parent_name)
            if result is not None:
                return result, parent

    parts = task_name.split("_")

    # Strategy 1: progressively shorter plain prefixes
    for i in range(len(parts) - 1, 0, -1):
        parent_name = "_".join(parts[:i])
        result, parent = _try_parent(parent_name)
        if result is not None:
            return result, parent

    # Strategy 2: for tasks that end with a known variant suffix (e.g. '_light', '_with_pddl'),
    # also try parents formed by combining the base prefix with that suffix.
    # Example: 'arabic_leaderboard_acva_arabic_literature_light'
    #   suffix = 'light', base parts = ['arabic','leaderboard','acva','arabic','literature']
    #   -> try parents: 'arabic_leaderboard_acva_arabic_literature_light' (already tried above),
    #      'arabic_leaderboard_acva_arabic_light', 'arabic_leaderboard_acva_light', ...
    # Example: 'acp_app_gen_with_pddl'
    #   suffix = 'with_pddl', suffix_parts = ['with', 'pddl'], base_parts = ['acp','app','gen']
    #   -> try parents: 'acp_app_gen_with_pddl' (already tried above),
    #      'acp_app_with_pddl', 'acp_with_pddl', ...
    KNOWN_VARIANT_SUFFIXES = (("light",), ("with", "pddl"))
    for suffix_parts in KNOWN_VARIANT_SUFFIXES:
        suffix_str = "_".join(suffix_parts)
        if task_name.endswith(f"_{suffix_str}"):
            # Check if the task_name ends with the suffix tokens
            num_suffix_parts = len(suffix_parts)
            if len(parts) > num_suffix_parts and parts[-num_suffix_parts:] == list(suffix_parts):
                base_parts = parts[:-num_suffix_parts]  # strip the suffix tokens
                for i in range(len(base_parts) - 1, 0, -1):
                    parent_name = "_".join(base_parts[:i]) + f"_{suffix_str}"
                    result, parent = _try_parent(parent_name)
                    if result is not None:
                        return result, parent

    return None, None


def _pairs_from_lazy_group(task_name, lm_eval_subtask_names, loader, extractor, evaluator_name, max_items,
                           train_ratio, log, upload_pairs_to_hf) -> list["ContrastivePair"]:
    """Load a known GROUP's subtasks one at a time and stop once there are enough pairs."""
    from lm_eval.api.task import ConfigurableTask

    random.shuffle(lm_eval_subtask_names)
    pairs_per_task = max(1, max_items // len(lm_eval_subtask_names)) if max_items else None

    all_pairs: list["ContrastivePair"] = []
    for subtask_name in lm_eval_subtask_names:
        if max_items is not None and len(all_pairs) >= max_items:
            break
        try:
            subtask_obj = loader.load_lm_eval_task(subtask_name)
        except Exception as _e:
            log.warning(f"Could not load subtask '{subtask_name}': {_e}")
            continue

        # subtask_obj may itself be a ConfigurableTask or a dict
        if isinstance(subtask_obj, ConfigurableTask):
            leaf_pairs_list = [(subtask_name, subtask_obj)]
        elif isinstance(subtask_obj, dict):
            leaf_pairs_list = _flatten_task_dict(subtask_obj)
        else:
            log.warning(f"Unexpected subtask type for '{subtask_name}': {type(subtask_obj)}")
            continue

        for leaf_name_full, leaf_task in leaf_pairs_list:
            if max_items is not None and len(all_pairs) >= max_items:
                break
            leaf_name = leaf_name_full.split("/")[-1] if "/" in leaf_name_full else leaf_name_full
            try:
                leaf_extractor = get_extractor(leaf_name)
            except Exception:
                leaf_extractor = extractor
            leaf_evaluator = getattr(leaf_extractor, 'evaluator_name', evaluator_name)
            try:
                if isinstance(leaf_extractor, HuggingFaceBenchmarkExtractor):
                    leaf_pairs = leaf_extractor.extract_contrastive_pairs(limit=pairs_per_task)
                else:
                    leaf_pairs = leaf_extractor.extract_contrastive_pairs(
                        leaf_task, limit=pairs_per_task, train_ratio=train_ratio
                    )
                leaf_pairs = _add_evaluator_to_pairs(leaf_pairs, leaf_evaluator, leaf_name_full)
                all_pairs.extend(leaf_pairs)
            except Exception as e:
                log.warning(f"Failed to extract from subtask '{leaf_name_full}': {e}")

    random.shuffle(all_pairs)
    if max_items is not None:
        all_pairs = all_pairs[:max_items]
    log.info(f"Extracted {len(all_pairs)} pairs from lazy group task")
    upload_pairs_to_hf(task_name, all_pairs)
    return all_pairs
