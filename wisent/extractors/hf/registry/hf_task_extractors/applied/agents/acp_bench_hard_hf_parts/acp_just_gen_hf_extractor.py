"""Parts of acp_bench_hard_hf.py, split by the tama size splitter; acp_bench_hard_hf.py imports every name back."""

from __future__ import annotations
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from ..acp_bench_hard_hf import log
from .hf_dataset_path import ACP_GEN_TASK_NAMES, AcpBenchHardHFExtractor, _WITH_PDDL_SUFFIX


class AcpJustGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_just_gen")

class AcpLandGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_land_gen")

class AcpNextaGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_nexta_gen")

class AcpAreachGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_areach_gen")

class AcpValGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_val_gen")

class AcpProgGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_prog_gen_with_pddl")

class AcpReachGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_reach_gen_with_pddl")

class AcpAppGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_app_gen_with_pddl")

class AcpJustGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_just_gen_with_pddl")

class AcpLandGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_land_gen_with_pddl")

class AcpNextaGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_nexta_gen_with_pddl")

class AcpAreachGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_areach_gen_with_pddl")

class AcpValGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_val_gen_with_pddl")


# ---------------------------------------------------------------------------
# Group extractors — aggregate multiple subtasks into a single extractor
# ---------------------------------------------------------------------------

class AcpBenchHardGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acp_bench_hard`` benchmark.

    Loads all generative ACP Bench Hard subtasks from HuggingFace and
    aggregates their pairs into a single list.  This bypasses the lm-eval
    task loader which requires optional dependencies (tarski, lark, pddl,
    kstar-planner).
    """

    evaluator_name = "generation"

    # All generative subtasks that make up acp_bench_hard
    SUBTASK_NAMES = ACP_GEN_TASK_NAMES

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs


class AcpBenchHardWithPddlGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acp_bench_hard_with_pddl`` benchmark.

    Loads all _with_pddl generative ACP Bench Hard subtasks from HuggingFace
    and aggregates their pairs into a single list.  Each subtask uses the base
    HF dataset config (the _with_pddl suffix only changes the prompt format —
    PDDL_domain and PDDL_problem columns are prepended to each prompt).
    """

    evaluator_name = "generation"

    # Only the _with_pddl subtasks
    SUBTASK_NAMES = tuple(t for t in ACP_GEN_TASK_NAMES if t.endswith(_WITH_PDDL_SUFFIX))

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs


class AcpBenchGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acpbench`` benchmark.

    Loads the generative (gen) ACP Bench subtasks from HuggingFace.
    Bool and MCQ subtasks require lm-eval and are omitted here since
    gen subtasks alone provide sufficient contrastive pairs.
    """

    evaluator_name = "generation"

    SUBTASK_NAMES = ACP_GEN_TASK_NAMES

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs
