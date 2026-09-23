"""ACP Bench Hard (generative tasks) HuggingFace extractor.

These tasks cannot be loaded via lm-eval because the YAML configuration
imports acp_utils.py which requires optional packages (tarski, lark, pddl,
kstar-planner) that are not installed. This extractor loads the dataset
directly from HuggingFace to bypass that dependency.

Note on *_with_pddl variants:
    The HuggingFace dataset ibm-research/acp_bench only has configs for the
    base task names (acp_app_gen, acp_prog_gen, etc.). There are no separate
    configs for the _with_pddl variants. However, each row in the base dataset
    already contains PDDL_domain and PDDL_problem columns. The _with_pddl
    tasks are identical to the base tasks but include these PDDL columns in the
    model prompt. This extractor therefore loads the base config for _with_pddl
    tasks (by stripping the _with_pddl suffix) and ensures PDDL fields are
    incorporated into the prompt.
"""
from __future__ import annotations

from typing import Any

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AcpBenchHardHFExtractor"]

log = setup_logger(__name__)


from .acp_bench_hard_hf_parts.hf_dataset_path import HF_DATASET_PATH, ACP_GEN_TASK_NAMES, _WITH_PDDL_SUFFIX, _HF_CONFIG_FOR_TASK, AcpBenchHardHFExtractor, AcpProgGenHFExtractor, AcpReachGenHFExtractor, AcpAppGenHFExtractor  # noqa: F401
from .acp_bench_hard_hf_parts.acp_just_gen_hf_extractor import AcpJustGenHFExtractor, AcpLandGenHFExtractor, AcpNextaGenHFExtractor, AcpAreachGenHFExtractor, AcpValGenHFExtractor, AcpProgGenWithPddlHFExtractor, AcpReachGenWithPddlHFExtractor, AcpAppGenWithPddlHFExtractor, AcpJustGenWithPddlHFExtractor, AcpLandGenWithPddlHFExtractor, AcpNextaGenWithPddlHFExtractor, AcpAreachGenWithPddlHFExtractor, AcpValGenWithPddlHFExtractor, AcpBenchHardGroupHFExtractor, AcpBenchHardWithPddlGroupHFExtractor, AcpBenchGroupHFExtractor  # noqa: F401
