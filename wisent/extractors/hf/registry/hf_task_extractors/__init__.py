"""Task-specific extractors for HuggingFace datasets."""

import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

from wisent.extractors.hf.registry.hf_task_extractors.applied.math.competition.aime import AIMEExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_generation.apps import AppsExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_analysis.codexglue import CodexglueExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_generation.conala import ConalaExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_generation.concode import ConcodeExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_analysis.ds_1000 import Ds1000Extractor
from wisent.extractors.hf.registry.hf_task_extractors.evaluation.knowledge.truthfulqa.hle import HleExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.math.competition.hmmt_to_olympiadbench.hmmt import HMMTExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_generation.humaneval_to_humanevalpack.humaneval import (
    HumanEvalExtractor,
    HumanEval64Extractor,
    HumanEvalPlusExtractor,
    HumanEvalInstructExtractor,
    HumanEval64InstructExtractor,
)
try:
    from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.multilang_benchmarks.livecodebench.livecodebench_modules.livecodebench import LivecodebenchExtractor
except ImportError:
    LivecodebenchExtractor = None
from wisent.extractors.hf.registry.hf_task_extractors.applied.math.benchmarks.livemathbench import LiveMathBenchExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.math.benchmarks.math500 import MATH500Extractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.competitive.mercury import MercuryExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.multilang_benchmarks.multilang.multiple_multipl_to_multiple.multipl_e import MultiplEExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.math.polymath.polymath import PolyMathExtractor
from wisent.extractors.hf.registry.hf_task_extractors.applied.coding.code_tasks.code_analysis.recode_to_scicode.recode import RecodeExtractor
from wisent.extractors.hf.registry.hf_task_extractors.evaluation.knowledge.mmlu.openllm_to_super.super_gpqa import SuperGpqaExtractor

__all__ = [
    "AIMEExtractor",
    "AppsExtractor",
    "CodexglueExtractor",
    "ConalaExtractor",
    "ConcodeExtractor",
    "Ds1000Extractor",
    "HleExtractor",
    "HMMTExtractor",
    "HumanEvalExtractor",
    "HumanEval64Extractor",
    "HumanEvalPlusExtractor",
    "HumanEvalInstructExtractor",
    "HumanEval64InstructExtractor",
    "LivecodebenchExtractor",
    "LiveMathBenchExtractor",
    "MATH500Extractor",
    "MercuryExtractor",
    "MultiplEExtractor",
    "PolyMathExtractor",
    "RecodeExtractor",
    "SuperGpqaExtractor",
]

