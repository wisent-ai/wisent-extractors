from __future__ import annotations

from typing import Type, Union
import importlib
import logging

from wisent.extractors.lm_eval.atoms import (
    LMEvalBenchmarkExtractor,
    UnsupportedLMEvalBenchmarkError,
)
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

from wisent.extractors.lm_eval.lm_extractor_manifest import EXTRACTORS as _LM_MANIFEST
from wisent.extractors.hf.hf_extractor_manifest import HF_EXTRACTORS as _HF_MANIFEST
from wisent.core.utils.infra_tools.errors import InvalidValueError, InvalidDataFormatError

__all__ = [
    "register_extractor",
    "get_extractor",
    "is_rate_limit_exc",
]


def is_rate_limit_exc(exc: BaseException) -> bool:
    """True if exc OR any link in its __cause__/__context__ chain is a 429.

    The bare 'except Exception:' blocks in lm_task_pairs_generation.py
    convert HF 429s into NoDocsAvailableError, hiding the real cause from
    string-matchers on the outer exception. Walk the chain so callers can
    decide to re-raise instead of swallowing.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        if "429" in msg or "too many requests" in msg or "rate limit" in msg:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False

LOG = logging.getLogger(__name__)


def _normalize_task_key(name: str) -> str:
    """Normalize task name for registry lookup: lowercase and replace dashes with underscores."""
    return (name or "").strip().lower().replace("-", "_")

# Combine LM-eval and HuggingFace manifests.
# By default, LM-eval extractors take precedence over HF for overlapping keys
# because lm-eval has the canonical task definitions (correct splits, group expansion,
# proper doc counts).  The HF manifest is used as a deliberate override only for tasks
# that genuinely cannot be served by lm-eval (code execution sandboxes, multimodal,
# missing optional packages, etc.).
#
# History: prior versions used `{**_LM_MANIFEST, **_HF_MANIFEST}` which made HF win
# for any overlap.  This caused MASSIVE data loss for tasks like med_concepts_qa
# (HF loaded medmcqa fallback ~6k pairs instead of lm-eval's 410k group-aggregated
# pairs across 15 leaf subtasks).  See bug fix in v0.1.8.
_HF_PRECEDENCE_KEYS = frozenset({
    # ACP Bench Hard generative tasks - lm-eval requires tarski/lark/pddl/kstar-planner
    "acp_app_gen", "acp_app_gen_with_pddl",
    "acp_areach_gen", "acp_areach_gen_with_pddl",
    "acp_bench_hard",
    "acp_just_gen", "acp_just_gen_with_pddl",
    "acp_land_gen", "acp_land_gen_with_pddl",
    "acp_nexta_gen", "acp_nexta_gen_with_pddl",
    "acp_prog_gen", "acp_prog_gen_with_pddl",
    "acp_reach_gen", "acp_reach_gen_with_pddl",
    "acp_val_gen", "acp_val_gen_with_pddl",
    # Code-execution benchmarks - lm-eval has them but HF wrapper handles pair gen better
    "humaneval", "humaneval_64_instruct", "humaneval_instruct",
    "mercury", "recode", "codexglue", "concode", "conala",
    # Multimodal / specialized
    "hle", "mmmu",
    # LM manifest entries that point to non-existent modules / classes -- HF must win
    "basque_glue",      # LM module 'basque_glue' missing
    "gsm8k_platinum",   # LM module 'gsm8k_platinum' missing
    "math500",          # LM class 'Math500Extractor' missing in math module
    "polymath_en_high", "polymath_en_medium",
    "polymath_zh_high", "polymath_zh_medium",  # polymath module missing
    "tmlu",             # LM module 'tmlu' missing
    "tag",              # LM module 'tag' missing
    # Misregistered LM entries that map to wrong dataset (flores -> AfroBenchCot mismatch)
    "flores",           # LM points to AfroBenchCotExtractor; HF FloresExtractor is canonical
    "livecodebench",    # LM points to non-existent submodule
    # LM extractors that fail to load lm-eval task (returns NoneType / no docs); HF works
    "paws_x",
    "doc",                          # LM extractor fails with NoneType
    "lambada_multilingual_stablelm",  # LM extractor fails with NoneType
    "math",                         # LM module has no MathExtractor class
    "meddialog", "meddialog_qsumm", "meddialog_qsumm_perplexity",
    "meddialog_raw_dialogues", "meddialog_raw_perplexity",  # subtask loader 'Entry' bug
    "okapi_hellaswag_multilingual",
    "okapi_mmlu_multilingual",
    "okapi_truthfulqa_multilingual",  # LM extractor returns NoneType
    "supergpqa",             # LM extractor returns NoneType
    "translation",           # LM expansion loads 40M+ docs across iwslt/wmt/etc, OOM
    "wmt14_en_fr", "wmt14_fr_en",  # LM loads 40M docs OOM; HF samples
    "wmt16_de_en", "wmt16_en_de", "wmt2016",  # same
})


def _build_combined_manifest() -> dict:
    """Merge LM and HF manifests on NORMALIZED keys (lowercase, dashes->underscores).

    Critical: LM has multiple spellings of the same task key (e.g. 'wmt16-de-en',
    'wmt16_de_en', 'wmt16_de-en' all referring to the same task).  Naively merging
    on raw keys leaves stale LM entries in the dict which, after normalization in
    _REGISTRY, can override HF entries that were supposed to win.  We therefore
    normalize keys here so each task has exactly ONE entry in the combined manifest.

    Precedence:
      - HF wins if the (normalized) key is in HF_PRECEDENCE_KEYS
      - Otherwise LM wins for any key it defines
      - HF wins for any key only it defines
    """
    norm_combined: dict[str, str] = {}
    # Insert LM first (will get overridden by HF if needed below)
    for k, v in _LM_MANIFEST.items():
        nk = (k or "").strip().lower().replace("-", "_")
        if nk and nk not in norm_combined:
            norm_combined[nk] = v
    # Insert HF, overriding LM whenever the key is in HF_PRECEDENCE_KEYS
    for k, v in _HF_MANIFEST.items():
        nk = (k or "").strip().lower().replace("-", "_")
        if not nk:
            continue
        if nk in _HF_PRECEDENCE_KEYS or nk not in norm_combined:
            norm_combined[nk] = v
    return norm_combined


_COMBINED_MANIFEST = _build_combined_manifest()

# Post-hoc overrides: keys where the combined manifest has the WRONG extractor.
# These take precedence over both LM and HF manifests.
_REGISTRY_OVERRIDES: dict[str, str] = {
    # lm-eval "multiple_*" tasks are BIG-bench multiple_choice groups,
    # NOT MultiPL-E coding tasks.  The HF manifest incorrectly maps them
    # to MultiplEExtractor (which loads nuprl/MultiPL-E, yielding ~161 pairs
    # instead of thousands).  Override to BigBenchExtractor.
    "multiple_cs": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_js": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_lua": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_pl": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_r": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_rb": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_sh": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
    "multiple_ts": "wisent.extractors.lm_eval.lm_task_extractors.bigbench:BigBenchExtractor",
}

# _COMBINED_MANIFEST already has normalized keys; apply overrides, then copy.
_manifest_with_overrides = dict(_COMBINED_MANIFEST)
_manifest_with_overrides.update(_REGISTRY_OVERRIDES)
_REGISTRY: dict[str, Union[str, Type[LMEvalBenchmarkExtractor]]] = _manifest_with_overrides


def register_extractor(name: str, ref: Union[str, Type[LMEvalBenchmarkExtractor]]) -> None:
    """
    Register a new extractor by name.
    arguments:
        name:
            Name/key for the extractor (case-insensitive).
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            LMEvalBenchmarkExtractor.
    raises:
        ValueError:
            If the name is empty or the string ref is malformed.
        TypeError:
            If the ref class does not subclass LMEvalBenchmarkExtractor.

    example:
        >>> from wisent.extractors.lm_eval.lm_extractor_registry import register_extractor
        >>> from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
        >>> class MyExtractor(LMEvalBenchmarkExtractor): ...
        >>> register_extractor("mytask", MyExtractor)
        >>> register_extractor("mytask2", "my_module:MyExtractor")
    """
    key = _normalize_task_key(name)
    if not key:
        raise InvalidValueError(param_name="name/key", actual="empty string", expected="non-empty string")

    if isinstance(ref, str):
        if ":" not in ref:
            raise InvalidDataFormatError(reason="String ref must be 'module_path:ClassName[.Inner]'.")
        _REGISTRY[key] = ref
        return

    if not issubclass(ref, LMEvalBenchmarkExtractor):
        raise TypeError(f"{getattr(ref, '__name__', ref)!r} must subclass LMEvalBenchmarkExtractor")

    _REGISTRY[key] = ref


def get_extractor(task_name: str) -> LMEvalBenchmarkExtractor:
    """
    Retrieve a registered extractor by task name.

    arguments:
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
            Case-insensitive. Tries exact match first, then prefix match.
            For example, "mmlu_anatomy" will match "mmlu" extractor.

    returns:
        An instance of the corresponding LMEvalBenchmarkExtractor subclass.

    raises:
        UnsupportedLMEvalBenchmarkError:
            If no extractor is registered for the given task name.
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass LMEvalBenchmarkExtractor.
    """

    key = _normalize_task_key(task_name)
    if not key:
        raise UnsupportedLMEvalBenchmarkError("Empty task name is not supported.")

    # Try exact match first
    ref = _REGISTRY.get(key)
    if ref:
        inst = _instantiate(ref)
        try:
            inst.task_name = task_name
        except Exception:
            pass
        return inst

    # Try longest-prefix matching for hierarchical task names.
    # This handles cases like "belebele_afr_latn" -> "belebele",
    # "bbh_fewshot_boolean_expressions" -> "bbh", etc.
    parts = key.split("_")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:i])
        ref = _REGISTRY.get(prefix)
        if ref:
            LOG.info(f"Using prefix match: '{task_name}' -> '{prefix}'")
            inst = _instantiate(ref)
            try:
                inst.task_name = task_name
            except Exception:
                pass
            return inst

    raise UnsupportedLMEvalBenchmarkError(
        f"No extractor registered for task '{task_name}'. "
        f"Known: {', '.join(sorted(_REGISTRY)) or '(none)'}"
    )

def _instantiate(ref: Union[str, Type[LMEvalBenchmarkExtractor]]) -> Union[LMEvalBenchmarkExtractor, HuggingFaceBenchmarkExtractor]:
    """
    Instantiate an extractor from a string reference or class.

    arguments:
        ref:
            Either a string "module_path:ClassName[.Inner]" or a subclass of
            LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor.

    returns:
        An instance of the corresponding extractor subclass.

    raises:
        ImportError:
            If the extractor class cannot be imported/resolved.
        TypeError:
            If the resolved class does not subclass LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor.
    """
    if not isinstance(ref, str):
        return ref()

    module_path, attr_path = ref.split(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ImportError(f"Cannot import module '{module_path}' for extractor '{ref}'.") from exc

    obj = mod
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ImportError(f"Extractor class '{attr_path}' not found in '{module_path}'.") from exc

    # Accept both LMEval and HuggingFace extractors
    if not isinstance(obj, type) or not (issubclass(obj, LMEvalBenchmarkExtractor) or issubclass(obj, HuggingFaceBenchmarkExtractor)):
        raise TypeError(f"Resolved object '{obj}' is not a LMEvalBenchmarkExtractor or HuggingFaceBenchmarkExtractor subclass.")
    return obj()
