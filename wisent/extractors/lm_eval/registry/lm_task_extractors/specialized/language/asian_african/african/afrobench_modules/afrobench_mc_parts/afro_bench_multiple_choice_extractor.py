"""Parts of afrobench_mc.py, split by the tama size splitter; afrobench_mc.py imports every name back."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..afrobench_mc import _LOG
from .task_names import _extract_abcd_choices, _extract_belebele, _extract_choices_list, _extract_classification, _extract_options_abcd_choices, _get_task_choices, _make_pair


class AfroBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AfroBench multiple-choice benchmarks.

    Handles the variety of document schemas used across afrobench subtasks:

    - belebele:        mc_answer1-4 + correct_answer_num (1-indexed int)
    - openai_mmlu /
      naijarc /
      uhura_arc_easy:  A/B/C/D text fields + Answer/answerKey letter
    - afrisenti /
      nollysenti /
      sib /
      masakhanews /
      injongointent:   text field + label/category/intent int index,
                       choices come from the lm-eval task config (doc_to_choice)
    - generic question/choices list:  question + choices list + answer int/letter
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)

        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

        # Extract static choices list from the task YAML config when present
        # (needed for classification tasks like afrisenti, sib, masakhanews, etc.)
        task_choices = _get_task_choices(lm_eval_task_data)

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, task_choices=task_choices)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid AfroBench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        task_choices: list[str] | None = None,
    ) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Schema 1: belebele — mc_answer1-4 + correct_answer_num (1-indexed)
            if "mc_answer1" in doc and "correct_answer_num" in doc:
                return _extract_belebele(doc, log)

            # Schema 2: A/B/C/D explicit choice fields (openai_mmlu)
            if "A" in doc and "B" in doc and "C" in doc and "D" in doc:
                return _extract_abcd_choices(doc, log)

            # Schema 2b: options_A/B/C/D explicit choice fields (naijarc)
            if "options_A" in doc and "options_B" in doc:
                return _extract_options_abcd_choices(doc, log)

            # Schema 3: classification — static choices from task YAML + int/str index in doc
            # (afrisenti, nollysenti, sib, masakhanews, injongointent)
            if task_choices is not None:
                pair = _extract_classification(doc, task_choices, log)
                if pair is not None:
                    return pair

            # Schema 4a: uhura-arc-easy — choices dict with text array + answerKey letter
            if "choices" in doc and isinstance(doc.get("choices"), dict) and "answerKey" in doc:
                return _extract_choices_dict(doc, log)

            # Schema 4b: generic list-of-choices in the doc itself
            if "choices" in doc:
                return _extract_choices_list(doc, log)

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})

        return None


def _extract_choices_dict(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """choices-dict schema (uhura-arc-easy): choices dict with text array + answerKey letter."""
    question = str(doc.get("question") or doc.get("Question") or "").strip()
    choices_container = doc.get("choices")
    if not isinstance(choices_container, dict):
        log.debug("choices_dict: choices is not a dict")
        return None
    choices = choices_container.get("text") or choices_container.get("label") or []
    if not choices:
        log.debug("choices_dict: empty choices text array")
        return None
    answer_key = doc.get("answerKey") or doc.get("Answer") or doc.get("answer")
    if not question or answer_key is None:
        log.debug("choices_dict: missing question or answerKey")
        return None
    answer_letter = str(answer_key).strip().upper()
    if answer_letter in ("A", "B", "C", "D", "E"):
        answer_idx = ord(answer_letter) - ord("A")
    else:
        try:
            answer_idx = int(answer_key)
        except (TypeError, ValueError):
            log.debug("choices_dict: cannot parse answerKey", extra={"key": answer_key})
            return None
    if not (0 <= answer_idx < len(choices)):
        log.debug("choices_dict: answer_idx out of range", extra={"idx": answer_idx})
        return None
    correct = str(choices[answer_idx]).strip()
    incorrect = str(choices[(answer_idx + 1) % len(choices)]).strip()
    return _make_pair(f"Question: {question}\nAnswer:", correct, incorrect)
