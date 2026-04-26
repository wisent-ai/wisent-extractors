from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ArcChallengeExtractor"]
_LOG = setup_logger(__name__)

task_names = ("arc_challenge",)

class ArcChallengeExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Arc_Challenge benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Arc_Challenge docs.

        Arc_Challenge schema:
            - question
            - choices: dict,
            - choices["text"]: list with possible choices strings
            - answerKey: str

        Args:
            lm_eval_task_data: lm-eval task instance for Arc_Challenge.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, train_ratio=train_ratio)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Arc_Challenge pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Arc_Challenge doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = str(doc.get("question", "")).strip()
            choices_dict = doc.get("choices", {}) or {}
            # Some arc_challenge_mt rows have choices.label without choices.text or
            # choices stored as a flat list — handle both shapes.
            if isinstance(choices_dict, dict):
                choices = choices_dict.get("text") or choices_dict.get("choices") or []
                labels = choices_dict.get("label") or []
            elif isinstance(choices_dict, list):
                choices = choices_dict
                labels = []
            else:
                choices = []
                labels = []

            answer_raw = doc.get("answerKey", doc.get("answer", ""))
            answer = str(answer_raw if answer_raw is not None else "").strip()
            answer_idx: int | None = None
            if answer:
                # Letter (A/B/C/...) — preferred when present
                if len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                # Position via labels list if provided (e.g. ["A","B","C","D"])
                elif labels and answer in labels:
                    answer_idx = labels.index(answer)
                # Numeric ("1".."N", possibly 1-indexed in arc_*_mt)
                else:
                    try:
                        idx = int(answer)
                    except ValueError:
                        idx = None
                    if idx is not None:
                        # Try 0-indexed first, then 1-indexed
                        if 0 <= idx < len(choices):
                            answer_idx = idx
                        elif 1 <= idx <= len(choices):
                            answer_idx = idx - 1

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            incorrect = choices[(answer_idx+1)%len(choices)]

            question = f"{question}"

            metadata = {
                "label": "arc_easy",
            }

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:  
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))