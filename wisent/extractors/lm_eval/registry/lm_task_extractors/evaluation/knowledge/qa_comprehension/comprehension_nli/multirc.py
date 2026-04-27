from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MultiRCExtractor"]
_LOG = setup_logger(__name__)


class MultiRCExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the MultiRC benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MultiRC docs.

        MultiRC schema:
            - paragraph: str
            - question: str
            - answer: dict
            - label: 0 or 1
            
        Args:
            lm_eval_task_data: lm-eval task instance for MultiRC.
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
            log.warning("No valid MultiRC pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MultiRC doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            paragraph = str(doc.get("paragraph") or "").strip()
            question = str(doc.get("question") or "").strip()
            answer = str(doc.get("answer") or "").strip()
            raw_label = doc.get("label")
            # MultiRC variants sometimes store label as a numeric string ("0"/"1"),
            # as a bool, or as yes/no/true/false text. Coerce to int 0/1.
            label: int | None
            if isinstance(raw_label, bool):
                label = int(raw_label)
            elif isinstance(raw_label, int):
                label = raw_label
            elif isinstance(raw_label, str):
                s = raw_label.strip().lower()
                if s in ("0", "1"):
                    label = int(s)
                elif s in ("yes", "true", "correct"):
                    label = 1
                elif s in ("no", "false", "incorrect"):
                    label = 0
                else:
                    label = None
            else:
                label = None

            if label not in {0, 1}:
                log.debug(
                    "Skipping doc due to invalid label",
                    extra={"doc": doc},
                )
                return None
            # Empty paragraph/question/answer happen for ~3 of 27243 super_glue
            # multirc rows. The pair is still meaningful (label is yes/no), so
            # use placeholders rather than dropping.
            if not paragraph:
                paragraph = "(no paragraph provided)"
            if not question:
                question = "(no question provided)"
            if not answer:
                answer = "(no answer provided)"

            prompt = f"{paragraph}\nQuestion: {question}\nAnswer: {answer}\nIs this answer correct?"

            correct = "Yes" if label == 1 else "No"
            incorrect = "No" if label == 1 else "Yes"

            metadata = {
                "label": "multirc",
            }

            return self._build_pair(
                question=prompt,
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