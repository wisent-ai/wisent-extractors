from __future__ import annotations

from typing import Any

from datasets import load_dataset

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import (
    NegativeResponse,
    PositiveResponse,
)
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger

__all__ = ["QuailExtractor"]

log = setup_logger(__name__)

task_names = ("quail",)


class QuailExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for QuAIL — Question Answering for Artificial Intelligence
    Linguistics (Rogers et al., 2020).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: textmachinelab/quail (HuggingFace)

    Verified schema (HF dataset card):
        - context:           str
        - question:          str
        - answers:           list[str]  (multiple-choice options)
        - correct_answer_id: int        (index into answers)
        - domain / metadata / question_type (unused for pairs)
    The labelled `validation` split is used as the contrastive-pair source.

    Multi-choice RC. Contrastive pair mirrors the established in-repo integer
    -index MC pattern (ReclorExtractor / LogiQA / SIQA / COPA):
        - positive = answers[correct_answer_id]
        - negative = answers[(correct_answer_id + 1) % len(answers)]
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from QuAIL validation docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading textmachinelab/quail (limit={max_items})")
        dataset = load_dataset("textmachinelab/quail", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid QuAIL pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single QuAIL doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answers = doc.get("answers", [])
            cid = doc.get("correct_answer_id")

            if (
                not context
                or not question
                or not isinstance(answers, (list, tuple))
                or len(answers) < 2
                or not isinstance(cid, int)
                or not (0 <= cid < len(answers))
            ):
                return None

            correct = str(answers[cid]).strip()
            incorrect = str(answers[(cid + 1) % len(answers)]).strip()
            if not correct or not incorrect:
                return None

            prompt = f"Passage: {context}\nQuestion: {question}"
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting QuAIL pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="quail",
        )
