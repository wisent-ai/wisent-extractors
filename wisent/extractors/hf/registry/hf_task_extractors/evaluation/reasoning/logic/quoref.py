from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import (
    NegativeResponse,
    PositiveResponse,
)
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger

__all__ = ["QuorefExtractor"]

log = setup_logger(__name__)

task_names = ("quoref",)


# Abstention negatives — identical set used by SQuAD2Extractor (verified in
# wisent/extractors/lm_eval/.../reading_comprehension/squad2.py); matches the
# RC-Abstain method definition in research/table.tex.
_RC_ABSTAIN_NEGATIVES = (
    "The information is not provided in the background.",
    "This cannot be determined from the background.",
    "The background does not contain this information.",
)


class QuorefExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Quoref — A Reading Comprehension Dataset with Questions
    Requiring Coreferential Reasoning (Dasigi et al., 2019).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: allenai/quoref (HuggingFace)

    Verified schema (HF dataset card):
        - id:       str
        - question: str
        - context:  str
        - title:    str
        - url:      str
        - answers:  {"answer_start": list[int] | int, "text": list[str] | str}
                    (SQuAD-derived shape; handled defensively)
    Splits: train (19399), validation (2418), no test. The labelled
    `validation` split is used as the contrastive-pair source.

    Extractive QA. Contrastive pair follows the in-repo RC-Abstain pattern
    (research/table.tex; mirrors SQuAD2Extractor exactly):
        - positive = gold answer span text
        - negative = random.choice(_RC_ABSTAIN_NEGATIVES)
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Quoref validation docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/quoref (limit={max_items})")
        dataset = load_dataset("allenai/quoref", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Quoref pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Quoref doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answers = doc.get("answers", {}) or {}

            if not context or not question:
                return None

            # answers.text may be a list (SQuAD-style parallel arrays) or a
            # singular string, depending on the loaded dataset version.
            text_field = answers.get("text") if isinstance(answers, dict) else None
            if isinstance(text_field, (list, tuple)) and text_field:
                correct = str(text_field[0]).strip()
            elif isinstance(text_field, str):
                correct = text_field.strip()
            else:
                return None

            if not correct:
                return None

            incorrect = random.choice(_RC_ABSTAIN_NEGATIVES)

            prompt = f"Passage: {context}\nQuestion: {question}\nAnswer:"
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting Quoref pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="quoref",
        )
