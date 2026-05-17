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

__all__ = ["ReclorExtractor"]

log = setup_logger(__name__)

task_names = ("reclor",)


class ReclorExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for ReClor — A Reading Comprehension Dataset Requiring Logical
    Reasoning (Yu et al., ICLR 2020).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: metaeval/reclor (HuggingFace)

    Verified schema (HF dataset viewer):
        - context:   str        (the passage)
        - question:  str
        - answers:   list[str]  (4 options)
        - label:     int        (0-3, index of the correct option)
        - id_string: str
    Splits: train (4640), validation (500). The test split is unlabeled, so the
    labeled `validation` split is used as the contrastive-pair source.

    4-choice multiple choice. Contrastive pair mirrors the established in-repo
    MC pattern (LogiQAExtractor / SIQAExtractor / COPAExtractor):
        - positive = answers[label]
        - negative = answers[(label + 1) % len(answers)]   (next-cyclic distractor)
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from ReClor validation docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading metaeval/reclor (limit={max_items})")
        dataset = load_dataset("metaeval/reclor", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid ReClor pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single ReClor doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answers = doc.get("answers", [])
            label = doc.get("label")

            if (
                not context
                or not question
                or not isinstance(answers, (list, tuple))
                or len(answers) < 2
                or not isinstance(label, int)
                or not (0 <= label < len(answers))
            ):
                return None

            correct = str(answers[label]).strip()
            incorrect = str(answers[(label + 1) % len(answers)]).strip()
            if not correct or not incorrect:
                return None

            prompt = f"Passage: {context}\nQuestion: {question}"
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting ReClor pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="reclor",
        )
