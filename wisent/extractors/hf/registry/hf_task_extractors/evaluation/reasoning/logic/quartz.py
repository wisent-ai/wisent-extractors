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

__all__ = ["QuartzExtractor"]

log = setup_logger(__name__)

task_names = ("quartz",)


class QuartzExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for QuaRTz — Open-Domain Reasoning About Qualitative
    Relationships (Tafjord et al., 2019).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: allenai/quartz (HuggingFace)

    Verified schema (HF dataset viewer):
        - question:  str
        - choices:   {"text": list[str], "label": list[str]}  (2 options, A/B)
        - answerKey: str   (the correct option label)
        - para:      str   (background knowledge sentence)
    Splits: train (2696), validation (384), test (784) — all labelled.
    The labelled `validation` split is used as the contrastive-pair source.

    2-choice multiple choice. Contrastive pair mirrors the established in-repo
    AllenAI MC pattern (OpenBookQA doc_to_target = choices.label.index(answerKey)):
        - positive = choices.text[idx]            where idx = label.index(answerKey)
        - negative = choices.text[(idx + 1) % n]  (the other option / 2C flip)
    The `para` background is prepended to the question as the benchmark presents it.
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from QuaRTz validation docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/quartz (limit={max_items})")
        dataset = load_dataset("allenai/quartz", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid QuaRTz pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single QuaRTz doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            question = str(doc.get("question", "")).strip()
            para = str(doc.get("para", "")).strip()
            choices = doc.get("choices", {}) or {}
            texts = list(choices.get("text", []) or [])
            labels = list(choices.get("label", []) or [])
            answer_key = str(doc.get("answerKey", "")).strip()

            if (
                not question
                or len(texts) < 2
                or len(texts) != len(labels)
                or answer_key not in labels
            ):
                return None

            idx = labels.index(answer_key)
            correct = str(texts[idx]).strip()
            incorrect = str(texts[(idx + 1) % len(texts)]).strip()
            if not correct or not incorrect:
                return None

            prompt = (
                f"{para}\nQuestion: {question}\nAnswer:"
                if para
                else f"Question: {question}\nAnswer:"
            )
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting QuaRTz pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="quartz",
        )
