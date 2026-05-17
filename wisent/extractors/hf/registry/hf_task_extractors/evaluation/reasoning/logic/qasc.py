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

__all__ = ["QascExtractor"]

log = setup_logger(__name__)

task_names = ("qasc",)


class QascExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for QASC — Question Answering via Sentence Composition
    (Khot et al., 2020).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: allenai/qasc (HuggingFace)

    Verified schema (HF dataset viewer):
        - question:  str
        - choices:   {"text": list[str], "label": list[str]}  (8 options, A-H)
        - answerKey: str   (the correct option label)
        - fact1/fact2/combinedfact/formatted_question (unused for pairs)
    Splits: train (8134), validation (926), test (920) — all labelled.
    The labelled `validation` split is used as the contrastive-pair source.

    8-choice multiple choice. Contrastive pair mirrors the established in-repo
    AllenAI MC pattern (OpenBookQA doc_to_target = choices.label.index(answerKey)):
        - positive = choices.text[idx]            where idx = label.index(answerKey)
        - negative = choices.text[(idx + 1) % n]  (next-cyclic distractor)
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from QASC validation docs.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/qasc (limit={max_items})")
        dataset = load_dataset("allenai/qasc", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid QASC pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single QASC doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            question = str(doc.get("question", "")).strip()
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

            prompt = f"Question: {question}\nAnswer:"
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting QASC pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="qasc",
        )
