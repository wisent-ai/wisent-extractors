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

__all__ = ["OpenbookqaFactExtractor"]

log = setup_logger(__name__)

task_names = ("openbookqa_fact",)


class OpenbookqaFactExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for the fact-augmented variant of OpenBookQA (Mihaylov et al.,
    2018), i.e. the `additional` config on HF allenai/openbookqa which adds
    the `fact1` supporting-knowledge field on top of the standard 4-choice
    MC schema.

    lm-eval-harness only ships the `main` config, so the fact variant must be
    sourced directly from HuggingFace.

    Dataset: allenai/openbookqa (config: additional)

    Verified schema (HF dataset card, additional config):
        - id:            str
        - question_stem: str
        - choices:       {"text": list[str], "label": list[str]}  (4 options, A-D)
        - answerKey:     str   (the correct option label)
        - fact1:         str   (the originating common-knowledge fact)
        - humanScore / clarity / turkIdAnonymized (unused)
    The labelled `validation` split is used as the contrastive-pair source.

    4-choice multiple choice. Contrastive pair mirrors the in-repo AllenAI MC
    pattern (OpenBookQA doc_to_target = choices.label.index(answerKey)):
        - positive = choices.text[idx]            where idx = label.index(answerKey)
        - negative = choices.text[(idx + 1) % n]  (next-cyclic distractor)
    The `fact1` is prepended to the question_stem so the model receives the
    fact-augmented prompt the benchmark variant intends.
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from OpenBookQA `additional` validation docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/openbookqa additional (limit={max_items})")
        dataset = load_dataset("allenai/openbookqa", "additional", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid OpenBookQA-fact pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single OpenBookQA additional doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            question_stem = str(doc.get("question_stem", "")).strip()
            fact1 = str(doc.get("fact1", "")).strip()
            choices = doc.get("choices", {}) or {}
            texts = list(choices.get("text", []) or [])
            labels = list(choices.get("label", []) or [])
            answer_key = str(doc.get("answerKey", "")).strip()

            if (
                not question_stem
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
                f"Fact: {fact1}\nQuestion: {question_stem}\nAnswer:"
                if fact1
                else f"Question: {question_stem}\nAnswer:"
            )
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting OpenBookQA-fact pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="openbookqa_fact",
        )
