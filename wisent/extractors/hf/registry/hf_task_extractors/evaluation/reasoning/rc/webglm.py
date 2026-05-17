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

__all__ = ["WebglmExtractor"]

log = setup_logger(__name__)

task_names = ("webglm",)

# Abstention negatives — identical set used by SQuAD2Extractor (verified in
# wisent/extractors/lm_eval/.../reading_comprehension/squad2.py); matches the
# RC-Abstain method definition in research/table.tex.
_RC_ABSTAIN_NEGATIVES = (
    "The information is not provided in the background.",
    "This cannot be determined from the background.",
    "The background does not contain this information.",
)


class WebglmExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for WebGLM-QA (Liu et al., 2023 — WebGLM paper).

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: THUDM/webglm-qa (HuggingFace)

    Verified schema (HF dataset card):
        - question:   str
        - answer:     str   (long-form generated response with inline citations)
        - references: list[str] (3-5 source snippets the answer was grounded on)
    Splits: train (43579), validation (1000), test (400) — all labelled.
    The labelled `validation` split is used as the contrastive-pair source.

    Long-form open-domain QA. Contrastive pair follows the in-repo RC-Abstain
    pattern (research/table.tex; mirrors SQuAD2Extractor exactly):
        - positive = gold answer string
        - negative = random.choice(_RC_ABSTAIN_NEGATIVES)
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from WebGLM-QA validation docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading THUDM/webglm-qa (limit={max_items})")
        dataset = load_dataset("THUDM/webglm-qa", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid WebGLM-QA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single WebGLM-QA doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            question = str(doc.get("question", "")).strip()
            answer = str(doc.get("answer", "")).strip()

            if not question or not answer:
                return None

            correct = answer
            incorrect = random.choice(_RC_ABSTAIN_NEGATIVES)

            prompt = f"Question: {question}\nAnswer:"
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting WebGLM-QA pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="webglm",
        )
