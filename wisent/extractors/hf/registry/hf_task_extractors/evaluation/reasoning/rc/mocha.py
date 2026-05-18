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

__all__ = ["MochaExtractor"]

log = setup_logger(__name__)

task_names = ("mocha",)

# Per the MOCHA paper (Chen et al., EMNLP 2020): score is a 1-5 human
# judgement (averaged over multiple annotators) of how similar a candidate
# is to the reference. The paper provides no per-score rubric, so the
# natural median-cutoff of the ordinal scale is used: score <= 2.0 selects
# the bottom 40% (clearly imperfect candidates) as negatives, and the
# noisy middle/top is skipped.
_NEGATIVE_SCORE_THRESHOLD = 2.0


class MochaExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MOCHA — Modeling Correctness with Human Annotations
    (Chen et al., EMNLP 2020).

    Dataset: anthonychen/mocha (HuggingFace)

    Verified schema (HF dataset card):
        - context:             str   (passage)
        - question:            str
        - reference:           str   (gold answer)
        - candidate:           str   (model-generated answer)
        - score:               float (1-5 averaged human judgement;
                                      -1 in the test split)
        - constituent_dataset: str   (source QA dataset; metadata)

    Splits: train (31069), validation (4009), test (6321).
    Loaded from the labelled `validation` split.

    Contrastive pair (user-authorised design):
        - positive = reference (definitionally correct gold answer)
        - negative = candidate, but only when score <= 2.0
          (bottom 40% of the 1-5 ordinal scale — clearly imperfect)
    Higher-score candidates are skipped to keep the contrast clean.
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MOCHA validation docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading anthonychen/mocha (limit={max_items})")
        dataset = load_dataset("anthonychen/mocha", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid MOCHA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MOCHA doc into a ContrastivePair, filtering by score.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            reference = str(doc.get("reference", "")).strip()
            candidate = str(doc.get("candidate", "")).strip()
            score = doc.get("score")

            if not context or not question or not reference or not candidate:
                return None
            if score is None:
                return None
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                return None
            if score_f > _NEGATIVE_SCORE_THRESHOLD:
                return None
            if reference == candidate:
                return None

            prompt = f"Passage: {context}\nQuestion: {question}\nAnswer:"
            return self._build_pair(prompt, reference, candidate)
        except Exception as exc:
            log.error("Error extracting MOCHA pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="mocha",
        )
