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

__all__ = ["ScitailExtractor"]

log = setup_logger(__name__)

task_names = ("scitail",)

# Binary NLI label space, verified from HF allenai/scitail dataset card.
_LABEL_SPACE = ("entails", "neutral")


class ScitailExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SciTail (Khot et al., 2018) — textual entailment dataset
    built from science multiple-choice exams + web sentences.

    Not present in lm-eval-harness, so sourced from HuggingFace.

    Dataset: allenai/scitail (config: tsv_format)

    Verified schema (HF dataset card, tsv_format):
        - premise:    str
        - hypothesis: str
        - label:      str in {"entails", "neutral"}
    Splits: train (23097), validation (1304), test (2126) — all labelled.

    Binary NLI. Contrastive pair mirrors the CB NLI extractor pattern
    (evaluation/general/task_types/nli/cb.py) — table.tex 2C-Flip:
        - prompt   = "{premise}\\nQuestion: {hypothesis}."
        - positive = gold label string
        - negative = the other label
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SciTail validation docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/scitail tsv_format (limit={max_items})")
        dataset = load_dataset("allenai/scitail", "tsv_format", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SciTail pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single SciTail doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            premise = str(doc.get("premise", "")).strip()
            hypothesis = str(doc.get("hypothesis", "")).strip()
            label = str(doc.get("label", "")).strip()

            if not premise or not hypothesis or label not in _LABEL_SPACE:
                return None

            idx = _LABEL_SPACE.index(label)
            correct = _LABEL_SPACE[idx]
            incorrect = _LABEL_SPACE[(idx + 1) % len(_LABEL_SPACE)]

            prompt = f"{premise}\nQuestion: {hypothesis}."
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting SciTail pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="scitail",
        )
