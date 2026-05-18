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

__all__ = ["PubmedEntailmentExtractor"]

log = setup_logger(__name__)

task_names = ("pubmed_entailment",)

# 3-class NLI label space used by MEDIQA-NLI / MedNLI / SNLI / MNLI.
_LABEL_SPACE = ("entailment", "neutral", "contradiction")


class PubmedEntailmentExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for the medical-NLI benchmark mapped to `pubmed_entailment`.

    There is no HF dataset literally named `pubmed_entailment`; the
    user-authorised mapping is to bigbio/mediqa_nli (MEDIQA-NLI shared task
    at ACL-BioNLP 2019), the canonical freely-accessible medical-domain
    entailment benchmark. (MedNLI itself requires PhysioNet credentialing
    and is not openly accessible on HF.)

    Dataset: bigbio/mediqa_nli (HuggingFace)

    Verified schema (BigBio / MEDIQA-NLI):
        - premise:    str
        - hypothesis: str
        - label:      str in {"entailment", "neutral", "contradiction"}

    3-class medical NLI. Contrastive pair mirrors the CB NLI extractor
    pattern (evaluation/general/task_types/nli/cb.py):
        - prompt   = "{premise}\\nQuestion: {hypothesis}."
        - positive = gold label string
        - negative = next-cyclic label (3-class)
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MEDIQA-NLI validation docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading bigbio/mediqa_nli (limit={max_items})")
        dataset = load_dataset("bigbio/mediqa_nli", split="validation")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pubmed_entailment pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MEDIQA-NLI doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            premise = str(doc.get("premise", "")).strip()
            hypothesis = str(doc.get("hypothesis", "")).strip()
            label = str(doc.get("label", "")).strip().lower()

            if not premise or not hypothesis or label not in _LABEL_SPACE:
                return None

            idx = _LABEL_SPACE.index(label)
            correct = _LABEL_SPACE[idx]
            incorrect = _LABEL_SPACE[(idx + 1) % len(_LABEL_SPACE)]

            prompt = f"{premise}\nQuestion: {hypothesis}."
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting pubmed_entailment pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="pubmed_entailment",
        )
