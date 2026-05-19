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

__all__ = ["SocialIqaHFExtractor"]

log = setup_logger(__name__)

task_names = ("socialiqa",)

# lighteval/siqa is the parquet mirror of Social IQA. The original
# allenai/social_i_qa repo still ships as a dataset script
# (`social_i_qa.py`), which newer `datasets` releases refuse with:
#   `RuntimeError: Dataset scripts are no longer supported, but found
#   social_i_qa.py`
# (job f718bd92 / socialiqa, local@ubuntu-server, 2026-05-19T05:46:52Z).
# lighteval/siqa carries the same schema verbatim — context / question /
# answerA / answerB / answerC / label — as train (33410) and validation
# (1954) parquet files, no script. Schema verified against the HF
# datasets-server first_rows API 2026-05-19.
_HF_DATASET = "lighteval/siqa"
_HF_SPLIT = "train"


class SocialIqaHFExtractor(HuggingFaceBenchmarkExtractor):
    """HF-direct extractor for Social IQA (`socialiqa`).

    Bypasses lm-eval-harness which routes `socialiqa` through the
    deprecated `allenai/social_i_qa` dataset script and now fails to
    load. Pair-extraction logic matches the original `SIQAExtractor`
    Format 1 path (the lm_eval variants `siqa` / `siqa_ca` use the same
    schema), so the contrastive pair semantics for `socialiqa` are
    unchanged — only the data source moves from lm-eval/script -> direct
    HF parquet.

    Schema (lighteval/siqa):
        context, question, answerA, answerB, answerC : str
        label                                        : str ("1" / "2" / "3")

    Pair:
        prompt   = "Context: <context>\\nQuestion: <question>"
        positive = choices[label - 1]
        negative = choices[(label - 1 + 1) mod 3]   (deterministic next)
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        max_items = self._normalize_limit(limit)

        log.info(f"Loading {_HF_DATASET} (split={_HF_SPLIT}, limit={max_items})")
        dataset = load_dataset(_HF_DATASET, split=_HF_SPLIT)

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is None:
                continue
            pairs.append(pair)
            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            log.warning("No valid SocialIQA pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answerA = str(doc.get("answerA", "")).strip()
            answerB = str(doc.get("answerB", "")).strip()
            answerC = str(doc.get("answerC", "")).strip()
            label_str = str(doc.get("label", "")).strip()

            if not all([context, question, answerA, answerB, answerC, label_str]):
                return None

            try:
                label_idx = int(label_str) - 1  # 1-indexed -> 0-indexed
            except (ValueError, TypeError):
                return None

            choices = [answerA, answerB, answerC]
            if not (0 <= label_idx < len(choices)):
                return None

            correct = choices[label_idx]
            incorrect = choices[(label_idx + 1) % len(choices)]

            prompt = f"Context: {context}\nQuestion: {question}"

            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting SocialIQA pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="socialiqa",
        )
