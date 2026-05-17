from __future__ import annotations

import ast
from typing import Any

from datasets import load_dataset

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import (
    NegativeResponse,
    PositiveResponse,
)
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger

__all__ = ["MusrExtractor"]

log = setup_logger(__name__)

task_names = ("musr",)

# Three subtask splits of TAUR-Lab/MuSR; matches lm-eval leaderboard_musr's
# subtask list (leaderboard_musr_{murder_mysteries,object_placements,team_allocation}).
_MUSR_SPLITS = ("murder_mysteries", "object_placements", "team_allocation")

# DOC_TO_TEXT verbatim from lm-eval lm_eval/tasks/leaderboard/musr/utils.py.
_DOC_TO_TEXT = "{narrative}\n\n{question}\n\n{choices}\nAnswer:"


class MusrExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MuSR — Multistep Soft Reasoning (Sprague et al., ICLR 2024,
    arXiv:2310.16049).

    MuSR is in lm-eval-harness as group `leaderboard_musr`, but no wisent
    extractor existed. Sourcing the underlying HF dataset directly (the same
    dataset the lm-eval task loads via dataset_path TAUR-Lab/MuSR).

    Dataset: TAUR-Lab/MuSR (HuggingFace)

    Verified schema (lm-eval utils.py + HF dataset card):
        - narrative:     str   (long-form, 3.78k-7.27k chars)
        - question:      str
        - choices:       str   (Python list literal; parsed via ast.literal_eval
                                exactly like lm-eval utils.py)
        - answer_index:  int   (0-based)
        - answer_choice: str   (gold answer text)
    Splits are the three subtasks: murder_mysteries (250),
    object_placements (256), team_allocation (250). All three are iterated.

    Multi-choice reasoning. Contrastive pair mirrors the in-repo integer-index
    MC pattern (ReclorExtractor / LogiQA / SIQA / COPA):
        - positive = choices[answer_index]
        - negative = choices[(answer_index + 1) % len(choices)]
    The prompt uses the verbatim lm-eval DOC_TO_TEXT format.
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from all three MuSR subtask splits.
        """
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []
        for split in _MUSR_SPLITS:
            log.info(f"Loading TAUR-Lab/MuSR split={split}")
            dataset = load_dataset("TAUR-Lab/MuSR", split=split)
            for doc in dataset:
                pair = self._extract_pair_from_doc(doc)
                if pair is not None:
                    pairs.append(pair)
                    if max_items is not None and len(pairs) >= max_items:
                        return pairs

        if not pairs:
            log.warning("No valid MuSR pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single MuSR doc into a ContrastivePair.
        Returns None when required fields are missing or malformed.
        """
        try:
            narrative = str(doc.get("narrative", "")).strip()
            question = str(doc.get("question", "")).strip()
            choices_field = doc.get("choices")
            answer_index = doc.get("answer_index")

            if not narrative or not question or choices_field is None:
                return None

            # `choices` is a string literal of a Python list — same parsing
            # lm-eval/tasks/leaderboard/musr/utils.py uses.
            try:
                choices_list = ast.literal_eval(choices_field) if isinstance(
                    choices_field, str
                ) else list(choices_field)
            except Exception:
                return None

            if (
                not isinstance(choices_list, (list, tuple))
                or len(choices_list) < 2
                or not isinstance(answer_index, int)
                or not (0 <= answer_index < len(choices_list))
            ):
                return None

            correct = str(choices_list[answer_index]).strip()
            incorrect = str(
                choices_list[(answer_index + 1) % len(choices_list)]
            ).strip()
            if not correct or not incorrect:
                return None

            # Render enumerated choices like the lm-eval utils.py.
            rendered_choices = "".join(
                f"{i + 1} - {c}\n" for i, c in enumerate(choices_list)
            )
            prompt = _DOC_TO_TEXT.format(
                narrative=narrative, question=question, choices=rendered_choices
            )
            return self._build_pair(prompt, correct, incorrect)
        except Exception as exc:
            log.error("Error extracting MuSR pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="musr",
        )
