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

__all__ = ["WildbenchExtractor"]

log = setup_logger(__name__)

task_names = ("wildbench",)

# Abstention negatives — identical set used by SQuAD2Extractor /
# QuorefExtractor / WebglmExtractor. For chat, these read as refusals
# to follow the instruction.
_RC_ABSTAIN_NEGATIVES = (
    "The information is not provided in the background.",
    "This cannot be determined from the background.",
    "The background does not contain this information.",
)


class WildbenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for WildBench (Lin et al., allenai/WildBench), a real-world
    chat benchmark scored by a GPT-4 judge with per-instance checklists.

    Dataset: allenai/WildBench (config v2, test split)

    Verified schema (HF dataset card):
        - conversation_input: list[dict]   (chat turns, role + content)
        - references:         dict          (references["gpt-4"] = a GPT-4
                                             generation, the dataset's
                                             highest-quality reference)
        - checklist:          list[str]     (judge questions; unused)
        - primary_tag:        str
        - intent:             str

    There is no traditional gold answer (judge-based eval), but
    references["gpt-4"] is the strongest reference response the dataset
    itself ships and is a legitimate gold-proxy.

    Contrastive pair (user-authorised design):
        - prompt   = the final user-turn content from conversation_input
        - positive = references["gpt-4"]
        - negative = random.choice(_RC_ABSTAIN_NEGATIVES)
                     (structurally a refusal of the chat instruction)
    """

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from WildBench v2 test docs.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Loading allenai/WildBench v2 (limit={max_items})")
        dataset = load_dataset("allenai/WildBench", "v2", split="test")

        pairs: list[ContrastivePair] = []
        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid WildBench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single WildBench doc into a ContrastivePair.
        """
        try:
            convo = doc.get("conversation_input") or []
            if not isinstance(convo, list) or not convo:
                return None
            # The benchmark's instruction is the last user turn.
            user_turn = None
            for turn in reversed(convo):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    user_turn = turn
                    break
            if user_turn is None:
                user_turn = convo[-1] if isinstance(convo[-1], dict) else None
            if user_turn is None:
                return None
            user_text = str(user_turn.get("content", "")).strip()
            if not user_text:
                return None

            references = doc.get("references") or {}
            gpt4_ref = (
                str(references.get("gpt-4", "")).strip()
                if isinstance(references, dict)
                else ""
            )
            if not gpt4_ref:
                return None

            incorrect = random.choice(_RC_ABSTAIN_NEGATIVES)
            return self._build_pair(user_text, gpt4_ref, incorrect)
        except Exception as exc:
            log.error("Error extracting WildBench pair", exc_info=exc)
            return None

    @staticmethod
    def _build_pair(question: str, correct: str, incorrect: str) -> ContrastivePair:
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label="wildbench",
        )
