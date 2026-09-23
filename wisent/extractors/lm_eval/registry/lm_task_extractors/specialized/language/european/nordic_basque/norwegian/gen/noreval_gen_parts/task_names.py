"""Parts of noreval_gen.py, split by the tama size splitter; noreval_gen.py imports every name back."""

from __future__ import annotations
import random
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..noreval_gen import _LOG


task_names = (
    # ask_gec
    "ask_gec_p0", "ask_gec_p1", "ask_gec_p2", "ask_gec_p3", "ask_gec_p4",
    # norrewrite/norsummarize instruct
    "norrewrite_instruct",
    "norsummarize_instruct",
    # norsumm
    "norsumm_nno_p0", "norsumm_nno_p1", "norsumm_nno_p2", "norsumm_nno_p3", "norsumm_nno_p4", "norsumm_nno_p5",
    "norsumm_nob_p0", "norsumm_nob_p1", "norsumm_nob_p2", "norsumm_nob_p3", "norsumm_nob_p4", "norsumm_nob_p5",
    # nortruthfulqa_gen
    "nortruthfulqa_gen_nno_p0", "nortruthfulqa_gen_nno_p1", "nortruthfulqa_gen_nno_p2", "nortruthfulqa_gen_nno_p3", "nortruthfulqa_gen_nno_p4",
    "nortruthfulqa_gen_nob_p0", "nortruthfulqa_gen_nob_p1", "nortruthfulqa_gen_nob_p2", "nortruthfulqa_gen_nob_p3", "nortruthfulqa_gen_nob_p4",
    # tatoeba
    "tatoeba_eng_nno_p0", "tatoeba_eng_nno_p1", "tatoeba_eng_nno_p2", "tatoeba_eng_nno_p3",
    "tatoeba_eng_nob_p0", "tatoeba_eng_nob_p1", "tatoeba_eng_nob_p2", "tatoeba_eng_nob_p3",
    "tatoeba_nno_eng_p0", "tatoeba_nno_eng_p1", "tatoeba_nno_eng_p2", "tatoeba_nno_eng_p3",
    "tatoeba_nob_eng_p0", "tatoeba_nob_eng_p1", "tatoeba_nob_eng_p2", "tatoeba_nob_eng_p3",
)
class NorevalGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Noreval generation benchmarks.

    Handles multiple Norwegian generation formats:
    1. ask_gec: Grammar error correction {source, correction}
    2. noridiom: Idiom completion {idiom_start, accepted_completions}
    3. norquad: QA {context, question, answers}
    4. norrewrite_instruct/norsummarize_instruct: Text transformation {prompt, context, target}
    5. nortruthfulqa_gen: TruthfulQA generation {question, correct_answers, incorrect_answers}

    For generation tasks, creates synthetic negative responses by:
    - Using source text as negative for correction tasks
    - Using wrong completions or shuffled text for other tasks
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Noreval generation docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Noreval.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Noreval generation pairs extracted", extra={"task": task_name})

        return pairs

    from ..noreval_gen_methods.extract_pair_from_doc import _extract_pair_from_doc

    @staticmethod
    def _create_shuffled_text(text: str) -> str:
        """Create a synthetic negative response by shuffling words or sentences."""
        # Try to shuffle sentences first
        sentences = text.split(". ")
        if len(sentences) > 2:
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            return ". ".join(shuffled)

        # If only one sentence, shuffle words
        words = text.split()
        if len(words) > 3:
            shuffled = words.copy()
            random.shuffle(shuffled)
            return " ".join(shuffled)

        # If too short, just return a generic wrong answer
        return "feil svar"  # "wrong answer" in Norwegian

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
        )
