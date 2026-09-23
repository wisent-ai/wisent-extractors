"""Parts of noreval_mc.py, split by the tama size splitter; noreval_mc.py imports every name back."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..noreval_mc import _LOG


task_names = (
    # ncb
    "ncb",
    # norbelebele
    "norbelebele",
    "norbelebele_p0", "norbelebele_p1", "norbelebele_p2", "norbelebele_p3", "norbelebele_p4",
    # norcommonsenseqa
    "norcommonsenseqa_nno", "norcommonsenseqa_nob",
    "norcommonsenseqa_nno_p0", "norcommonsenseqa_nno_p1", "norcommonsenseqa_nno_p2", "norcommonsenseqa_nno_p3", "norcommonsenseqa_nno_p4",
    "norcommonsenseqa_nob_p0", "norcommonsenseqa_nob_p1", "norcommonsenseqa_nob_p2", "norcommonsenseqa_nob_p3", "norcommonsenseqa_nob_p4",
    # norec
    "norec_document_p0", "norec_document_p1", "norec_document_p2", "norec_document_p3", "norec_document_p4",
    "norec_sentence_p0", "norec_sentence_p1", "norec_sentence_p2", "norec_sentence_p3", "norec_sentence_p4",
    # noropenbookqa
    "noropenbookqa_nno", "noropenbookqa_nob",
    "noropenbookqa_nno_p0", "noropenbookqa_nno_p1", "noropenbookqa_nno_p2", "noropenbookqa_nno_p3", "noropenbookqa_nno_p4",
    "noropenbookqa_nob_p0", "noropenbookqa_nob_p1", "noropenbookqa_nob_p2", "noropenbookqa_nob_p3", "noropenbookqa_nob_p4",
    # nortruthfulqa_mc
    "nortruthfulqa_mc_nno", "nortruthfulqa_mc_nob",
    "nortruthfulqa_mc_nno_p0", "nortruthfulqa_mc_nno_p1", "nortruthfulqa_mc_nno_p2", "nortruthfulqa_mc_nno_p3", "nortruthfulqa_mc_nno_p4",
    "nortruthfulqa_mc_nob_p0", "nortruthfulqa_mc_nob_p1", "nortruthfulqa_mc_nob_p2", "nortruthfulqa_mc_nob_p3", "nortruthfulqa_mc_nob_p4",
    # nrk_quiz_qa
    "nrk_quiz_qa_nno", "nrk_quiz_qa_nob",
    "nrk_quiz_qa_nno_p0", "nrk_quiz_qa_nno_p1", "nrk_quiz_qa_nno_p2", "nrk_quiz_qa_nno_p3", "nrk_quiz_qa_nno_p4",
    "nrk_quiz_qa_nob_p0", "nrk_quiz_qa_nob_p1", "nrk_quiz_qa_nob_p2", "nrk_quiz_qa_nob_p3", "nrk_quiz_qa_nob_p4",
)
class NorevalMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Noreval multiple-choice benchmarks.

    Handles three different Norwegian multiple-choice formats:
    1. NCB: Simple correct/wrong pairs for grammar correction
    2. NorTruthfulQA: TruthfulQA-style with question + mc1_targets {choices, labels}
    3. NRK Quiz QA: Standard multiple-choice with question + choices + answer
    """


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Noreval docs.

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
            log.warning("No valid Noreval pairs extracted", extra={"task": task_name})

        return pairs

    from ..noreval_mc_methods.extract_pair_from_doc import _extract_pair_from_doc

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
