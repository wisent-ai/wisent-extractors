"""Parts of evalita_llm.py, split by the tama size splitter; evalita_llm.py imports every name back."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..evalita_llm import _LOG


task_names = (
    "evalita-mp",      # Parent group (alias: Evalita-LLM) - all tasks
    "evalita-mp_gen",  # Only generative tasks subgroup
    "evalita-mp_mc",   # Only perplexity-based tasks subgroup
)

class EvalitaLlmExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Evalita-LLM benchmark - Italian LLM evaluation tasks.

    Evalita-LLM is a benchmark for evaluating Large Language Models on Italian.
    It includes both multiple-choice and generative tasks across various domains.

    This extractor handles the parent groups and individual tasks not covered by
    specific extractors (evalita-mp and evalita-sp have their own extractors).
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
        Build contrastive pairs from Evalita Llm docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Evalita Llm.
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
            log.warning("No valid Evalita Llm pairs extracted", extra={"task": task_name})

        return pairs

    from ..evalita_llm_methods.extract_pair_from_doc import _extract_pair_from_doc

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
