"""Parts of french_bench.py, split by the tama size splitter; french_bench.py imports every name back."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..french_bench import _LOG


task_names = (
    "french_bench",
    "french_bench_arc_challenge",
    "french_bench_boolqa",
    "french_bench_extra",
    "french_bench_fquadv2",
    "french_bench_fquadv2_bool",
    "french_bench_fquadv2_genq",
    "french_bench_fquadv2_hasAns",
    "french_bench_gen",
    "french_bench_grammar",
    "french_bench_hellaswag",
    "french_bench_mc",
    "french_bench_multifquad",
    "french_bench_opus_perplexity",
    "french_bench_orangesum_abstract",
    "french_bench_orangesum_title",
    "french_bench_perplexity",
    "french_bench_reading_comp",
    "french_bench_topic_based_nli",
    "french_bench_trivia",
    "french_bench_vocab",
    "french_bench_wikitext_fr",
    "french_bench_xnli",
)

class FrenchBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the French Bench benchmark."""


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
        Build contrastive pairs from French Bench docs.

        Args:
            lm_eval_task_data: lm-eval task instance for French Bench.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        if lm_eval_task_data is None:
            from datasets import load_dataset
            task_name = getattr(self, "task_name", "")
            docs = []
            if task_name == "french_bench_fquadv2_hasAns":
                for s in ("test_hasAns", "valid_hasAns"):
                    try:
                        ds = load_dataset("manu/fquad2_test", split=s, trust_remote_code=True)
                        docs.extend(list(ds))
                    except Exception:
                        continue
            if max_items:
                docs = docs[:max_items]
        else:
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
            log.warning("No valid French Bench pairs extracted", extra={"task": task_name})

        return pairs

    from ..french_bench_methods.extract_pair_from_doc import _extract_pair_from_doc

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
