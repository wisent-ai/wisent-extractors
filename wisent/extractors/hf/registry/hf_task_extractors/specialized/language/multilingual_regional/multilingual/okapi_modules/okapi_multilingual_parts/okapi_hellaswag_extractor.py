"""Parts of okapi_multilingual.py, split by the tama size splitter; okapi_multilingual.py imports every name back."""

from __future__ import annotations
import random
from typing import Any
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.config_tools.constants import INDEX_FIRST
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from ..okapi_multilingual import log
from .okapi_mmlu_extractor import _OKAPI_LANGS, _fetch_okapi_parquet


class OkapiHellaswagExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi HellaSwag - Multilingual HellaSwag benchmark.

    Dataset: jon-tow/okapi_hellaswag on HuggingFace

    Multilingual translation of HellaSwag commonsense inference benchmark
    across many languages.
    """

    evaluator_name = "okapi_hellaswag"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi HellaSwag extractor.

        Args:
            language: Optional language filter
        """
        super().__init__()
        task_name = getattr(self, "task_name", None)
        if language is not None:
            self.language = language
        elif task_name:
            parts = task_name.split("_")
            if len(parts) >= 3 and parts[-1] not in ("multilingual", "hellaswag"):
                self.language = parts[-1]
            else:
                self.language = None
        else:
            self.language = None

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi HellaSwag dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        config = self.language if self.language else "de"
        docs = None
        for ds_name in ["jon-tow/okapi_hellaswag", "lighteval/okapi_hellaswag"]:
            try:
                docs = self.load_dataset(
                    dataset_name=ds_name,
                    dataset_config=config,
                    split="validation",
                    limit=max_items,
                    trust_remote_code=True,
                )
                log.info(f"Loaded {len(docs)} examples from {ds_name} ({config})")
                break
            except Exception as e:
                log.debug(f"Failed to load {ds_name}: {e}")
        if not docs:
            # Fallback: direct CDN parquet download bypassing rate-limited API
            log.warning("load_dataset failed for all sources; falling back to direct CDN parquet")
            languages = [config] if self.language else _OKAPI_LANGS
            raw = []
            for lang in languages:
                items = _fetch_okapi_parquet("jon-tow/okapi_hellaswag", lang, "validation")
                if items:
                    log.info(f"Direct CDN fetched {len(items)} items for okapi_hellaswag/{lang}")
                    raw.extend(items)
                if max_items is not None and len(raw) >= max_items:
                    break
            docs = raw

        if not docs:
            log.error("Failed to load Okapi HellaSwag from any source (including direct CDN)")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            ctx = doc.get("ctx", doc.get("context", "")).strip()
            endings = doc.get("endings", [])
            raw_label = doc.get("label", INDEX_FIRST)
            label = int(raw_label) if str(raw_label).isdigit() else INDEX_FIRST

            if not ctx or not endings:
                return None

            # Build completion prompt
            choice_letters = ['A', 'B', 'C', 'D']
            choices_text = "\n".join(
                f"{choice_letters[i]}. {e}" for i, e in enumerate(endings[:4])
            )

            task_prompt = f"""Complete the following:

{ctx}

Options:
{choices_text}

Most likely completion:"""

            # Correct answer
            if isinstance(label, int) and label < len(endings):
                correct = choice_letters[label]
            else:
                correct = "A"

            # Incorrect answer
            wrong_indices = [i for i in range(len(endings)) if i != label]
            incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            metadata = {
                "label": "okapi_hellaswag",
                "source": "jon-tow/okapi_hellaswag",
                "language": self.language or "multilingual",
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Okapi HellaSwag pair: {exc}", exc_info=True)
            return None
