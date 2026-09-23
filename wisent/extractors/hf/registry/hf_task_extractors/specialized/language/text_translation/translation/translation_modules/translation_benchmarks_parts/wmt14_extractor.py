"""Parts of translation_benchmarks.py, split by the tama size splitter; translation_benchmarks.py imports every name back."""

from __future__ import annotations
import random
from typing import Any, Optional
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from ..translation_benchmarks import log


class WMT14Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for WMT14 translation benchmark.

    Dataset: wmt14 on HuggingFace

    WMT14 (Workshop on Machine Translation 2014) contains parallel corpora
    for English-French and English-German translation tasks.
    """

    evaluator_name = "generation"

    # WMT14 HuggingFace dataset name (with org prefix) and available configs.
    _DATASET_NAME = "wmt/wmt14"
    # Configs follow target-source convention: fr-en, de-en, etc.
    _KNOWN_CONFIGS = ("cs-en", "de-en", "fr-en", "hi-en", "ru-en")

    def __init__(self, lang_pair: Optional[str] = None):
        """
        Initialize WMT14 extractor.

        Args:
            lang_pair: Language pair (e.g., 'en-fr', 'fr-en', 'en-de', 'de-en')
        """
        super().__init__()
        # Derive lang_pair from task_name if not explicitly provided.
        # task_name is set by the registry, e.g. "wmt14_en_fr" -> "en-fr"
        task_name = getattr(self, "task_name", None)
        if lang_pair is not None:
            self.lang_pair = lang_pair
        elif task_name and "_" in task_name:
            # e.g. "wmt14_en_fr" -> strip prefix -> "en_fr" -> "en-fr"
            suffix = task_name.split("_", 1)[1] if task_name.startswith("wmt") else task_name
            self.lang_pair = suffix.replace("_", "-")
        else:
            self.lang_pair = lang_pair if lang_pair is not None else "fr-en"
        parts = self.lang_pair.split("-")
        self.source_lang = parts[0] if len(parts) > 0 else "en"
        self.target_lang = parts[1] if len(parts) > 1 else "fr"

    def _resolve_config(self) -> str | None:
        """Find the correct HuggingFace config for the requested language pair.

        WMT datasets use target-source convention (e.g. 'fr-en' not 'en-fr').
        Try both orderings and match against known configs.
        """
        candidates = [
            f"{self.source_lang}-{self.target_lang}",
            f"{self.target_lang}-{self.source_lang}",
        ]
        for c in candidates:
            if c in self._KNOWN_CONFIGS:
                return c
        # Fallback: return the first candidate (will be tried by load_dataset)
        return candidates[0]

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from WMT14 dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        config = self._resolve_config()
        try:
            docs = self.load_dataset(
                dataset_name=self._DATASET_NAME,
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from WMT14 ({config})")
        except Exception as e:
            # Try the other config ordering
            alt = f"{self.target_lang}-{self.source_lang}" if config == f"{self.source_lang}-{self.target_lang}" else f"{self.source_lang}-{self.target_lang}"
            try:
                docs = self.load_dataset(
                    dataset_name=self._DATASET_NAME,
                    dataset_config=alt,
                    split="test",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from WMT14 ({alt})")
                config = alt
            except Exception as e2:
                log.error(f"Failed to load WMT14: {e2}")
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
            translation = doc.get("translation", {})
            source_text = translation.get(self.source_lang, "").strip()
            target_text = translation.get(self.target_lang, "").strip()

            if not source_text or not target_text:
                return None

            lang_names = {
                "en": "English",
                "fr": "French",
                "de": "German",
            }

            source_name = lang_names.get(self.source_lang, self.source_lang.upper())
            target_name = lang_names.get(self.target_lang, self.target_lang.upper())

            prompt = f"Translate the following from {source_name} to {target_name}:\n{source_text}"

            correct_translation = target_text

            words = target_text.split()
            if len(words) < 2:
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = " ".join(shuffled_words)

            metadata = {
                "label": f"wmt14_{self.source_lang}_{self.target_lang}",
                "source": "wmt14",
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_translation,
                incorrect=incorrect_translation,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting WMT14 pair: {exc}", exc_info=True)
            return None
