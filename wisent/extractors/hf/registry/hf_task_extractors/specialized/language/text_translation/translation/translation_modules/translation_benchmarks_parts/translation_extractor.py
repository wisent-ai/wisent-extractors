"""Parts of translation_benchmarks.py, split by the tama size splitter; translation_benchmarks.py imports every name back."""

from __future__ import annotations
import random
from typing import Any, Optional
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from ..translation_benchmarks import log


class TranslationExtractor(HuggingFaceBenchmarkExtractor):
    """
    Generic extractor for translation tasks.

    Can be configured for different language pairs and datasets.
    """

    evaluator_name = "generation"

    def __init__(
        self,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize Translation extractor.

        Args:
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'de', 'fr')
            dataset_name: HuggingFace dataset name
        """
        super().__init__()
        # Derive settings from task_name if available (e.g. "translation" -> defaults)
        task_name = getattr(self, "task_name", None)
        self.source_lang = source_lang if source_lang is not None else "en"
        self.target_lang = target_lang if target_lang is not None else "de"
        self.dataset_name = dataset_name if dataset_name is not None else "wmt/wmt14"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from translation dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # Build config from language pair
            config = f"{self.source_lang}-{self.target_lang}"
            docs = self.load_dataset(
                dataset_name=self.dataset_name,
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from {self.dataset_name} ({config})")
        except Exception as e:
            # Try reversed config
            try:
                config = f"{self.target_lang}-{self.source_lang}"
                docs = self.load_dataset(
                    dataset_name=self.dataset_name,
                    dataset_config=config,
                    split="test",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from {self.dataset_name} ({config})")
            except Exception as e2:
                log.error(f"Failed to load translation dataset: {e2}")
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
            # Standard translation format
            translation = doc.get("translation", {})
            source_text = translation.get(self.source_lang, "").strip()
            target_text = translation.get(self.target_lang, "").strip()

            if not source_text or not target_text:
                return None

            lang_names = {
                "en": "English",
                "de": "German",
                "fr": "French",
                "es": "Spanish",
                "ro": "Romanian",
                "cs": "Czech",
                "fi": "Finnish",
                "ru": "Russian",
                "zh": "Chinese",
            }

            source_name = lang_names.get(self.source_lang, self.source_lang.upper())
            target_name = lang_names.get(self.target_lang, self.target_lang.upper())

            prompt = f"Translate the following from {source_name} to {target_name}:\n{source_text}"

            correct_translation = target_text

            # Create incorrect by shuffling words
            words = target_text.split()
            if len(words) < 2:
                incorrect_translation = "[incorrect translation]"
            else:
                shuffled_words = words.copy()
                random.shuffle(shuffled_words)
                incorrect_translation = " ".join(shuffled_words)

            metadata = {
                "label": f"{self.dataset_name}_{self.source_lang}_{self.target_lang}",
                "source": self.dataset_name,
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
            log.error(f"Error extracting translation pair: {exc}", exc_info=True)
            return None
