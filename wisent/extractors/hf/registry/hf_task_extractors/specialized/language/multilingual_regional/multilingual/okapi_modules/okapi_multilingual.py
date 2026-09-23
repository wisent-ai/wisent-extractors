"""Extractors for Okapi multilingual benchmarks (MMLU, HellaSwag, TruthfulQA)."""
from __future__ import annotations

import json
import os
import random
from typing import Any

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.config_tools.constants import INDEX_FIRST, SENSOR_LAST_OFFSET
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "OkapiMMLUExtractor",
    "OkapiHellaswagExtractor",
    "OkapiTruthfulQAExtractor",
]

log = setup_logger(__name__)


_OKAPI_LANGS = (
    "ar bn ca da de es eu fr gu hi hr hu hy id it kn ml mr ne nl pt ro ru sk sr sv ta te uk vi zh"
).split()


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_okapi_json(repo: str, lang: str, split_filename: str) -> list[dict] | None:
    """Fetch JSON data from HF CDN directly bypassing the rate-limited API.

    URL pattern (works for jon-tow/okapi_mmlu and jon-tow/okapi_truthfulqa):
        https://huggingface.co/datasets/{repo}/resolve/main/data/{lang}_{split}.json

    Returns parsed JSON list or None on failure.
    """
    import requests

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/{lang}_{split_filename}.json"
    try:
        resp = requests.get(url, headers=_hf_headers(), timeout=120, allow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        log.debug(f"Direct fetch {url} -> HTTP {resp.status_code}")
    except Exception as exc:
        log.debug(f"Direct fetch {url} failed: {exc}")
    return None


def _fetch_okapi_parquet(repo: str, lang: str, split: str) -> list[dict] | None:
    """Fetch parquet data from HF auto-converted parquet endpoint bypassing the API.

    URL pattern:
        https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{lang}/{split}/0000.parquet

    Returns list of dicts or None on failure.
    """
    import io

    import pyarrow.parquet as pq
    import requests

    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/"
        f"{lang}/{split}/0000.parquet"
    )
    try:
        resp = requests.get(url, headers=_hf_headers(), timeout=180, allow_redirects=True)
        if resp.status_code != 200:
            log.debug(f"Direct parquet fetch {url} -> HTTP {resp.status_code}")
            return None
        table = pq.read_table(io.BytesIO(resp.content))
        return table.to_pylist()
    except Exception as exc:
        log.debug(f"Direct parquet fetch {url} failed: {exc}")
        return None


class OkapiMMLUExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi MMLU - Multilingual MMLU benchmark.

    Dataset: jon-tow/okapi_mmlu on HuggingFace

    Multilingual translation of MMLU (Measuring Massive Multitask Language
    Understanding) covering many tasks across many languages.
    """

    evaluator_name = "okapi_mmlu"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi MMLU extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'es')
        """
        super().__init__()
        # Derive language from task_name if not provided
        # e.g. "okapi_mmlu_multilingual" -> None (use all), "okapi_mmlu_de" -> "de"
        task_name = getattr(self, "task_name", None)
        if language is not None:
            self.language = language
        elif task_name:
            parts = task_name.split("_")
            if len(parts) >= 3 and parts[-1] not in ("multilingual", "mmlu", "hellaswag", "truthfulqa"):
                self.language = parts[-1]
            else:
                self.language = None
        else:
            self.language = None

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi MMLU dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        config = self.language if self.language else "de"
        docs = None
        for ds_name in ["jon-tow/okapi_mmlu", "lighteval/okapi_mmlu", "open-llm-leaderboard/okapi_mmlu"]:
            try:
                docs = self.load_dataset(
                    dataset_name=ds_name,
                    dataset_config=config,
                    split="test",
                    limit=max_items,
                    trust_remote_code=True,
                )
                log.info(f"Loaded {len(docs)} examples from {ds_name} ({config})")
                break
            except Exception as e:
                log.debug(f"Failed to load {ds_name}: {e}")
        if not docs:
            # Fallback: direct CDN download bypassing rate-limited API
            log.warning("load_dataset failed for all sources; falling back to direct CDN download")
            languages = [config] if self.language else _OKAPI_LANGS
            raw = []
            for lang in languages:
                items = _fetch_okapi_json("jon-tow/okapi_mmlu", lang, "test")
                if items:
                    log.info(f"Direct CDN fetched {len(items)} items for okapi_mmlu/{lang}")
                    for it in items:
                        # Normalize loader-script schema -> doc schema
                        raw.append({
                            "question": it.get("instruction", ""),
                            "choices": [
                                it.get("option_a", ""),
                                it.get("option_b", ""),
                                it.get("option_c", ""),
                                it.get("option_d", ""),
                            ],
                            "answer": it.get("answer", "A"),
                            "id": it.get("id", ""),
                        })
                if max_items is not None and len(raw) >= max_items:
                    break
            docs = raw

        if not docs:
            log.error("Failed to load Okapi MMLU from any source (including direct CDN)")
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
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            raw_answer = doc.get("answer", INDEX_FIRST)
            if isinstance(raw_answer, str) and len(raw_answer) == SENSOR_LAST_OFFSET and raw_answer.isalpha():
                answer_idx = ord(raw_answer.upper()) - ord('A')
            elif str(raw_answer).isdigit():
                answer_idx = int(raw_answer)
            else:
                answer_idx = INDEX_FIRST

            if not question or not choices:
                return None

            # Build multiple choice prompt
            choice_letters = ['A', 'B', 'C', 'D']
            choices_text = "\n".join(
                f"{choice_letters[i]}. {c}" for i, c in enumerate(choices[:4])
            )

            task_prompt = f"""Question: {question}

{choices_text}

Answer:"""

            # Correct answer
            if isinstance(answer_idx, int) and answer_idx < len(choices):
                correct = choice_letters[answer_idx]
            else:
                correct = "A"

            # Incorrect answer
            wrong_indices = [i for i in range(len(choices)) if i != answer_idx]
            incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            metadata = {
                "label": "okapi_mmlu",
                "source": "jon-tow/okapi_mmlu",
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
            log.error(f"Error extracting Okapi MMLU pair: {exc}", exc_info=True)
            return None


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




# Re-export from split module
from wisent.extractors.hf.hf_task_extractors.okapi_multilingual_truthfulqa import (
    OkapiTruthfulQAExtractor,
)
