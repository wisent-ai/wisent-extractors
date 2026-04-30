from __future__ import annotations

import os
from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

# Languages that the okapi_truthfulqa dataset supports (per loading script)
_OKAPI_TQA_LANGS = (
    "ar bn ca da de es eu fr gu hi hr hu hy id it kn ml mr ne nl pt ro ru sk sr sv ta te uk vi zh"
).split()


def _hf_headers_tqa() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_okapi_tqa_json(lang: str) -> list[dict] | None:
    """Direct CDN fetch for jon-tow/okapi_truthfulqa bypassing rate-limited API.

    URL pattern: https://huggingface.co/datasets/jon-tow/okapi_truthfulqa/resolve/main/data/{lang}_validation.json
    """
    import requests

    url = (
        f"https://huggingface.co/datasets/jon-tow/okapi_truthfulqa/resolve/main/"
        f"data/{lang}_validation.json"
    )
    try:
        resp = requests.get(url, headers=_hf_headers_tqa(), timeout=120, allow_redirects=True)
        if resp.status_code == 200:
            return resp.json()
        log.debug(f"Direct fetch {url} -> HTTP {resp.status_code}")
    except Exception as exc:
        log.debug(f"Direct fetch {url} failed: {exc}")
    return None


class OkapiTruthfulQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Okapi TruthfulQA - Multilingual TruthfulQA benchmark.

    Dataset: jon-tow/okapi_truthfulqa on HuggingFace

    Multilingual translation of TruthfulQA benchmark measuring truthfulness
    across 26 languages.
    """

    evaluator_name = "okapi_truthfulqa"

    def __init__(self, language: str | None = None):
        """
        Initialize Okapi TruthfulQA extractor.

        Args:
            language: Optional language filter
        """
        super().__init__()
        task_name = getattr(self, "task_name", None)
        if language is not None:
            self.language = language
        elif task_name:
            parts = task_name.split("_")
            if len(parts) >= 3 and parts[-1] not in ("multilingual", "truthfulqa"):
                self.language = parts[-1]
            else:
                self.language = None
        else:
            self.language = None

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Okapi TruthfulQA dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        config = self.language if self.language else "es"
        docs = None
        for ds_name, cfg, split in [
            ("jon-tow/okapi_truthfulqa", config, "validation"),
            ("alvarobartt/truthfulqa-okapi-eval-es", None, "validation"),
        ]:
            try:
                kwargs = dict(split=split, limit=max_items, trust_remote_code=True)
                if cfg:
                    kwargs["dataset_config"] = cfg
                docs = self.load_dataset(dataset_name=ds_name, **kwargs)
                log.info(f"Loaded {len(docs)} examples from {ds_name}")
                self._dataset_format = "mc_targets"
                break
            except Exception as e:
                log.debug(f"Failed to load {ds_name}: {e}")
        if not docs:
            # Fallback: direct CDN download bypassing rate-limited API
            log.warning("load_dataset failed for all sources; falling back to direct CDN download")
            languages = [config] if self.language else _OKAPI_TQA_LANGS
            raw = []
            for lang in languages:
                items = _fetch_okapi_tqa_json(lang)
                if items:
                    log.info(f"Direct CDN fetched {len(items)} items for okapi_truthfulqa/{lang}")
                    # Normalize loader-script flat schema to nested mc1_targets/mc2_targets schema
                    for it in items:
                        normalized = {"question": it.get("question", "")}
                        if "mc1_targets_choices" in it:
                            normalized["mc1_targets"] = {
                                "choices": it.get("mc1_targets_choices", []),
                                "labels": it.get("mc1_targets_labels", []),
                            }
                        else:
                            normalized["mc1_targets"] = it.get("mc1_targets", {})
                        if "mc2_targets_choices" in it:
                            normalized["mc2_targets"] = {
                                "choices": it.get("mc2_targets_choices", []),
                                "labels": it.get("mc2_targets_labels", []),
                            }
                        else:
                            normalized["mc2_targets"] = it.get("mc2_targets", {})
                        raw.append(normalized)
                if max_items is not None and len(raw) >= max_items:
                    break
            docs = raw
            if docs:
                self._dataset_format = "mc_targets"

        if not docs:
            log.error("Failed to load Okapi TruthfulQA from any source (including direct CDN)")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        Supports two formats:
        1. Original: question, best_answer, incorrect_answers
        2. MC format: question, mc1_targets (with choices and labels)
        """
        try:
            question = doc.get("question", "").strip()

            if not question:
                return None

            # Handle mc1_targets format
            mc1_targets = doc.get("mc1_targets", {})
            if mc1_targets and "choices" in mc1_targets:
                choices = mc1_targets.get("choices", [])
                labels = mc1_targets.get("labels", [])

                if not choices:
                    return None

                # Find correct answer (label=1) and incorrect answer (label=0)
                correct = None
                incorrect = None
                for i, (choice, label) in enumerate(zip(choices, labels)):
                    if label == 1 and correct is None:
                        correct = choice.strip()
                    elif label == 0 and incorrect is None:
                        incorrect = choice.strip()

                if not correct:
                    correct = choices[0].strip() if choices else "Unknown"
                if not incorrect:
                    incorrect = choices[-1].strip() if len(choices) > 1 else "I don't know"
            else:
                # Original format
                best_answer = doc.get("best_answer", "").strip()
                incorrect_answers = doc.get("incorrect_answers", [])

                if not best_answer:
                    return None

                correct = best_answer
                if incorrect_answers and isinstance(incorrect_answers, list):
                    incorrect = incorrect_answers[0] if incorrect_answers else "I don't know."
                else:
                    incorrect = "I don't know."

            task_prompt = f"""Question: {question}

Provide a truthful answer:"""

            metadata = {
                "label": "okapi_truthfulqa",
                "source": "jon-tow/okapi_truthfulqa",
                "language": self.language or "multilingual",
                "is_multilingual_benchmark": True,
                "is_truthfulness_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Okapi TruthfulQA pair: {exc}", exc_info=True)
            return None
