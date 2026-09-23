"""Parts of aradice.py, split by the tama size splitter; aradice.py imports every name back."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import bind
if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
from ..aradice import _LOG


class AradiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AraDiCE benchmark - Arabic dialect multiple-choice tasks."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Build case-insensitive doc lookup
            doc_lower = {k.lower(): v for k, v in doc.items()}

            # --- TruthfulQA MC1 format: question + mc1_targets {choices, labels} ---
            if "mc1_targets" in doc_lower and isinstance(doc_lower.get("mc1_targets"), dict):
                question = str(doc_lower.get("question", "")).strip()
                mc1 = doc_lower["mc1_targets"]
                choices = [str(c).strip() for c in mc1.get("choices", [])]
                labels = mc1.get("labels", [])
                # The correct answer is the choice with label == 1
                answer_idx = next((i for i, l in enumerate(labels) if l == 1), 0)
                if question and choices and 0 <= answer_idx < len(choices):
                    return self._build_pair(
                        question=question,
                        correct=choices[answer_idx],
                        incorrect=choices[(answer_idx + 1) % len(choices)],
                        metadata={"label": "aradice"},
                    )
                return None

            # --- PIQA format: goal + sol1 + sol2 + label ---
            if "goal" in doc_lower and "sol1" in doc_lower:
                question = str(doc_lower["goal"]).strip()
                choices = [
                    str(doc_lower.get("sol1", "")).strip(),
                    str(doc_lower.get("sol2", "")).strip(),
                ]
                answer = doc_lower.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, float)) else 0
                if question and len(choices) == 2 and 0 <= answer_idx < 2:
                    return self._build_pair(
                        question=question,
                        correct=choices[answer_idx],
                        incorrect=choices[(answer_idx + 1) % 2],
                        metadata={"label": "aradice"},
                    )
                return None

            # --- OpenBookQA format: question.stem + question.choices [{label,text}] + answerKey ---
            if "question" in doc and isinstance(doc.get("question"), dict):
                q_data = doc["question"]
                question = str(q_data.get("stem", "")).strip()
                raw_choices = q_data.get("choices", [])
                choices = [str(c.get("text", "")).strip() for c in raw_choices]
                labels = [str(c.get("label", "")).strip() for c in raw_choices]
                answer_key = str(doc.get("answerKey", "A")).strip()
                answer_idx = labels.index(answer_key) if answer_key in labels else 0
                if question and choices and 0 <= answer_idx < len(choices):
                    return self._build_pair(
                        question=question,
                        correct=choices[answer_idx],
                        incorrect=choices[(answer_idx + 1) % len(choices)],
                        metadata={"label": "aradice"},
                    )
                return None

            # --- Standard MC format ---
            # Try multiple format patterns for question
            question = (doc_lower.get("question", doc_lower.get("query", doc_lower.get("input",
                        doc_lower.get("instruction", doc_lower.get("prompt", doc_lower.get("sentence", ""))))))).strip()

            # Try multiple format patterns for choices
            choices = doc_lower.get("choices", doc_lower.get("options", doc_lower.get("answers", [])))

            # Handle "Option A"/"Option B" format (with spaces, any case)
            if not choices and ("option a" in doc_lower or "option_a" in doc_lower):
                choices = [
                    str(doc_lower.get("option a", doc_lower.get("option_a", ""))).strip(),
                    str(doc_lower.get("option b", doc_lower.get("option_b", ""))).strip(),
                    str(doc_lower.get("option c", doc_lower.get("option_c", ""))).strip(),
                    str(doc_lower.get("option d", doc_lower.get("option_d", ""))).strip(),
                ]
                choices = [c for c in choices if c]

            # Handle option1/option2 format (winogrande-style, 1-indexed answer)
            is_winogrande = False
            if not choices and "option1" in doc_lower:
                choices = [
                    str(doc_lower.get("option1", "")).strip(),
                    str(doc_lower.get("option2", "")).strip(),
                ]
                choices = [c for c in choices if c]
                is_winogrande = True

            # Handle binary tasks with target but no explicit choices (e.g., boolq)
            if not choices and "target" in doc_lower:
                target = str(doc_lower.get("target", "")).strip()
                if target in ["نعم", "لا"]:
                    choices = ["لا", "نعم"]
                elif target in ["yes", "no", "true", "false"]:
                    choices = ["no", "yes"]

            # Try multiple format patterns for answer
            # Default to 0 if no answer field (AraDiCE cultural tasks always have answer=0)
            answer = doc_lower.get("answer", doc_lower.get("answerkey", doc_lower.get("label", doc_lower.get("target", doc_lower.get("gold", 0 if choices else None)))))

            # Handle different answer formats
            if isinstance(answer, str):
                if len(answer) == 1 and answer.isalpha():
                    # Answer is like 'A', 'B', 'C', 'D'
                    answer_idx = ord(answer.upper()) - ord('A')
                elif answer.isdigit():
                    # Answer is like '0', '1', '2', '3'
                    answer_idx = int(answer)
                    if is_winogrande and answer_idx >= 1:
                        answer_idx -= 1
                else:
                    # Try to match answer text against choices
                    if choices:
                        try:
                            answer_idx = choices.index(answer)
                        except ValueError:
                            # For binary tasks, default to 1 (yes) if answer looks positive
                            if len(choices) == 2:
                                answer_idx = 1
                            else:
                                return None
                    else:
                        return None
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()
            metadata = {"label": "aradice"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

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
