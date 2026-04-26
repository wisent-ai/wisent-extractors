from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SuperGlueT5PromptExtractor"]
_LOG = setup_logger(__name__)

task_names = ("super-glue-t5-prompt",)
class SuperGlueT5PromptExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Super Glue T5 Prompt benchmark."""


    evaluator_name = "generation"
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
            # RTE / CB format — premise + hypothesis + label (entailment)
            if "premise" in doc and "hypothesis" in doc and "label" in doc and "choice1" not in doc:
                premise = str(doc.get("premise", "")).strip()
                hypothesis = str(doc.get("hypothesis", "")).strip()
                label = doc.get("label")
                # CB: 0=entailment 1=contradiction 2=neutral; RTE: 0=entailment 1=not_entailment
                label_map_rte = {0: "True", 1: "False"}
                label_map_cb = {0: "True", 1: "False", 2: "Neither"}
                if premise and hypothesis and isinstance(label, int):
                    if label in label_map_cb:
                        correct = label_map_cb[label] if label in (0, 1, 2) else None
                        incorrect = "False" if correct == "True" else "True"
                        return self._build_pair(
                            question=f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis?",
                            correct=correct,
                            incorrect=incorrect,
                            metadata={"label": "super_glue_t5_prompt"},
                        )

            # WiC format — sentence1 + sentence2 + word + label
            if "sentence1" in doc and "sentence2" in doc and "word" in doc and "label" in doc:
                s1 = str(doc.get("sentence1", "")).strip()
                s2 = str(doc.get("sentence2", "")).strip()
                w = str(doc.get("word", "")).strip()
                label = doc.get("label")
                if s1 and s2 and w and isinstance(label, int):
                    correct = "True" if label == 1 else "False"
                    incorrect = "False" if label == 1 else "True"
                    return self._build_pair(
                        question=f"Sentence 1: {s1}\nSentence 2: {s2}\nDoes the word \"{w}\" mean the same thing in both?",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "super_glue_t5_prompt"},
                    )

            # MultiRC format — paragraph + question + answer + label
            if "paragraph" in doc and "question" in doc and "answer" in doc and "label" in doc:
                p = str(doc.get("paragraph", "")).strip()
                q = str(doc.get("question", "")).strip()
                a = str(doc.get("answer", "")).strip()
                label = doc.get("label")
                if p and q and a and isinstance(label, int):
                    correct = "True" if label == 1 else "False"
                    incorrect = "False" if label == 1 else "True"
                    return self._build_pair(
                        question=f"Paragraph: {p}\nQuestion: {q}\nAnswer: {a}\nIs the answer correct?",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "super_glue_t5_prompt"},
                    )

            # ReCoRD format — passage + query + entities + answers
            if "passage" in doc and "query" in doc and ("entities" in doc or "answers" in doc):
                passage = str(doc.get("passage", "")).strip()
                query = str(doc.get("query", "")).strip()
                answers = doc.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers] if answers else []
                entities = doc.get("entities") or []
                if isinstance(entities, str):
                    entities = [entities]
                if isinstance(answers, list) and answers and passage and query:
                    correct = str(answers[0]).strip()
                    incorrect = ""
                    for e in entities:
                        e_str = str(e).strip()
                        if e_str and e_str != correct:
                            incorrect = e_str
                            break
                    if correct and incorrect:
                        return self._build_pair(
                            question=f"{passage}\n\n{query}",
                            correct=correct,
                            incorrect=incorrect,
                            metadata={"label": "super_glue_t5_prompt"},
                        )

            # COPA format — premise + choice1 + choice2 + question + label
            if "premise" in doc and "choice1" in doc and "choice2" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                choice1 = str(doc.get("choice1", "")).strip()
                choice2 = str(doc.get("choice2", "")).strip()
                q_type = str(doc.get("question", "effect")).strip()
                label = doc.get("label", -1)
                if premise and choice1 and choice2 and label in (0, 1):
                    correct = choice1 if label == 0 else choice2
                    incorrect = choice2 if label == 0 else choice1
                    connector = "because" if q_type == "cause" else "so"
                    return self._build_pair(
                        question=f"{premise} {connector}",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "super_glue_t5_prompt"},
                    )

            # Format 0: WSC format — text + span1_text + span2_text + label
            if "text" in doc and "span1_text" in doc and "span2_text" in doc and "label" in doc:
                text = str(doc.get("text", "")).strip()
                span1 = str(doc.get("span1_text", "")).strip()
                span2 = str(doc.get("span2_text", "")).strip()
                label = doc.get("label", -1)
                if text and span1 and span2 and label in (0, 1):
                    correct = "True" if label == 1 else "False"
                    incorrect = "False" if label == 1 else "True"
                    return self._build_pair(
                        question=f"Sentence: {text}\nDoes \"{span2}\" refer to \"{span1}\"?\nAnswer:",
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "super_glue_t5_prompt"},
                    )

            # Format 1: BoolQ format (question + passage + label)
            if "question" in doc and "passage" in doc and "label" in doc:
                question = str(doc.get("question", "")).strip()
                passage = str(doc.get("passage", "")).strip()
                label = doc.get("label")

                if question and label is not None:
                    # label 1 = True, 0 = False
                    if label == 1:
                        correct = "True"
                        incorrect = "False"
                    else:
                        correct = "False"
                        incorrect = "True"

                    formatted_question = f"Passage: {passage}\n\nQuestion: {question}\nAnswer (True/False):"
                    metadata = {"label": "super_glue_t5_prompt"}
                    return self._build_pair(
                        question=formatted_question,
                        correct=correct,
                        incorrect=incorrect,
                        metadata=metadata,
                    )

            # Format 2: Multiple choice format
            # Try multiple format patterns for question
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", ""))))).strip()

            # Try multiple format patterns for choices
            choices = doc.get("choices", doc.get("options", doc.get("answers", [])))

            # Handle option_a/b/c/d format
            if not choices and "option_a" in doc:
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]

            # Try multiple format patterns for answer
            answer = doc.get("answer", doc.get("label", doc.get("target", None)))

            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
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

            formatted_question = f"Question: {question}\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "super_glue_t5_prompt"}

            return self._build_pair(
                question=formatted_question,
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
        from wisent.core.primitives.contrastive_pairs.core.io.response import (
            NegativeResponse,
            PositiveResponse,
        )
        return ContrastivePair(
            prompt=question,
            positive_response=PositiveResponse(model_response=correct),
            negative_response=NegativeResponse(model_response=incorrect),
            label=(metadata or {}).get("label"),
        )
