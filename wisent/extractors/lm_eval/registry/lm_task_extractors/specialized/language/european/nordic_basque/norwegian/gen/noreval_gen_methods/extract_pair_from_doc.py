"""Methods of NorevalGenerationExtractor in noreval_gen.py, split by the tama size splitter; the class imports each one back."""

from __future__ import annotations
from typing import Any
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.cli.cli_logger import bind
from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_LONG
from ..noreval_gen import _LOG


def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
    """
        Convert a single Noreval generation doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
    log = bind(_LOG, doc_id=doc.get("id", "unknown"))

    try:
        # NorRewrite/NorSummarize: prompt + context + response
        if "prompt" in doc and "context" in doc and "response" in doc:
            prompt = str(doc.get("prompt", "")).strip()
            context = str(doc.get("context", "")).strip()
            response = str(doc.get("response", "")).strip()
            if prompt and response:
                full_prompt = f"{prompt}\n{context}".strip()
                words = response.split()
                incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil svar"
                return self._build_pair(
                    question=full_prompt,
                    correct=response,
                    incorrect=incorrect,
                    metadata={"label": "noreval_instruct"},
                )

        # NorSumm: article + summaries (list)
        if "article" in doc and "summaries" in doc:
            article = str(doc.get("article", "")).strip()
            summaries = doc.get("summaries", [])
            if article and summaries:
                if isinstance(summaries, list) and summaries:
                    summary = str(summaries[0]).strip() if not isinstance(summaries[0], dict) else str(summaries[0].get("text", summaries[0].get("summary", ""))).strip()
                else:
                    summary = str(summaries).strip()
                if summary:
                    words = summary.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil samandrag"
                    return self._build_pair(
                        question=f"Lag eit samandrag av: {article[:1500]}",
                        correct=summary,
                        incorrect=incorrect,
                        metadata={"label": "noreval_summ"},
                    )

        # Tatoeba: sourceString + targetString
        if "sourceString" in doc and "targetString" in doc:
            src = str(doc.get("sourceString", "")).strip()
            tgt = str(doc.get("targetString", "")).strip()
            if src and tgt:
                words = tgt.split()
                incorrect = " ".join(reversed(words)) if len(words) > 1 else "feil"
                return self._build_pair(
                    question=f"Omsett: {src}",
                    correct=tgt,
                    incorrect=incorrect,
                    metadata={"label": "noreval_tatoeba"},
                )

        # Format 1: ask_gec - {source, correction}
        if "source" in doc and "correction" in doc:
            source = str(doc["source"]).strip()
            correction = str(doc["correction"]).strip()

            if not source or not correction:
                log.debug("Skipping doc due to empty source/correction", extra={"doc": doc})
                return None

            question = f"Correct the following Norwegian sentence:\n{source}"
            correct = correction
            incorrect = source  # Use the uncorrected version as negative

            metadata = {"label": "noreval_gec"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        # Format 2: noridiom - {idiom_start, accepted_completions}
        elif "idiom_start" in doc and "accepted_completions" in doc:
            idiom_start = str(doc["idiom_start"]).strip()
            accepted_completions = doc["accepted_completions"]

            if not idiom_start or not accepted_completions:
                log.debug("Skipping doc due to empty idiom fields", extra={"doc": doc})
                return None

            correct = str(accepted_completions[0]).strip()

            # Create synthetic negative by using a different completion
            incorrect = "ukjent"  # "unknown" in Norwegian

            question = f"Complete the Norwegian idiom: {idiom_start}"

            metadata = {"label": "noreval_idiom"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        # Format 3: norquad - {context, question, answers}
        elif "context" in doc and "question" in doc and "answers" in doc:
            context = str(doc.get("context", "")).strip()
            question = str(doc["question"]).strip()
            answers = doc["answers"]

            if not question or not answers:
                log.debug("Skipping doc due to empty question/answers", extra={"doc": doc})
                return None

            # Extract answer text
            if isinstance(answers, dict) and "text" in answers:
                answer_texts = answers["text"]
                if answer_texts:
                    correct = str(answer_texts[0]).strip()
                else:
                    return None
            else:
                return None

            # Create synthetic negative by shuffling words in the correct answer
            incorrect = self._create_shuffled_text(correct)

            formatted_question = f"Context: {context[:DISPLAY_TRUNCATION_MEDIUM]}...\n\nQuestion: {question}" if len(context) > DISPLAY_TRUNCATION_MEDIUM else f"Context: {context}\n\nQuestion: {question}"

            metadata = {"label": "noreval_qa"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        # Format 4: norrewrite_instruct/norsummarize_instruct - {prompt, context, target}
        elif "prompt" in doc and "context" in doc and "target" in doc:
            prompt = str(doc["prompt"]).strip()
            context = str(doc["context"]).strip()
            target = str(doc["target"]).strip()

            if not prompt or not context or not target:
                log.debug("Skipping doc due to empty prompt/context/target", extra={"doc": doc})
                return None

            question = f"{prompt}\n\n{context[:DISPLAY_TRUNCATION_LONG]}..." if len(context) > DISPLAY_TRUNCATION_LONG else f"{prompt}\n\n{context}"
            correct = target

            # Create synthetic negative by shuffling sentences in the target
            incorrect = self._create_shuffled_text(target)

            metadata = {"label": "noreval_rewrite"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        # Format 5: nortruthfulqa_gen - {question, correct_answers, incorrect_answers}
        elif "question" in doc and "correct_answers" in doc:
            question = str(doc["question"]).strip()
            correct_answers = doc.get("correct_answers", [])
            incorrect_answers = doc.get("incorrect_answers", [])

            if not question or not correct_answers:
                log.debug("Skipping doc due to empty question/correct_answers", extra={"doc": doc})
                return None

            correct = str(correct_answers[0]).strip()

            # Use provided incorrect answer if available, otherwise create synthetic
            if incorrect_answers:
                incorrect = str(incorrect_answers[0]).strip()
            else:
                incorrect = self._create_shuffled_text(correct)

            formatted_question = f"Question: {question}"

            metadata = {"label": "noreval_truthfulqa_gen"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        else:
            log.debug("Skipping doc due to unrecognized format", extra={"doc": doc})
            return None

    except Exception as exc:
        log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
        return None
