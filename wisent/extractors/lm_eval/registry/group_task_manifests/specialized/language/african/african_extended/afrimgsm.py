"""Afrimgsm group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors.specialized.language.asian_african.african.afrimgsm"

AFRIMGSM_TASKS = {
    "afrimgsm": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
    "afrimgsm_direct_amh": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
}
