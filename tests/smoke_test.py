from wisent.extractors.lm_eval.registry.lm_extractor_registry import get_extractor
from wisent.extractors.hf.registry.hf_extractor_registry import get_extractor as hf_get
from wisent.extractors.lm_eval.manifest.lm_extractor_manifest import EXTRACTORS as LM
from wisent.extractors.hf.registry.hf_extractor_manifest import EXTRACTORS as HF
from wisent.extractors.hf.manifest.atoms import HuggingFaceBenchmarkExtractor

print(f"lm_eval manifest: {len(LM)} entries")
print(f"hf manifest: {len(HF)} entries")
ext = get_extractor("gsm8k")
print(f"gsm8k -> {type(ext).__name__}")
docs = HuggingFaceBenchmarkExtractor.load_all_splits("gsm8k", "main")[:3]
print(f"gsm8k load_all_splits smoke: {len(docs)} docs; keys={sorted(docs[0]) if docs else []}")
