from wisent.extractors.lm_eval.registry.lm_extractor_registry import get_extractor
from wisent.extractors.hf.registry.hf_extractor_registry import get_extractor as hf_get
from wisent.extractors.lm_eval.manifest.lm_extractor_manifest import EXTRACTORS as LM
from wisent.extractors.hf.registry.hf_extractor_manifest import EXTRACTORS as HF
print(f"lm_eval manifest: {len(LM)} entries")
print(f"hf manifest: {len(HF)} entries")
ext = get_extractor("gsm8k")
print(f"gsm8k -> {type(ext).__name__}")
