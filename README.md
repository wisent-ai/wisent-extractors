<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="wisent-extractors by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/wisent-extractors) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/wisent-extractors/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.ai) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# wisent-extractors

Benchmark extractors split out of the [wisent](https://github.com/wisent-ai/wisent)
monorepo. Contains:

- `wisent.extractors.lm_eval` — 676 extractors for lm-eval-harness tasks
- `wisent.extractors.hf` — 223 extractors for wisent-proprietary HuggingFace benchmarks

## Install

```
pip install wisent-extractors
```

## Usage

```python
from wisent.extractors.lm_eval.registry.lm_extractor_registry import get_extractor

extractor = get_extractor("gsm8k")
pairs = extractor.extract_contrastive_pairs(limit=100)
```

## Namespace packaging

This package is a namespace package that shares the `wisent.*` import root
with `wisent-core` and `wisent-evaluators`. All three can be installed
side-by-side without conflict.