# wisent-extractors

<!-- wisent-readme-signals:start -->
[![CI](https://github.com/wisent-ai/wisent-extractors/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/wisent-ai/wisent-extractors/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/wisent-ai/wisent-extractors?display_name=tag&sort=semver)](https://github.com/wisent-ai/wisent-extractors/releases)
[![Downloads](https://img.shields.io/github/downloads/wisent-ai/wisent-extractors/total)](https://github.com/wisent-ai/wisent-extractors/releases)
[![License](https://img.shields.io/github/license/wisent-ai/wisent-extractors)](https://github.com/wisent-ai/wisent-extractors)
[![Discord](https://img.shields.io/badge/Discord-Join%20Wisent-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)
<!-- wisent-readme-signals:end -->


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
