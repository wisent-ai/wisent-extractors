"""Resolve every registered extractor reference and report what each key reaches.

`get_extractor(name)` turns a registry entry "module.path:Class" into a class by
importing the module and walking the attribute path, then constructs it. This
does the same resolution for every key of both registries (lm_eval and hf),
without constructing anything, and prints one line per key:

    lm_eval:gsm8k  GSM8KExtractor < LMEvalBenchmarkExtractor < ABC < object
    hf:aime        error  ImportError: Cannot import module ...

A class is named with its bases in method resolution order, which do not
change when its definition moves to another file or folder (a split, a
regrouping), while a reference that now reaches another class, or nothing at
all, reads differently. With `--compare FILE` (the output of an
earlier run) it exits 1 and names every key that resolved in that run and
resolves differently now; keys that did not resolve before are reported but do
not fail, because they were already broken. With `--known FILE` it exits 1
when a key does not resolve and FILE does not list it (one key per line, two
spaces, the reason), or when a key FILE lists resolves now and should leave
the list; CI runs this form on every push.

Usage (in an environment where `pip install -e .` has run):
    python tests/registry/resolution.py > before.txt
    python tests/registry/resolution.py --compare before.txt
    python tests/registry/resolution.py --known tests/registry/known-unresolved.txt
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path


def _registries() -> dict[str, dict]:
    from wisent.extractors.hf.registry import hf_extractor_registry
    from wisent.extractors.lm_eval.registry import lm_extractor_registry

    return {"lm_eval": lm_extractor_registry._REGISTRY, "hf": hf_extractor_registry._REGISTRY}


def _resolve(ref) -> str:
    if not isinstance(ref, str):
        target = ref
    else:
        module_path, attr_path = ref.split(":", 1)
        try:
            target = importlib.import_module(module_path)
            for part in attr_path.split("."):
                target = getattr(target, part)
        except Exception as error:  # the report names every failure; one must not hide the rest
            message = str(error).splitlines()[0] if str(error) else ""
            return f"error  {type(error).__name__}: {message}"
    if not inspect.isclass(target):
        return f"{target!r}  not a class"
    return " < ".join(klass.__qualname__ for klass in target.__mro__)


def report() -> dict[str, str]:
    lines = {}
    for family, registry in _registries().items():
        for key in sorted(registry):
            lines[f"{family}:{key}"] = _resolve(registry[key])
    return lines


def _read(path: Path) -> dict[str, str]:
    earlier = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition("  ")
        if key:
            earlier[key] = rest
    return earlier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", type=Path, help="output of an earlier run to compare against")
    parser.add_argument("--known", type=Path, help="keys allowed not to resolve, each with its reason")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    now = report()
    if args.known:
        known = _read(args.known)
        failing = [key for key, outcome in now.items() if outcome.startswith("error") and key not in known]
        healed = [key for key in known if key in now and not now[key].startswith("error")]
        for key in failing:
            print(f"UNRESOLVED {key}: {now[key]}")
        for key in healed:
            print(f"RESOLVES NOW {key}: {now[key]}; remove it from {args.known}")
        print(f"{len(now)} keys; {len(failing)} unresolved and not known; {len(healed)} known but resolving")
        return 1 if failing or healed else 0
    if not args.compare:
        for key, outcome in now.items():
            print(f"{key}  {outcome}")
        return 0
    earlier = _read(args.compare)
    changed = [key for key, outcome in earlier.items()
               if not outcome.startswith("error") and now.get(key) != outcome]
    still_broken = [key for key, outcome in earlier.items() if outcome.startswith("error")]
    for key in changed:
        print(f"CHANGED {key}: was {earlier[key]!r}, now {now.get(key, 'missing')!r}")
    resolved = sum(1 for outcome in now.values() if not outcome.startswith("error"))
    print(f"{len(now)} keys, {resolved} resolve; {len(changed)} changed since the earlier run; "
          f"{len(still_broken)} did not resolve then either")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
