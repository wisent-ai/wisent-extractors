"""Print this package's public surface: the benchmark task names it can extract.

What a caller of this package depends on is not its Python symbols — those are two
registry functions that never change — but **which benchmarks it supports**. Adding a
benchmark is a capability; removing one breaks whoever extracted it yesterday. So the
task names are the public contract, and this prints them for the shared versioning
rule to compare.

Read with `ast`, never by importing. Importing this package pulls in `lm_eval`,
`datasets` and `torch`, and a release decision must not depend on a machine having
them. It also means this runs unchanged against an unpacked sdist, so the surface of
an already published version can be recovered exactly rather than assumed.

The manifests are module-level uppercase dicts whose keys are literal task names:

    LM_EXTRACTORS_A_TO_O: dict[str, str] = {"arithmetic": ..., "asdiv": ...}
    GSM8K_TASKS = {"gsm8k": ...}

Names are prefixed with the family they belong to, because `lm_eval` and `hf` may
offer a task of the same name through different code.

Usage:
    python3 scripts/surface.py [root]     # root defaults to the repository
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

SUFFIXES = ("_TASKS", "_EXTRACTORS")
FAMILIES = ("lm_eval", "hf")


def is_manifest_name(name: str) -> bool:
    """A module-level constant that holds task names."""
    if not name.isupper():
        return False
    return name.endswith(SUFFIXES) or name == "EXTRACTORS"


def family_of(path: pathlib.Path, root: pathlib.Path) -> str:
    """Which extractor family a module belongs to, from its location."""
    parts = path.relative_to(root).parts
    for family in FAMILIES:
        if family in parts:
            return family
    return "other"


def task_names(source: pathlib.Path) -> list:
    """Literal dict keys assigned to manifest constants in one module."""
    try:
        tree = ast.parse(source.read_text(), filename=str(source))
    except OSError as error:
        raise SystemExit(f"{source}: {error}") from error
    except SyntaxError as error:
        # Refuse rather than skip. A module that does not parse cannot be imported
        # either, so its tasks are unreachable at runtime; skipping it would report a
        # smaller surface, and the rule would read that as a removed capability. The
        # surface is unknown here, not shrunk.
        raise SystemExit(
            f"{source}: does not parse, so the surface is unknown: {error}"
        ) from error

    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(is_manifest_name(target.id) for target in targets):
            continue
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.append(key.value)
    return found


def surface(root: pathlib.Path, tolerant: bool = False) -> tuple:
    """The surface, and the modules that had to be skipped to produce it.

    `tolerant` exists for one job: recovering the surface of an artifact that was
    already published with a module that does not parse. Such a module cannot be
    imported by whoever installed it either, so its tasks were never really on offer
    and leaving them out is the truthful reading. Skipped modules are always
    reported, never swallowed.
    """
    package = root / "wisent" / "extractors"
    if not package.is_dir():
        raise SystemExit(f"{package} is not a directory; is {root} the repository root?")
    names = set()
    skipped = []
    for source in sorted(package.rglob("*.py")):
        family = family_of(source, root)
        try:
            tasks = task_names(source)
        except SystemExit:
            if not tolerant:
                raise
            skipped.append(str(source.relative_to(root)))
            continue
        for task in tasks:
            names.add(f"{family}:{task}")
    if not names:
        raise SystemExit(
            f"no task names found under {package}. Either the manifests moved, or "
            "they stopped being literal dicts — both change what this package "
            "promises, so refusing rather than reporting an empty surface"
        )
    return sorted(names), skipped


def main(argv: list) -> int:
    tolerant = "--tolerant" in argv
    positional = [arg for arg in argv if not arg.startswith("-")]
    root = (
        pathlib.Path(positional[int(False)])
        if positional
        else pathlib.Path(__file__).resolve().parent.parent
    )
    names, skipped = surface(root, tolerant)
    document = {"surface": names}
    if skipped:
        document["unparseable"] = skipped
    print(json.dumps(document, indent=int(True) + int(True)))
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
