"""Regenerate released-surface.json from the artifact callers can actually install.

The baseline is the one input to the version gate that cannot be derived from the
working tree: it is the set of benchmark tasks the version already in somebody's
site-packages can extract. It must therefore be recovered, never typed.

The recovered file stamps its "source" field with the fleet marker grammar, so every
repository's gate reads it the same way:

    source = "<marker> <free prose tail>"

with the marker the first whitespace-delimited token, one of

    pypi-sdist:<filename>       recovered from a published sdist
    pypi-wheel:<filename>       recovered from a published pure-Python wheel
    npm-tarball:<registry path> recovered from a published npm tarball
    crates-io:<filename>        recovered from a published crate
    stado:<object path>         recovered from an artifact in the release channel
    gh-release:<tag>            recovered from an asset on a GitHub Release
    git-archive:<tag>           reproduced from a git tag
    head:<full sha>             last resort: nothing published, no usable tag

in that order of preference. This distribution is `wisent-extractors` on PyPI --
setup.py names it, PyPI serves it, and the served version carries an sdist -- so
`pypi-sdist` is the best tier that actually exists and MARKER below is the only tier
this generator produces. It refuses loudly rather than quietly dropping to
pypi-wheel, git-archive or head, because a baseline recovered from a worse artifact
than the one that exists measures every later release against the wrong thing. The
gate in .github/workflows/version-check.yml understands the whole grammar even so --
its job is to catch a baseline that was hand-edited into claiming a tier.

npm's scope trap has no counterpart here: PyPI names carry no scope, so no path is
assembled and none can be. The one PyPI-shaped hazard is the opposite one, and it is
handled: the index answers *generically* when it has never heard of a project, so a
lookup of an empty or misspelled name reads as proven absence. Hence the name is
asserted non-empty before it is asked about, the answer must name the subject back,
and a positive control runs through the same function, the same URL shape and the
same anonymous access as the subject.

THE TRAP: the baseline is the LATEST PUBLISHED version, never the declared one. The
moment somebody bumps setup.py ahead of a release, looking up the declared version
404s, and a generator that read that as "nothing is published" would throw away the
real baseline and compare everything against HEAD.

THE SECOND TRAP: absence is a three-state answer. An index that says "no such
project" and an index that could not be reached are different facts, and the wrong
reading is always the passing one, so this never infers absence from a client error.

Usage:
    python3 tests/versioning_baseline.py                     # print the recovered baseline
    python3 tests/versioning_baseline.py --write             # write released-surface.json
    python3 tests/versioning_baseline.py --declared-version  # print setup.py's version
"""

from __future__ import annotations

import ast
import io
import json
import pathlib
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request


import sys as _size_split_sys
from pathlib import Path as _SizeSplitPath
_size_split_bytecode = _size_split_sys.dont_write_bytecode
_size_split_sys.dont_write_bytecode = True
if str(_SizeSplitPath(__file__).resolve().parent) not in _size_split_sys.path:
    _size_split_sys.path.insert(0, str(_SizeSplitPath(__file__).resolve().parent))
from versioning_baseline_parts.zero import ZERO, ONE, MARKER, LOWER_TIERS, INDEX, USER_AGENT, CONTROL, NOT_FOUND, STATED_ABSENT, NAMED, ABSENT, UNPROVEN, SEPARATORS, MODES, normalized, ask_index, control, latest_published, unpack_sdist, read  # noqa: F401
_size_split_sys.dont_write_bytecode = _size_split_bytecode
del _size_split_sys, _SizeSplitPath, _size_split_bytecode


sys.path.insert(ZERO, str(pathlib.Path(__file__).resolve().parent))

import versioning_surface as extractor  # noqa: E402  (path set above so this runs from anywhere)


REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPOSITORY / "setup.py"
BASELINE = REPOSITORY / "released-surface.json"


def setup_keyword(name: str) -> str:
    """One literal keyword of the setup() call in setup.py.

    Read with `ast`, never by executing setup.py: running it imports setuptools and
    walks the tree through find_namespace_packages, and neither the version nor the
    distribution name should depend on a machine that can do either.

    A missing, non-literal or empty value is a refusal rather than an empty string.
    PyPI answers generically about a project it does not know, so an empty name would
    be asked about and reported absent -- proven absence of nothing at all.
    """
    try:
        tree = ast.parse(MANIFEST.read_text(encoding="utf-8"), filename=str(MANIFEST))
    except OSError as error:
        raise SystemExit(f"{MANIFEST}: {error}") from error
    except SyntaxError as error:
        raise SystemExit(f"{MANIFEST}: does not parse, so nothing here is known: {error}") from error

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        if called != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != name or not isinstance(keyword.value, ast.Constant):
                continue
            value = str(keyword.value.value).strip()
            if value:
                return value
    raise SystemExit(
        f"{MANIFEST} declares no literal setup({name}=...). Either it moved or it turned "
        f"dynamic; both mean this generator does not know the {name} it would otherwise "
        "ask the index about, and asking about an empty one reads as proven absence"
    )


def baseline() -> dict:
    project = setup_keyword("name")
    version, files = latest_published(project)
    with tempfile.TemporaryDirectory() as scratch:
        filename, root = unpack_sdist(files, version, pathlib.Path(scratch))
        names, skipped = read(root)
    document = {
        "version": version,
        "source": f"{MARKER}:{filename} unpacked and read by tests/versioning_surface.py",
        "surface": names,
    }
    if skipped:
        document["unparseable"] = skipped
    return document


def main(argv: list) -> int:
    unknown = [argument for argument in argv if argument not in MODES]
    if unknown:
        raise SystemExit(
            f"unknown argument(s) {' '.join(unknown)}. This takes no arguments to print the "
            f"recovered baseline, or one of {', '.join(MODES)}. Refusing rather than printing "
            "when a misspelt --write asked for a file to be written"
        )
    if "--declared-version" in argv:
        print(setup_keyword("version"))
        return ZERO

    document = baseline()
    text = json.dumps(document, indent=ONE + ONE) + "\n"
    if "--write" in argv:
        BASELINE.write_text(text, encoding="utf-8")
        marker = document["source"].split(" ")[ZERO]
        print(
            f"wrote {BASELINE.name}: {document['version']}, "
            f"{len(document['surface'])} names, {marker}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)
    return ZERO


if __name__ == "__main__":
    sys.exit(main(sys.argv[ONE:]))
