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
    python3 scripts/baseline.py                     # print the recovered baseline
    python3 scripts/baseline.py --write             # write released-surface.json
    python3 scripts/baseline.py --declared-version  # print setup.py's version
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

ZERO = int(False)
ONE = int(True)

sys.path.insert(ZERO, str(pathlib.Path(__file__).resolve().parent))

import surface as extractor  # noqa: E402  (path set above so this runs from anywhere)

MARKER = "pypi-sdist"
LOWER_TIERS = ("pypi-wheel", "git-archive", "head")
INDEX = "https://pypi.org/pypi"
USER_AGENT = "wisent-extractors-baseline (scripts/baseline.py)"
# A project PyPI certainly serves, asked through the exact same spelling and the same
# (absent) credential as the subject. Anything narrower is a second subject rather
# than a control.
CONTROL = "pip"
NOT_FOUND = int("404")
STATED_ABSENT = "not found"

NAMED = "named"
ABSENT = "absent"
UNPROVEN = "unproven"

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPOSITORY / "setup.py"
BASELINE = REPOSITORY / "released-surface.json"
SEPARATORS = re.compile(r"[-_.]+")
MODES = ("--write", "--declared-version")


def normalized(name: str) -> str:
    """A distribution name in the one spelling PyPI compares by (PEP 503)."""
    return SEPARATORS.sub("-", name).lower()


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


def ask_index(project: str) -> tuple:
    """What PyPI says about a project: named with its document, absent, or unproven.

    Three states, never two. `urlopen` raising and the index stating that it has no
    such project are different facts that a client's error status collapses into one,
    and the wrong reading is the passing one: a two-state probe would report this
    package absent on every DNS hiccup and drop the baseline to a tier below the one
    that exists.

    So absence is read from the answer's CONTENT -- the index must both answer 404 and
    say so -- and presence is only believed when the document names the subject back.
    An error page, a rate-limit page or a redirect to something else fails all three
    tests and comes back `unproven`, which is not the same as fine.
    """
    request = urllib.request.Request(
        f"{INDEX}/{project}/json", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").lower()
        if error.code == NOT_FOUND and STATED_ABSENT in body:
            return ABSENT, None
        return UNPROVEN, None
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return UNPROVEN, None
    served = document.get("info", {}).get("name", "")
    if normalized(served) != normalized(project):
        return UNPROVEN, None
    return NAMED, document


def control() -> None:
    """Refuse unless this probe can still recognise a project PyPI definitely serves.

    Content-reading already covers transport silence, which makes this look
    redundant. It is not: `unproven` is also what a broken expression produces, so a
    probe that can no longer recognise ANY published project would refuse forever
    while the index answers perfectly, and the operator would blame PyPI. This says
    which side is broken.
    """
    state, _ = ask_index(CONTROL)
    if state != NAMED:
        raise SystemExit(
            f"this probe cannot recognise {CONTROL}, which PyPI definitely serves "
            f"(it read '{state}'), so its answer about any other project is meaningless. "
            "Fix the probe; do not read its verdict as absence"
        )


def latest_published(project: str) -> tuple:
    """The newest version PyPI serves for a project, and that version's files.

    Asked of the project rather than of any particular version, so a bump that has
    not been released yet cannot be mistaken for the project never having been
    released.
    """
    state, document = ask_index(project)
    control()
    if state == UNPROVEN:
        raise SystemExit(
            f"the index did not answer about {project}, so whether it is published is "
            "unproven. A baseline is not regenerated from an unanswered question"
        )
    if state == ABSENT:
        raise SystemExit(
            f"PyPI states it serves no {project}. The tiers below {MARKER} "
            f"({', '.join(LOWER_TIERS)}) are not implemented here because this package has "
            "always been published; refusing rather than inventing a baseline"
        )
    version = document["info"]["version"]
    return version, document["releases"].get(version, document.get("urls", []))


def unpack_sdist(files: list, version: str, into: pathlib.Path) -> tuple:
    """Download the sdist for a version; return its filename and unpacked root."""
    sdists = [entry for entry in files if entry.get("packagetype") == "sdist"]
    if not sdists:
        raise SystemExit(
            f"the published {version} has no sdist, only "
            f"{sorted({entry.get('packagetype') for entry in files})}. pypi-wheel is a real "
            "tier and this generator does not implement it, so it refuses rather than "
            "reporting a baseline stamped with a tier it did not read"
        )
    entry = sdists[ZERO]
    with urllib.request.urlopen(
        urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    ) as response:
        blob = response.read()
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive.getmembers():
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(
                    f"{entry['filename']} holds a member outside the archive root "
                    f"({member.name}); refusing to unpack it"
                )
        archive.extractall(into)
    roots = [child for child in into.iterdir() if child.is_dir()]
    if len(roots) != ONE:
        raise SystemExit(
            f"{entry['filename']}: expected one top-level directory, got {roots}"
        )
    return entry["filename"], roots[ZERO]


def read(root: pathlib.Path) -> tuple:
    """The surface of an unpacked artifact.

    Static, through the same extractor the gate runs against the working tree: one
    reader, so a disagreement between the two sides of the comparison cannot be an
    artefact of reading them differently.

    Tolerant only here, and never for the candidate: a module that does not parse in
    something already published could not be imported by whoever installed it either,
    so its tasks were never really on offer, and leaving them out is the truthful
    reading of what that release could extract. What it must never do is pass
    unmentioned, so every skipped module is reported on stderr AND carried in the
    baseline. The published 0.1.62 is exactly this case.
    """
    try:
        return extractor.surface(root)
    except SystemExit as error:
        names, skipped = extractor.surface(root, tolerant=True)
        print(f"note: {error}", file=sys.stderr)
        return names, skipped


def baseline() -> dict:
    project = setup_keyword("name")
    version, files = latest_published(project)
    with tempfile.TemporaryDirectory() as scratch:
        filename, root = unpack_sdist(files, version, pathlib.Path(scratch))
        names, skipped = read(root)
    document = {
        "version": version,
        "source": f"{MARKER}:{filename} unpacked and read by scripts/surface.py",
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
