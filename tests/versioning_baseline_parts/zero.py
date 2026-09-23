"""Parts of versioning_baseline.py, split by the tama size splitter; versioning_baseline.py imports every name back."""

from __future__ import annotations
import io
import json
import pathlib
import re
import sys
import tarfile
import urllib.error
import urllib.request
import versioning_surface as extractor


ZERO = int(False)
ONE = int(True)


MARKER = "pypi-sdist"
LOWER_TIERS = ("pypi-wheel", "git-archive", "head")
INDEX = "https://pypi.org/pypi"
USER_AGENT = "wisent-extractors-versioning"
# A project PyPI certainly serves, asked through the exact same spelling and the same
# (absent) credential as the subject. Anything narrower is a second subject rather
# than a control.
CONTROL = "pip"
NOT_FOUND = int("404")
STATED_ABSENT = "not found"

NAMED = "named"
ABSENT = "absent"
UNPROVEN = "unproven"


SEPARATORS = re.compile(r"[-_.]+")
MODES = ("--write", "--declared-version")


def normalized(name: str) -> str:
    """A distribution name in the one spelling PyPI compares by (PEP 503)."""
    return SEPARATORS.sub("-", name).lower()


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
