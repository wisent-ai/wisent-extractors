"""Parts of pile.py, split by the tama size splitter; pile.py imports every name back."""

from __future__ import annotations
from ..pile import _LOG


_PILE_CDN_CACHE: dict[str, list[dict]] = {}


def _hf_pile_headers() -> dict:
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_monology_pile_cdn(filename: str = "test.jsonl.zst") -> list[dict]:
    """Download and parse a JSON-L file from monology/pile-uncopyrighted via CDN.

    The CDN endpoint (resolve/main/...) bypasses the rate-limited /api/datasets
    endpoint. The file is zstandard-compressed; each line is JSON with shape:
        {"text": "...", "meta": {"pile_set_name": "Github"}}

    Cached in module-level dict for the process lifetime.
    """
    if filename in _PILE_CDN_CACHE:
        return _PILE_CDN_CACHE[filename]

    import io
    import json as _json
    import requests
    import zstandard as zstd

    url = f"https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/{filename}"
    log = _LOG
    try:
        resp = requests.get(url, headers=_hf_pile_headers(), timeout=600, stream=False)
        if resp.status_code != 200:
            log.warning(f"CDN fetch {url} -> HTTP {resp.status_code}")
            return []
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(resp.content, max_output_size=10 * 1024 * 1024 * 1024)
        rows: list[dict] = []
        for line in io.BytesIO(raw):
            try:
                rows.append(_json.loads(line))
            except Exception:
                continue
        log.info(f"Loaded {len(rows)} pile rows from CDN {filename}")
        _PILE_CDN_CACHE[filename] = rows
        return rows
    except Exception as exc:
        log.warning(f"CDN pile fetch failed: {str(exc)[:200]}")
        return []


task_names = (
    "pile",
    "pile_arxiv", "pile_bookcorpus2", "pile_books3", "pile_dm-mathematics", "pile_enron",
    "pile_europarl", "pile_freelaw", "pile_github", "pile_gutenberg", "pile_hackernews",
    "pile_nih-exporter", "pile_opensubtitles", "pile_openwebtext2", "pile_philpapers",
    "pile_pile-cc", "pile_pubmed-abstracts", "pile_pubmed-central", "pile_stackexchange",
    "pile_ubuntu-irc", "pile_uspto", "pile_wikipedia", "pile_youtubesubtitles"
)
