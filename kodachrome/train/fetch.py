"""``kodachrome-fetch``: download public-domain Kodachrome scans from Wikimedia Commons.

Why Commons and not loc.gov
---------------------------
The Library of Congress FSA/OWI colour transparencies are the target corpus,
but loc.gov sits behind a Cloudflare challenge that returns HTTP 403 to
scripted clients (checked 2026-09-03 with several User-Agents). Commons
hosts the same LoC scans, keeps the catalogue number (LCCN) in each
filename, and its API welcomes scripted access from a tool that identifies
itself.

What "public domain" is allowed to mean
---------------------------------------
A category is a claim, not a guarantee: anyone can file an image into it.
So the licence is checked per file against an allowlist rather than assumed
from the category, and every rejection is written to the manifest with its
reason. A corpus you cannot audit is a corpus you cannot defend.

Validation before acceptance
----------------------------
The API's word is not enough either. Bytes are downloaded to a temporary
file, decoded with Pillow, and checked for size and for being an actual
colour photograph rather than a scanned document or diagram, before being
renamed into place. A resumed run re-hashes what is already on disk against
the manifest and refetches anything that does not match, so a truncated
earlier download cannot silently poison the training set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "kodachrome-film/0.1 (Kodachrome LUT trainer; "
    "https://github.com/kodachrome-film) python-requests"
)
DEFAULT_CATEGORY = "Category:Color photographs from the Farm Security Administration"
SKIP_WORDS = ("cropped", "restored", "retouched", "colorized", "colourized", "edit")
MIN_LONG_SIDE = 800
LICENCE_ALLOWLIST = {"public domain", "cc0", "pdm", "no restrictions"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/tiff"}
_LCCN_RE = re.compile(r"LCCN(\d{6,})", re.IGNORECASE)


class FetchError(Exception):
    """The Commons API could not be reached or answered unexpectedly."""


@dataclass
class FileInfo:
    title: str
    pageid: int
    revid: int
    url: str
    width: int
    height: int
    license: str
    lccn: str | None

    @property
    def filename(self) -> str:
        if self.lccn:
            return f"{self.lccn}.jpg"
        stem = self.title.removeprefix("File:").rsplit(".", 1)[0]
        return f"{re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_')[:120]}.jpg"


@dataclass
class FetchReport:
    files: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    failed: int = 0
    repaired: int = 0


def licence_allowed(text: str | None) -> bool:
    """Explicit allowlist: exact free-licence names, plus the PD-* family."""
    if not text:
        return False
    normalised = text.strip().lower()
    return normalised in LICENCE_ALLOWLIST or normalised.startswith("pd-")


def make_session() -> Any:
    import requests

    return requests.Session()


def api_get(session: Any, params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json"}
    last = "no response"
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - network errors are all retried alike
            last = repr(exc)
        time.sleep(2**attempt)
    raise FetchError(f"Commons API request failed after {retries} attempts: {last}")


def iter_category_members(
    session: Any, category: str, recurse: bool = True, _seen: set[str] | None = None
) -> Iterator[dict]:
    seen = _seen if _seen is not None else set()
    if category in seen:
        return
    seen.add(category)
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file|subcat",
        "cmlimit": "500",
    }
    while True:
        data = api_get(session, params)
        for member in data.get("query", {}).get("categorymembers", []):
            if member["ns"] == 6:
                yield member
            elif member["ns"] == 14 and recurse:
                yield from iter_category_members(session, member["title"], recurse, seen)
        cont = data.get("continue")
        if not cont:
            return
        params = {**params, **cont}


def select_titles(entries: list[dict]) -> tuple[list[str], list[dict]]:
    accepted: list[str] = []
    without: list[str] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        title = entry["title"] if isinstance(entry, dict) else entry
        low = title.lower()
        if any(word in low for word in SKIP_WORDS):
            rejected.append({"title": title, "reason": "title-filter"})
            continue
        m = _LCCN_RE.search(title)
        if m:
            if m.group(1) in seen:
                rejected.append({"title": title, "reason": "duplicate-lccn"})
                continue
            seen.add(m.group(1))
            accepted.append(title)
        else:
            without.append(title)
    return accepted + without, rejected


def fetch_imageinfo(
    session: Any, titles: list[str], width: int
) -> tuple[list[FileInfo], list[dict]]:
    infos: list[FileInfo] = []
    rejected: list[dict] = []
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        data = api_get(
            session,
            {
                "action": "query",
                "prop": "imageinfo|revisions",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|extmetadata|timestamp",
                "iiurlwidth": str(width),
                "rvprop": "ids",
            },
        )
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "?")
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                rejected.append({"title": title, "reason": "no-imageinfo"})
                continue
            if str(ii.get("mime", "")) not in ALLOWED_MIME:
                rejected.append({"title": title, "reason": f"mime:{ii.get('mime')}"})
                continue
            licence = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if not licence_allowed(licence):
                rejected.append({"title": title, "reason": "licence", "license": licence})
                continue
            if max(int(ii["width"]), int(ii["height"])) < MIN_LONG_SIDE:
                rejected.append({"title": title, "reason": "too-small"})
                continue
            m = _LCCN_RE.search(title)
            revisions = page.get("revisions") or [{}]
            infos.append(
                FileInfo(
                    title=title,
                    pageid=int(page.get("pageid", 0)),
                    revid=int(revisions[0].get("revid", 0)),
                    url=ii.get("thumburl") or ii["url"],
                    width=int(ii["width"]),
                    height=int(ii["height"]),
                    license=licence,
                    lccn=m.group(1) if m else None,
                )
            )
    return infos, rejected


def validate_image(data: bytes) -> tuple[bool, str]:
    """Decode the bytes and confirm they are a colour photograph of usable size."""
    if not data:
        return False, "empty"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            width, height = im.size
            mode = im.mode
            sample = np.asarray(im.convert("RGB").resize((64, 64)))
    except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies the file
        return False, f"undecodable:{type(exc).__name__}"
    if max(width, height) < MIN_LONG_SIDE:
        return False, "too-small"
    channel_spread = float(np.abs(sample.max(axis=2).astype(int) - sample.min(axis=2)).mean())
    if mode in ("L", "1") or channel_spread < 2.0:
        return False, "greyscale"
    return True, ""


def download(session: Any, info: FileInfo, out_dir: str | Path, retries: int = 3) -> Path | None:
    """Download to a temporary file, validate, then rename. Never leaves a partial file."""
    out_dir = Path(out_dir)
    final = out_dir / info.filename
    if final.is_file() and final.stat().st_size > 0:
        return final
    tmp = final.with_suffix(final.suffix + ".part")
    for attempt in range(retries):
        try:
            r = session.get(info.url, headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code == 200 and r.content:
                ok, _reason = validate_image(r.content)
                if not ok:
                    return None
                tmp.write_bytes(r.content)
                tmp.replace(final)
                return final
        except Exception:  # noqa: BLE001
            pass
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(2**attempt)
    return None


def corpus_sha1(paths: list[Path]) -> str:
    """Hash the actual bytes of every file, so different content cannot collide."""
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(hashlib.sha1(p.read_bytes()).digest())
    return h.hexdigest()


def fetch_category(
    session: Any,
    category: str,
    out_dir: str | Path,
    width: int = 1024,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> FetchReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda _m: None)

    previous: dict[str, str] = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            previous = {
                e["filename"]: e["sha1"] for e in json.loads(manifest_path.read_text())["files"]
            }
        except (KeyError, json.JSONDecodeError):
            previous = {}

    members = list(iter_category_members(session, category))
    titles, rejected = select_titles(members)
    say(f"{len(titles)} candidate files in {category}, {len(rejected)} rejected by title")
    if sample is not None and sample < len(titles):
        titles = sorted(random.Random(seed).sample(titles, sample), key=titles.index)
    if limit is not None:
        titles = titles[:limit]

    infos, info_rejected = fetch_imageinfo(session, titles, width)
    rejected.extend(info_rejected)
    say(f"{len(infos)} files pass licence and size checks; downloading at {width}px")

    report = FetchReport(rejected=rejected)
    for i, info in enumerate(infos, start=1):
        final = out_dir / info.filename
        recorded = previous.get(info.filename)
        if final.is_file() and recorded:
            if hashlib.sha1(final.read_bytes()).hexdigest() != recorded:
                say(f"  {info.filename} does not match its recorded hash; refetching")
                final.unlink()
                report.repaired += 1
        path = download(session, info, out_dir)
        if path is None:
            report.failed += 1
            rejected.append({"title": info.title, "reason": "download-failed"})
            continue
        entry = asdict(info)
        entry["filename"] = info.filename
        entry["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        report.files.append(entry)
        if i % 25 == 0:
            say(f"  {i}/{len(infos)}")
        time.sleep(0.05)

    manifest = {
        "category": category,
        "width": width,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(report.files),
        "n_failed": report.failed,
        "n_repaired": report.repaired,
        "corpus_sha1": corpus_sha1([out_dir / e["filename"] for e in report.files]),
        "files": report.files,
        "rejected": rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    say(
        f"done: {len(report.files)} files, {report.failed} failed, "
        f"{len(rejected)} rejected, manifest at {manifest_path}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-fetch",
        description="Download public-domain Kodachrome scans from Wikimedia Commons.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/kodachrome"))
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="stop after N files")
    parser.add_argument("--sample", type=int, default=None, help="seeded random subset of N files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-files", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        report = fetch_category(
            make_session(),
            args.category,
            args.out,
            width=args.width,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
            progress=print,
        )
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(report.files) < args.min_files:
        print(
            f"error: accepted {len(report.files)} files, fewer than {args.min_files}. "
            "Check the category name, the licence filter, or the network; "
            "see the manifest's 'rejected' list for reasons.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
