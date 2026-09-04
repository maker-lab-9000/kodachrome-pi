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
from collections import Counter
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
# Which licence families a fetch may accept. The default is public domain
# only. "cc-by" and "cc-by-sa" exist because the Kodachrome slides people
# actually remember -- K-14 era, 1970s to 2000s -- are almost all CC BY or
# CC BY-SA on Commons (of 852 candidates, 107 were PD). A fitted LUT carries
# no image content, so training on them is defensible, but it is a choice:
# it must be asked for, it is written into the manifest and the artifact's
# provenance, and every accepted file's author and licence are recorded so
# attribution is possible. NC and ND variants are never accepted.
LICENCE_POLICIES = ("pd", "cc-by", "cc-by-sa")
DEFAULT_LICENCES = frozenset({"pd"})
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
    artist: str = ""

    @property
    def filename(self) -> str:
        """LCCN when there is one; otherwise the title stem plus the Commons page id.

        The page id is what makes the name unique. Without it, "Ahaggar
        Mountains 1981 01" and "Ahaggar Mountains 1981-01" normalise to the
        same stem, the second download overwrites the first, and the manifest
        lists both: 79 of 835 K-14 files were lost that way before anyone
        counted the JPEGs on disk against the manifest.
        """
        if self.lccn:
            return f"{self.lccn}.jpg"
        stem = self.title.removeprefix("File:").rsplit(".", 1)[0]
        return f"{re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_')[:100]}_{self.pageid}.jpg"


@dataclass
class FetchReport:
    files: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    failed: int = 0
    repaired: int = 0


def parse_licences(text: str) -> frozenset[str]:
    """``"pd,cc-by"`` -> ``{"pd", "cc-by"}``, refusing anything not in LICENCE_POLICIES."""
    names = frozenset(n.strip().lower() for n in text.split(",") if n.strip())
    unknown = names - set(LICENCE_POLICIES)
    if unknown or not names:
        raise ValueError(
            f"unknown licence policy {sorted(unknown)}; choose from {', '.join(LICENCE_POLICIES)}"
        )
    return names


def licence_allowed(text: str | None, licences: frozenset[str] = DEFAULT_LICENCES) -> bool:
    """Explicit allowlist per policy. NC and ND are refused under every policy."""
    if not text:
        return False
    n = text.strip().lower()
    if "pd" in licences and (n in LICENCE_ALLOWLIST or n.startswith("pd-")):
        return True
    if "-nc" in n or "-nd" in n:
        return False
    if "cc-by-sa" in licences and n.startswith("cc by-sa"):
        return True
    if "cc-by" in licences and (n.startswith("cc by ") or n == "attribution"):
        return True
    return False


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
                # A file in both a category and one of its subcategories is
                # reached twice. Before this check it was listed twice, hashed
                # twice into corpus_sha1, and counted twice: 79 of "835" K-14
                # files and 7 of "1,140" LoC files were the same file again.
                if member["title"] in seen:
                    continue
                seen.add(member["title"])
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
    session: Any, titles: list[str], width: int, licences: frozenset[str] = DEFAULT_LICENCES
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
            if not licence_allowed(licence, licences):
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
                    artist=re.sub(
                        r"<[^>]+>", "", ii.get("extmetadata", {}).get("Artist", {}).get("value", "")
                    ).strip()[:200],
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


def download(
    session: Any, info: FileInfo, out_dir: str | Path, retries: int = 3
) -> tuple[Path | None, str]:
    """Download to a temporary file, validate, then rename. Never leaves a partial file.

    Returns ``(path, "")`` on success and ``(None, reason)`` on failure. The
    reason is carried rather than discarded because it lands in the manifest,
    and "a corpus you cannot audit is a corpus you cannot defend" is this
    module's whole premise. Collapsing every failure into one generic label
    would hide the byte-level colour and size check — the very check that
    catches a scanned document the API described as a photograph.

    Content that decodes but fails validation returns immediately: a greyscale
    scan will still be a greyscale scan on the third attempt, so retrying only
    spends the backoff budget. Transport failures do get the retries.
    """
    out_dir = Path(out_dir)
    final = out_dir / info.filename
    if final.is_file() and final.stat().st_size > 0:
        return final, ""
    tmp = final.with_suffix(final.suffix + ".part")
    reason = "download-failed"
    for attempt in range(retries):
        try:
            r = session.get(info.url, headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code != 200:
                reason = f"http-{r.status_code}"
            elif not r.content:
                reason = "empty-response"
            else:
                ok, why = validate_image(r.content)
                if not ok:
                    return None, why
                tmp.write_bytes(r.content)
                tmp.replace(final)
                return final, ""
        except Exception as exc:  # noqa: BLE001 - every transport failure retries alike
            reason = f"error:{type(exc).__name__}"
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(2**attempt)
    return None, reason


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
    licences: frozenset[str] = DEFAULT_LICENCES,
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

    infos, info_rejected = fetch_imageinfo(session, titles, width, licences)
    rejected.extend(info_rejected)
    say(f"{len(infos)} files pass licence and size checks; downloading at {width}px")

    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})[:3]
        raise FetchError(f"filename collision among selected files, e.g. {dup}")

    report = FetchReport(rejected=rejected)
    for i, info in enumerate(infos, start=1):
        final = out_dir / info.filename
        recorded = previous.get(info.filename)
        if final.is_file() and recorded:
            if hashlib.sha1(final.read_bytes()).hexdigest() != recorded:
                say(f"  {info.filename} does not match its recorded hash; refetching")
                final.unlink()
                report.repaired += 1
        path, reason = download(session, info, out_dir)
        if path is None:
            report.failed += 1
            rejected.append({"title": info.title, "reason": reason})
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
        "licence_policy": sorted(licences),
        "licences": dict(sorted(Counter(e["license"] for e in report.files).items())),
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


def write_attribution(manifest_path: str | Path, out_path: str | Path) -> int:
    """Write a Markdown attribution list from a fetch manifest; returns the file count.

    CC BY and CC BY-SA ask for attribution when a work is reused. A fitted
    LUT reuses no pixels, but the honest position is to make attribution
    trivial rather than argue it is unnecessary, and the manifest lives in a
    git-ignored data directory. This file is meant to be committed.
    """
    m = json.loads(Path(manifest_path).read_text())
    files = sorted(m["files"], key=lambda e: e["title"].lower())
    lines = [
        f"# Reference corpus attribution: {m['category']}",
        "",
        f"Fetched {m['fetched_at']} from Wikimedia Commons, licence policy "
        f"`{', '.join(m.get('licence_policy', ['pd']))}`. {len(files)} files. "
        "Each line is title, author as recorded by Commons, licence, and the file page.",
        "",
        "| # | title | author | licence |",
        "|---|---|---|---|",
    ]
    for i, e in enumerate(files, 1):
        title = e["title"].removeprefix("File:")
        page = "https://commons.wikimedia.org/wiki/" + e["title"].replace(" ", "_")
        author = (e.get("artist") or "unknown").replace("|", "/")
        lines.append(f"| {i} | [{title.replace('|', '/')}]({page}) | {author} | {e['license']} |")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return len(files)


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
    parser.add_argument("--attribution", type=Path, default=None,
                        help="also write a Markdown attribution list of every accepted file")
    parser.add_argument("--licences", default="pd",
                        help="comma-separated: pd (default), cc-by, cc-by-sa; NC/ND never")
    args = parser.parse_args(argv)
    try:
        licences = parse_licences(args.licences)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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
            licences=licences,
        )
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.attribution is not None:
        n = write_attribution(args.out / "manifest.json", args.attribution)
        print(f"attribution for {n} files written to {args.attribution}")
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
