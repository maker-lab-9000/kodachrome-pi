import io
import json

import numpy as np
import pytest
from PIL import Image

from kodachrome.train.fetch import (
    API_URL,
    FileInfo,
    download,
    fetch_category,
    fetch_imageinfo,
    iter_category_members,
    licence_allowed,
    main,
    parse_licences,
    select_titles,
    validate_image,
)

CAT = "Category:Test"
SUB = "Category:Sub"


def _photo_bytes(w=1200, h=900, seed=0):
    rgb = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _grey_bytes(w=1200, h=900):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w), 128, dtype=np.uint8), "L").save(buf, "JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, handler, files=None, fail_urls=()):
        self.handler = handler
        self.files = files or {}
        self.fail_urls = set(fail_urls)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        assert headers and "kodachrome-film" in headers["User-Agent"]
        if url == API_URL:
            return FakeResponse(payload=self.handler(params))
        if url in self.fail_urls:
            return FakeResponse(status=500)
        return FakeResponse(content=self.files.get(url, b""))


def _members(items, cont=None):
    out = {"query": {"categorymembers": items}}
    if cont:
        out["continue"] = {"cmcontinue": cont, "continue": "-||"}
    return out


def _handler(params):
    if params.get("list") == "categorymembers":
        if params["cmtitle"] == CAT and "cmcontinue" not in params:
            return _members(
                [
                    {"ns": 6, "title": "File:A LCCN2017000001.jpg", "pageid": 11},
                    {"ns": 14, "title": SUB, "pageid": 12},
                ],
                cont="c1",
            )
        if params["cmtitle"] == CAT:
            return _members([{"ns": 6, "title": "File:B LCCN2017000002.jpg", "pageid": 13}])
        if params["cmtitle"] == SUB:
            return _members(
                [
                    {"ns": 6, "title": "File:C (cropped) LCCN2017000003.jpg", "pageid": 14},
                    {"ns": 14, "title": CAT, "pageid": 15},
                ]
            )
    # Substring, not equality: the real call asks for "imageinfo|revisions",
    # because the revision id is part of the provenance record. Matching on
    # equality here silently sent every imageinfo call down the AssertionError
    # path, where api_get retried it three times with backoff before failing.
    if "imageinfo" in params.get("prop", ""):
        pages = {}
        for i, title in enumerate(params["titles"].split("|")):
            small = "small" in title
            nonfree = "nonfree" in title
            pages[str(i)] = {
                "title": title,
                "pageid": 100 + i,
                "imageinfo": [
                    {
                        "url": f"https://upload/{i}.jpg",
                        "thumburl": f"https://upload/thumb/{i}.jpg",
                        "width": 300 if small else 4000,
                        "height": 200 if small else 3000,
                        "mime": "image/jpeg",
                        "timestamp": "2020-01-01T00:00:00Z",
                        "extmetadata": {
                            "LicenseShortName": {
                                "value": "CC BY-SA 4.0" if nonfree else "Public domain"
                            }
                        },
                    }
                ],
                "revisions": [{"revid": 900 + i}],
            }
        return {"query": {"pages": pages}}
    raise AssertionError(f"unexpected params {params}")


@pytest.mark.parametrize(
    "text, allowed",
    [
        ("Public domain", True),
        ("CC0", True),
        ("PDM", True),
        ("PD-USGov", True),
        ("PD-1996", True),
        ("CC BY-SA 4.0", False),
        ("GFDL", False),
        ("", False),
        (None, False),
    ],
)
def test_licence_allowlist(text, allowed):
    assert licence_allowed(text) is allowed


def test_iter_category_members_follows_continue_and_recurses_once():
    titles = [m["title"] for m in iter_category_members(FakeSession(_handler), CAT)]
    assert titles == [
        "File:A LCCN2017000001.jpg",
        "File:C (cropped) LCCN2017000003.jpg",
        "File:B LCCN2017000002.jpg",
    ]


def test_select_titles_records_why_each_rejection_happened():
    entries = [
        {"title": "File:A LCCN2017000001.jpg"},
        {"title": "File:A again LCCN2017000001.jpg"},
        {"title": "File:C (cropped) LCCN2017000003.jpg"},
        {"title": "File:Zed no lccn.jpg"},
    ]
    accepted, rejected = select_titles(entries)
    assert accepted == ["File:A LCCN2017000001.jpg", "File:Zed no lccn.jpg"]
    reasons = {r["title"]: r["reason"] for r in rejected}
    assert reasons["File:A again LCCN2017000001.jpg"] == "duplicate-lccn"
    assert reasons["File:C (cropped) LCCN2017000003.jpg"] == "title-filter"


def test_fetch_imageinfo_rejects_small_and_non_free():
    infos, rejected = fetch_imageinfo(
        FakeSession(_handler),
        ["File:X LCCN2017000009.jpg", "File:small.jpg", "File:nonfree.jpg"],
        1024,
    )
    assert [i.title for i in infos] == ["File:X LCCN2017000009.jpg"]
    reasons = {r["title"]: r["reason"] for r in rejected}
    assert reasons["File:small.jpg"] == "too-small"
    assert reasons["File:nonfree.jpg"] == "licence"
    info = infos[0]
    assert info.url == "https://upload/thumb/0.jpg"
    assert info.lccn == "2017000009" and info.filename == "2017000009.jpg"
    assert info.pageid == 100 and info.revid == 900


def test_validate_image_accepts_photos_and_rejects_junk():
    ok, reason = validate_image(_photo_bytes())
    assert ok and reason == ""
    assert validate_image(b"not an image")[0] is False
    assert validate_image(_photo_bytes(w=400, h=300))[1] == "too-small"
    assert validate_image(_grey_bytes())[1] == "greyscale"


def test_download_is_atomic_and_leaves_nothing_on_failure(tmp_path):
    info = FileInfo(
        "File:T LCCN2017000001.jpg",
        1,
        2,
        "https://upload/1.jpg",
        1200,
        900,
        "Public domain",
        "2017000001",
    )
    session = FakeSession(_handler, files={"https://upload/1.jpg": _photo_bytes()})
    path, reason = download(session, info, tmp_path)
    assert path is not None and path.name == "2017000001.jpg" and reason == ""
    calls = len(session.calls)
    assert download(session, info, tmp_path) == (path, "")  # resumed, not re-fetched
    assert len(session.calls) == calls

    bad = FileInfo(
        "File:U LCCN2017000002.jpg",
        1,
        2,
        "https://upload/bad.jpg",
        1,
        1,
        "Public domain",
        "2017000002",
    )
    failing = FakeSession(_handler, fail_urls={"https://upload/bad.jpg"})
    assert download(failing, bad, tmp_path, retries=1) == (None, "http-500")
    assert list(tmp_path.glob("*.part")) == [], "no partial files may remain"
    assert not (tmp_path / "2017000002.jpg").exists()


def test_download_rejects_undecodable_content(tmp_path):
    info = FileInfo(
        "File:V LCCN2017000003.jpg",
        1,
        2,
        "https://upload/x.jpg",
        1200,
        900,
        "Public domain",
        "2017000003",
    )
    session = FakeSession(_handler, files={"https://upload/x.jpg": b"garbage"})
    path, reason = download(session, info, tmp_path)
    assert path is None
    assert reason.startswith("undecodable"), f"the manifest would record {reason!r}"
    assert not (tmp_path / "2017000003.jpg").exists()
    # Bad content is not retried: it will fail identically every time.
    assert len(session.calls) == 1


def test_a_rejected_photo_records_why_not_just_that_it_failed(tmp_path):
    """The manifest must distinguish a scanned document from a network error."""
    info = FileInfo(
        "File:G LCCN2017000007.jpg",
        1,
        2,
        "https://upload/g.jpg",
        1200,
        900,
        "Public domain",
        "2017000007",
    )
    session = FakeSession(_handler, files={"https://upload/g.jpg": _grey_bytes()})
    path, reason = download(session, info, tmp_path)
    assert path is None
    assert reason == "greyscale", "a document that decodes must not read as 'download-failed'"


@pytest.mark.parametrize(
    "title, expected",
    [("File:svg.jpg", "mime:image/svg+xml"), ("File:noinfo.jpg", "no-imageinfo")],
)
def test_metadata_level_rejections_are_named(title, expected):
    """Both defensive branches in fetch_imageinfo, which no fixture reached before."""

    def handler(params):
        if params.get("prop", "").startswith("imageinfo"):
            page = {"title": title, "pageid": 1, "revisions": [{"revid": 9}]}
            if "noinfo" not in title:
                page["imageinfo"] = [
                    {
                        "url": "https://upload/x.svg",
                        "thumburl": "https://upload/x.svg",
                        "width": 4000,
                        "height": 3000,
                        "mime": "image/svg+xml",
                        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
                    }
                ]
            return {"query": {"pages": {"0": page}}}
        raise AssertionError("unexpected params")

    infos, rejected = fetch_imageinfo(FakeSession(handler), [title], 1024)
    assert infos == []
    assert rejected == [{"title": title, "reason": expected}]


def test_fetch_category_writes_a_manifest_with_hashes_and_rejections(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    report = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    assert [e["lccn"] for e in report.files] == ["2017000001", "2017000002"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["category"] == CAT
    assert manifest["corpus_sha1"]
    assert any(r["reason"] == "title-filter" for r in manifest["rejected"])
    for entry in manifest["files"]:
        assert len(entry["sha1"]) == 40
        assert entry["pageid"] and entry["revid"]


def test_resume_revalidates_against_the_manifest_hash(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    victim = tmp_path / "2017000001.jpg"
    victim.write_bytes(_photo_bytes(seed=99))  # same name, different content
    report = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    assert victim.read_bytes() == files["https://upload/thumb/0.jpg"], (
        "corrupt file must be refetched"
    )
    assert report.repaired == 1


def test_main_enforces_min_files(tmp_path, monkeypatch, capsys):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    monkeypatch.setattr(
        "kodachrome.train.fetch.make_session", lambda: FakeSession(_handler, files=files)
    )
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "5"]) == 1
    assert "fewer than 5" in capsys.readouterr().err
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "2"]) == 0


def test_licence_policy_admits_exactly_what_it_names():
    """PD only by default; CC BY and CC BY-SA must be asked for; NC/ND never."""
    pd = frozenset({"pd"})
    by = frozenset({"pd", "cc-by"})
    bysa = frozenset({"pd", "cc-by", "cc-by-sa"})
    assert licence_allowed("Public domain") and licence_allowed("PD-USGov", pd)
    assert not licence_allowed("CC BY 3.0") and licence_allowed("CC BY 3.0", by)
    assert licence_allowed("Attribution", by)
    assert not licence_allowed("CC BY-SA 4.0", by) and licence_allowed("CC BY-SA 4.0", bysa)
    for bad in ("CC BY-NC-SA 2.0", "CC BY-ND 4.0", "GFDL 1.2", "", None):
        assert not licence_allowed(bad, bysa), bad
    assert parse_licences("pd, cc-by") == by
    with pytest.raises(ValueError, match="unknown licence policy"):
        parse_licences("pd,cc-by-nc")


def test_filenames_are_unique_even_when_title_stems_collide():
    """Two Commons titles that normalise to the same stem must not share a file."""
    def info(title, pageid, lccn=None):
        return FileInfo(title, pageid, 1, "u", 2000, 1500, "CC BY-SA 4.0", lccn)

    a = info("File:Ahaggar Mountains 1981 01.jpg", 111)
    b = info("File:Ahaggar Mountains 1981-01.jpg", 222)
    assert a.filename != b.filename
    assert a.filename.endswith("_111.jpg") and b.filename.endswith("_222.jpg")
    lccn = info("File:Whatever LCCN2017877392.tif", 333, "2017877392")
    assert lccn.filename == "2017877392.jpg"
