"""Golden-set resolution: word-boundary matching + comment-level gold."""
from __future__ import annotations

from eval.golden.build_seed import _phrase_pattern, resolve
from rag.ingest import Comment


def _c(sid: str, text: str) -> Comment:
    return Comment(source_id=sid, source="yt", text=text, timestamp=None)


def test_word_boundary_kills_substring_false_positives():
    # the audit's exact case: "sd" inside Spanish "desde"
    assert not _phrase_pattern("sd card").search("viendo el vídeo desde mi edge 50")
    assert not _phrase_pattern("heat").search("the heater and the theater")
    assert not _phrase_pattern("card").search("cardboard box")


def test_separator_tolerance():
    assert _phrase_pattern("144hz").search("a 144 hz panel")
    assert _phrase_pattern("144hz").search("smooth 144hz display")
    assert _phrase_pattern("low light").search("great low-light shots")
    assert _phrase_pattern("sd card").search("no sd-card slot")


def test_stem_wildcard():
    p = _phrase_pattern("heat*")
    assert p.search("it heats up") and p.search("heating issue") and p.search("too much heat")
    assert not p.search("wheat bread")


def test_resolve_is_comment_level_no_spillover():
    comments = [
        _c("a", "the battery drains fast after the android 15 update"),
        _c("b", "nice screen"),                      # adjacent junk must NOT become gold
        _c("c", "i love the android 15 skin"),       # only one phrase -> not gold
    ]
    gold = resolve(["battery", "android 15"], comments)
    assert gold == ["a"]


def test_resolve_empty_must_is_refuse():
    assert resolve([], [_c("a", "anything")]) == []
