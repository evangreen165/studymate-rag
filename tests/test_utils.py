from __future__ import annotations

from studymate.utils import clean_text, language_ratio, sha256_text, stable_id


def test_clean_text_removes_page_numbers_and_extra_spaces() -> None:
    raw = "Заголовок\n\n  12  \nТекст   с   пробелами.\n"
    cleaned = clean_text(raw)
    assert "12" not in cleaned.split("\n")
    assert "Текст с пробелами." in cleaned


def test_language_ratio() -> None:
    assert language_ratio("Привет мир") == 1.0
    assert language_ratio("Hello world") == 0.0
    assert 0.4 < language_ratio("Привет world") < 0.6


def test_hashes_are_stable() -> None:
    assert sha256_text("abc") == sha256_text("abc")
    assert stable_id("a", 1, "b") == stable_id("a", 1, "b")
    assert stable_id("a", 1) != stable_id("a", 2)
