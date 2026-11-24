from __future__ import annotations

import pytest

from studymate.topics import infer_topic, infer_topic_with_score, topic_scores


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Что такое полиморфизм в ООП?", "oop"),
        ("Что такое инкапсуляция и зачем она нужна?", "oop"),
        ("Как создать индекс в SQL и когда он помогает?", "sql"),
        ("Чем отличается git merge от git rebase?", "git"),
        ("Как работает трехстороннее рукопожатие TCP?", "networks"),
        ("Что такое динамическое программирование?", "algorithms"),
        ("Как создать несколько точек входа в модуль Linux?", "linux"),
    ],
)
def test_infer_topic(question: str, expected: str) -> None:
    assert infer_topic(question) == expected


def test_short_acronym_does_not_match_inside_word() -> None:
    # «макроопределения» содержит подстроку «ооп», но это не тема ООП.
    question = "Как создать несколько точек входа в модуль Linux, используя макроопределения?"
    assert topic_scores(question).get("oop", 0) == 0
    assert infer_topic(question) == "linux"


def test_unknown_topic_returns_none() -> None:
    assert infer_topic("Какова вероятность выпадения орла?") is None


def test_confidence_is_reported() -> None:
    slug, confidence = infer_topic_with_score("Чем отличается git merge от git rebase?")
    assert slug == "git"
    assert confidence >= 4
