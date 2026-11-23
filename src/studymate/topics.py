from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    slug: str
    label: str
    short: str


TOPICS: list[Topic] = [
    Topic("cs", "Computer Science", "CS"),
    Topic("git", "Git", "Git"),
    Topic("linux", "Linux", "Linux"),
    Topic("sql", "SQL и базы данных", "SQL"),
    Topic("algorithms", "Алгоритмы и структуры данных", "Алгоритмы"),
    Topic("networks", "Компьютерные сети", "Сети"),
    Topic("oop", "ООП", "ООП"),
    Topic("other", "Другое", "Другое"),
]
TOPIC_BY_SLUG = {topic.slug: topic for topic in TOPICS}
TOPIC_BY_LABEL = {topic.label: topic for topic in TOPICS}


def topic_label(slug: str | None) -> str:
    if not slug or slug == "all":
        return "Все темы"
    return TOPIC_BY_SLUG.get(slug, Topic(slug, slug, slug)).label
