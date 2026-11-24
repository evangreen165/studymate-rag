from __future__ import annotations

import re
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


# Маркеры тем: (подстрока в вопросе, вес). Сильные термины дают вес 3–4,
# общие слова — 1, чтобы маршрутизация не срабатывала на случайное совпадение.
_TOPIC_RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "git": (
        ("git", 4), ("гит", 4), ("коммит", 3), ("ветк", 2), ("merge", 3),
        ("rebase", 3), ("конфликт", 1), ("репозитор", 3), ("push", 2),
        ("pull", 2), ("stash", 3), ("checkout", 3), ("слиян", 2),
        ("перебазир", 3), ("revert", 3), ("reset", 2), ("origin", 2),
    ),
    "sql": (
        ("sql", 4), ("субд", 4), ("база данных", 3), ("базы данных", 3),
        ("баз данных", 3), ("таблиц", 1), ("индекс", 1), ("запрос", 1),
        ("join", 3), ("соединен", 2), ("транзакц", 3), ("нормализац", 3),
        ("денормализац", 3), ("первичн", 2), ("внешн", 2), ("select", 3),
        ("insert", 3), ("update", 3), ("delete", 3), ("mysql", 4),
        ("postgres", 4), ("sqlite", 4), ("view", 1), ("триггер", 2),
        ("права", 2), ("делегиров", 3), ("привилег", 2),
    ),
    "linux": (
        ("linux", 4), ("линукс", 4), ("bash", 3), ("терминал", 2),
        ("командн", 2), ("grep", 3), ("chmod", 3), ("chown", 3),
        ("файлов систем", 3), ("ядр", 2), ("демон", 2), ("systemd", 3),
        ("apt", 2), ("sudo", 3), ("конвейер", 2), ("pipe", 2),
        ("процесс", 1), ("поток", 1), ("systemctl", 4), ("cron", 3),
        ("модул", 2), ("макрос", 1), ("системный вызов", 3),
    ),
    "networks": (
        ("tcp", 4), ("udp", 4), ("ip", 2), ("dns", 3), ("http", 2),
        ("сеть", 2), ("сети", 2), ("протокол", 2), ("маршрутизатор", 3),
        ("маршрут", 2), ("коммутатор", 3), ("пакет", 2), ("ethernet", 3),
        ("osi", 3), ("сокет", 3), ("порт", 1), ("рукопожат", 3), ("nat", 3),
        ("vlan", 3), ("dhcp", 4), ("arp", 4), ("маска подсети", 3),
        ("ошибок", 1), ("канальн", 1),
    ),
    "oop": (
        ("ооп", 4), ("полиморф", 4), ("инкапсул", 4), ("наследован", 3),
        ("абстракт", 3), ("виртуальн", 3), ("перегруз", 3), ("класс", 1),
        ("объект", 1), ("интерфейс", 2), ("конструктор", 2),
        ("деструктор", 2), ("паттерн", 2), ("solid", 3), ("c++", 3),
        ("агрегац", 2), ("композиц", 2), ("поздн связыван", 3),
        ("базовый класс", 3), ("не изменяя существующий код", 4),
    ),
    "algorithms": (
        ("алгоритм", 3), ("сложност", 2), ("сортировк", 3), ("граф", 2),
        ("дерево", 2), ("динамическ", 2), ("рекурси", 3),
        ("структура данных", 3), ("хеш", 3), ("стек", 2), ("очередь", 2),
        ("бинарн", 2), ("жадн", 2), ("программирован", 1), ("матриц", 1),
        ("кратчайш", 2), ("обход", 1), ("куча", 2), ("связн список", 3),
        ("вычислен", 1),
    ),
    "cs": (
        ("computer science", 4), ("информатик", 3), ("архитектура эвм", 3),
        ("процессор", 2), ("бит", 1), ("байт", 1), ("кодирован", 2),
        ("компилятор", 2), ("интерпретатор", 2), ("операционн систем", 2),
    ),
}


def _marker_present(marker: str, text: str) -> bool:
    """Короткие аббревиатуры ищем как отдельные слова, основы — как подстроки."""
    if len(marker) <= 3:
        pattern = rf"(?<![a-zа-я0-9]){re.escape(marker)}(?![a-zа-я0-9])"
        return re.search(pattern, text) is not None
    return marker in text


def topic_scores(question: str) -> dict[str, int]:
    text = (question or "").lower().replace("ё", "е")
    scores: dict[str, int] = {}
    for slug, rules in _TOPIC_RULES.items():
        score = sum(weight for marker, weight in rules if _marker_present(marker, text))
        if score:
            scores[slug] = score
    return scores


def infer_topic_with_score(question: str) -> tuple[str | None, int]:
    scores = topic_scores(question)
    if not scores:
        return None, 0
    slug, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return (slug if score >= 2 else None), score


def infer_topic(question: str) -> str | None:
    """Определяет тему вопроса по ключевым терминам.

    Возвращает slug темы, если уверенность достаточная, иначе None —
    тогда поиск идёт по всему корпусу.
    """
    return infer_topic_with_score(question)[0]


def topic_label(slug: str | None) -> str:
    if not slug or slug == "all":
        return "Все темы"
    return TOPIC_BY_SLUG.get(slug, Topic(slug, slug, slug)).label
