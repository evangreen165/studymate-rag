from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import api_client
import chat_store
from studymate.topics import TOPICS

st.set_page_config(
    page_title="StudyMate — ассистент по учебным материалам",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root { --sm-accent: #4f8cff; --sm-accent2: #7c5cff; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1380px; }
#MainMenu, footer { visibility: hidden; }
.hero {
    background: linear-gradient(120deg, rgba(79,140,255,.16), rgba(124,92,255,.14));
    border: 1px solid rgba(79,140,255,.25); border-radius: 18px;
    padding: 22px 26px; margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -.02em; }
.hero p { margin: 6px 0 0; opacity: .8; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
    display: inline-block; padding: 4px 11px; border-radius: 999px;
    background: rgba(79,140,255,.12); border: 1px solid rgba(79,140,255,.28);
    font-size: .78rem; white-space: nowrap;
}
.chip.ok { background: rgba(35,180,120,.13); border-color: rgba(35,180,120,.35); }
.chip.warn { background: rgba(240,170,40,.14); border-color: rgba(240,170,40,.4); }
.chip.err { background: rgba(235,80,80,.14); border-color: rgba(235,80,80,.4); }
.source-card {
    border-left: 3px solid var(--sm-accent); padding: 8px 12px; margin: 8px 0;
    background: rgba(79,140,255,.06); border-radius: 0 10px 10px 0;
}
.source-card b { font-size: .92rem; }
.source-card .meta { opacity: .65; font-size: .78rem; }
.small-muted { opacity: .65; font-size: .8rem; }
div[data-testid="stChatMessage"] { border-radius: 14px; }
div[data-testid="stMetricValue"] { font-size: 1.45rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 6px 14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=8, show_spinner=False)
def cached_documents() -> dict:
    return api_client.documents()


@st.cache_data(ttl=15, show_spinner=False)
def cached_topics() -> list[dict]:
    return api_client.topics()


@st.cache_data(ttl=10, show_spinner=False)
def cached_stats() -> dict:
    return api_client.stats()


def clear_cache() -> None:
    st.cache_data.clear()


def api_state() -> tuple[bool, dict | None, str]:
    try:
        health = api_client.health()
        return True, health, ""
    except api_client.APIError as exc:
        return False, None, str(exc)


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}".replace(",", " ")


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.2f} с" if value >= 1000 else f"{value:.0f} мс"


def poll_index(placeholder, *, max_seconds: int = 900) -> dict:
    """Показывает прогресс фоновой индексации, не блокируя сервер."""
    started = time.time()
    last: dict = {}
    while time.time() - started < max_seconds:
        try:
            last = api_client.index_status()
        except api_client.APIError as exc:
            placeholder.error(f"Не удалось получить статус: {exc}")
            return last
        progress = min(int(last.get("progress", 0)), 100) / 100
        placeholder.progress(progress, text=last.get("message", "Индексация…"))
        if last.get("status") in {"done", "error", "cancelled"}:
            if last.get("status") == "done":
                placeholder.success(last.get("message", "Индексация завершена"))
            elif last.get("status") == "error":
                placeholder.error(last.get("error", "Ошибка индексации"))
            else:
                placeholder.warning(last.get("message", "Индексация отменена"))
            return last
        time.sleep(1.5)
    placeholder.info(
        "Индексация продолжается на сервере. Обновите статус позже — вкладку можно закрыть."
    )
    return last


def init_state() -> None:
    if "active_chat_id" not in st.session_state:
        chat = chat_store.active_chat() or chat_store.create_chat()
        st.session_state.active_chat_id = chat["id"]
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("upload_results", None)
    st.session_state.setdefault("feedback", {})
    st.session_state.setdefault("top_k", 6)
    st.session_state.setdefault("provider", "openrouter")
    st.session_state.setdefault("model", "")
    st.session_state.setdefault("max_tokens", 1024)


init_state()
online, health, health_error = api_state()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.55rem;font-weight:800;letter-spacing:-.03em'>📚 StudyMate</div>"
        "<div class='small-muted'>RAG-ассистент по материалам курса</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    if online:
        st.markdown(
            f"<span class='chip ok'>● API онлайн · v{health.get('version','?')}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span class='chip err'>● API недоступен</span>", unsafe_allow_html=True)
        st.caption(health_error)
        st.code("uv run uvicorn studymate.api.main:app --port 8000", language="bash")

    st.divider()
    try:
        provider_data = api_client.providers()
    except api_client.APIError:
        provider_data = {}
    provider_options = ["openrouter", "local", "mock"]
    default_provider = provider_data.get("default", "openrouter")
    provider = st.selectbox(
        "Провайдер LLM",
        provider_options,
        index=provider_options.index(default_provider) if default_provider in provider_options else 0,
        key="provider",
    )
    if provider == "openrouter":
        default_model = provider_data.get("openrouter", {}).get("model", "")
        model = st.text_input("Модель OpenRouter", value=st.session_state.model or default_model)
        st.session_state.model = model
        if not provider_data.get("openrouter", {}).get("configured", False):
            st.warning("OPENROUTER_API_KEY не задан в .env")
    elif provider == "local":
        local = provider_data.get("local", {})
        st.caption(f"GGUF: {local.get('model','—')}")
        if not local.get("configured"):
            st.warning("Локальная GGUF-модель не найдена")
    else:
        st.caption("Mock-провайдер: детерминированные ответы для тестов.")

    topic_slug = st.selectbox(
        "Тема",
        ["all"] + [topic.slug for topic in TOPICS],
        format_func=lambda slug: "Все темы" if slug == "all" else next(
            (topic.label for topic in TOPICS if topic.slug == slug), slug
        ),
    )
    st.session_state.top_k = st.slider("Фрагментов в контексте (top-k)", 1, 12, st.session_state.top_k)
    with st.expander("Дополнительно"):
        st.session_state.max_tokens = st.slider(
            "Максимум токенов ответа", 256, 4096, st.session_state.max_tokens, step=128
        )

    st.divider()
    col_new, col_count = st.columns([2, 1])
    with col_new:
        if st.button("➕ Новый чат", use_container_width=True):
            chat = chat_store.create_chat()
            st.session_state.active_chat_id = chat["id"]
            st.session_state.feedback = {}
            st.rerun()
    with col_count:
        st.caption(f"{len(chat_store.list_chats())} чатов")

    chats = chat_store.list_chats()
    for chat in chats[:12]:
        selected = chat["id"] == st.session_state.active_chat_id
        label = ("▸ " if selected else "") + chat["title"][:34]
        cols = st.columns([5, 1])
        if cols[0].button(label, key=f"chat_{chat['id']}", use_container_width=True):
            chat_store.set_active(chat["id"])
            st.session_state.active_chat_id = chat["id"]
            st.session_state.feedback = {}
            st.rerun()
        if cols[1].button("✕", key=f"del_{chat['id']}", help="Удалить чат"):
            chat_store.delete_chat(chat["id"])
            if st.session_state.active_chat_id == chat["id"]:
                remaining = chat_store.list_chats()
                st.session_state.active_chat_id = remaining[0]["id"] if remaining else chat_store.create_chat()["id"]
            st.rerun()

    active_chat_obj = chat_store.active_chat()
    if active_chat_obj:
        st.download_button(
            "⬇️ Экспорт чата",
            data=chat_store.export_chat(active_chat_obj),
            file_name=f"studymate_chat_{active_chat_obj['id']}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with st.expander("Очистить историю"):
        st.caption("Удаляются все чаты на этом компьютере.")
        if st.button("Удалить все чаты", type="secondary"):
            chat_store.clear_all()
            chat = chat_store.create_chat()
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    st.divider()
    st.caption("GitHub · evangreen165/studymate-rag")

# ---------------- Header ----------------
stats_data: dict = {}
docs_data: dict = {}
if online:
    try:
        stats_data = cached_stats()
        docs_data = cached_documents()
    except api_client.APIError as exc:
        st.error(f"Не удалось получить данные: {exc}")

summary = docs_data.get("summary", {})
index_meta = stats_data.get("index", {}).get("meta", {}) or {}
chunks = stats_data.get("index", {}).get("chunks", 0) or 0
bm25_chunks = stats_data.get("index", {}).get("meta", {}).get("bm25_chunks", 0) or 0
provider_info = stats_data.get("providers", {}).get(provider, {})
model_label = model if provider == "openrouter" else provider_info.get("model", provider)

st.markdown(
    f"""
    <div class="hero">
      <h1>Ассистент по учебным материалам курса</h1>
      <p>Отвечает на вопросы студентов по PDF-конспектам и показывает источники с номерами страниц.</p>
      <div class="chips">
        <span class="chip {'ok' if online else 'err'}">{'API онлайн' if online else 'API недоступен'}</span>
        <span class="chip">📄 {fmt_int(summary.get('unique_documents', summary.get('documents', 0)))} документов</span>
        <span class="chip">📃 {fmt_int(summary.get('text_pages', 0))} страниц с текстом</span>
        <span class="chip">🧩 {fmt_int(chunks)} плотных · {fmt_int(bm25_chunks)} BM25</span>
        <span class="chip">🧠 {model_label or '—'}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_tab, materials_tab, metrics_tab, about_tab = st.tabs(
    ["💬 Чат", "📁 Материалы и индекс", "📊 Метрики агента", "ℹ️ О системе"]
)

# ---------------- Chat ----------------
with chat_tab:
    active = chat_store.active_chat()
    if active is None:
        active = chat_store.create_chat()
        st.session_state.active_chat_id = active["id"]

    if not active["messages"]:
        st.markdown("#### С чего начать")
        st.caption("Выберите пример или задайте свой вопрос по материалам курса.")
        examples = [
            "Чем отличается git merge от git rebase?",
            "Как работает LEFT JOIN и чем он отличается от INNER JOIN?",
            "Что такое полиморфизм в ООП?",
            "Как оценивается сложность алгоритма O(n log n)?",
        ]
        cols = st.columns(2)
        for index, example in enumerate(examples):
            if cols[index % 2].button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    for message in active["messages"]:
        role = message.get("role", "assistant")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))
            continue
        meta = message.get("meta") or {}
        with st.chat_message("assistant", avatar="📚"):
            if meta.get("error"):
                st.error(meta["error"])
                continue
            st.markdown(message.get("content", ""))
            if meta.get("refusal"):
                st.info("Ассистент не нашёл ответа в предоставленных материалах и честно сообщил об этом.")
            sources = meta.get("sources") or []
            if sources:
                with st.expander(f"Источники · {len(sources)}", expanded=False):
                    for number, source in enumerate(sources, start=1):
                        st.markdown(
                            f"<div class='source-card'><b>[{number}] {source.get('title','—')}</b>"
                            f"<div class='meta'>стр. {source.get('page','—')} · "
                            f"тема: {source.get('topic','—')} · score {source.get('score',0):.4f}</div>"
                            f"<div class='small-muted'>{source.get('snippet','')[:500]}…</div></div>",
                            unsafe_allow_html=True,
                        )
            usage = meta.get("usage") or {}
            timings = meta.get("timings") or {}
            request_id = meta.get("request_id", "")
            st.caption(
                f"⏱ {fmt_ms(timings.get('total_ms'))} · "
                f"🔎 поиск {fmt_ms(timings.get('retrieval_ms'))} · "
                f"✍️ генерация {fmt_ms(timings.get('generation_ms'))} · "
                f"🪙 {fmt_int(usage.get('total_tokens'))} ток. "
                f"({fmt_int(usage.get('prompt_tokens'))} вход + {fmt_int(usage.get('completion_tokens'))} выход) · "
                f"⚡ {usage.get('tokens_per_second', 0):.1f} ток/с · "
                f"{meta.get('provider','—')}/{meta.get('model','—')}"
            )
            if request_id:
                feedback_state = st.session_state.feedback.get(request_id)
                fcol1, fcol2, fcol3 = st.columns([1, 1, 8])
                if fcol1.button("👍", key=f"up_{request_id}", disabled=feedback_state is not None):
                    try:
                        api_client.feedback(request_id, 1)
                        st.session_state.feedback[request_id] = 1
                        st.toast("Спасибо за оценку")
                    except api_client.APIError as exc:
                        st.toast(f"Не удалось сохранить оценку: {exc}")
                if fcol2.button("👎", key=f"down_{request_id}", disabled=feedback_state is not None):
                    try:
                        api_client.feedback(request_id, -1)
                        st.session_state.feedback[request_id] = -1
                        st.toast("Спасибо, ошибка будет учтена")
                    except api_client.APIError as exc:
                        st.toast(f"Не удалось сохранить оценку: {exc}")
                if feedback_state is not None:
                    fcol3.caption("Оценка сохранена")

    question = st.chat_input("Задайте вопрос по учебным материалам…")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = ""

    if question:
        if not question.strip():
            st.warning("Введите вопрос.")
        else:
            chat_store.append_message(
                active["id"], {"role": "user", "content": question.strip(), "ts": datetime.utcnow().isoformat()}
            )
            with st.spinner("Ищу фрагменты в материалах и формулирую ответ…"):
                client_started = time.perf_counter()
                try:
                    result = api_client.chat(
                        question.strip(),
                        top_k=st.session_state.top_k,
                        topic=topic_slug,
                        provider=provider,
                        model=st.session_state.model if provider == "openrouter" else None,
                        max_tokens=st.session_state.max_tokens,
                    )
                    client_ms = (time.perf_counter() - client_started) * 1000
                    meta = {
                        "request_id": result.get("request_id"),
                        "provider": result.get("provider"),
                        "model": result.get("model"),
                        "usage": result.get("usage", {}),
                        "timings": {**result.get("timings", {}), "client_ms": round(client_ms, 2)},
                        "sources": result.get("sources", []),
                        "used_sources": result.get("used_sources", []),
                        "refusal": result.get("refusal", False),
                        "topic": result.get("topic"),
                        "top_k": result.get("top_k"),
                    }
                    chat_store.append_message(
                        active["id"],
                        {
                            "role": "assistant",
                            "content": result.get("answer", ""),
                            "meta": meta,
                            "ts": datetime.utcnow().isoformat(),
                        },
                    )
                except api_client.APIError as exc:
                    chat_store.append_message(
                        active["id"],
                        {
                            "role": "assistant",
                            "content": "",
                            "meta": {"error": str(exc)},
                            "ts": datetime.utcnow().isoformat(),
                        },
                    )
            clear_cache()
            st.rerun()

# ---------------- Materials ----------------
with materials_tab:
    if not online:
        st.error("API недоступен — управление материалами требует запущенного сервера.")
    else:
        st.subheader("Корпус и индекс")
        cols = st.columns(6)
        cols[0].metric("Документов", fmt_int(summary.get("unique_documents", 0)))
        cols[1].metric("Страниц", fmt_int(summary.get("pages", 0)))
        cols[2].metric("С текстом", fmt_int(summary.get("text_pages", 0)))
        cols[3].metric("Плотных", fmt_int(chunks))
        cols[4].metric("BM25", fmt_int(bm25_chunks))
        cols[5].metric("Объём", f"{(summary.get('size_bytes', 0) / 1024 / 1024):.0f} МБ")

        docs = docs_data.get("documents", [])
        if docs:
            frame = pd.DataFrame(docs)
            topic_counts = (
                frame[frame["status"] != "duplicate"]
                .groupby("topic", as_index=True)
                .size()
                .sort_values(ascending=False)
            )
            st.bar_chart(topic_counts, height=220)
        else:
            st.info("В манифесте пока нет документов. Загрузите PDF или соберите манифест.")

        st.markdown("#### Управление индексом")
        status = api_client.index_status()
        status_cols = st.columns([2, 1, 1])
        badge = {
            "idle": "🟡 ожидает",
            "running": "🔵 выполняется",
            "done": "🟢 готово",
            "error": "🔴 ошибка",
            "cancelled": "⚪ отменено",
        }.get(status.get("status"), status.get("status"))
        status_cols[0].markdown(f"**Статус:** {badge} · {status.get('message','')}")
        if status.get("total"):
            status_cols[0].caption(
                f"Чанков: {status.get('embedded',0)} / {status.get('total',0)} · "
                f"режим: {'пересборка' if status.get('rebuild') else 'добавление'}"
            )
        if status.get("status") == "running":
            status_cols[0].progress(min(int(status.get("progress", 0)), 100) / 100)
        if status_cols[1].button("Обновить статус", use_container_width=True):
            clear_cache()
            st.rerun()
        if status.get("status") == "running" and status_cols[2].button(
            "Остановить", use_container_width=True
        ):
            api_client.cancel_index()
            st.rerun()
        if status.get("status") == "error":
            st.error(status.get("error", "Неизвестная ошибка индексации"))
        if status.get("status") == "done":
            st.success(status.get("message", "Индексация завершена"))

        st.markdown("#### Загрузить новые материалы")
        upload_cols = st.columns([3, 1])
        uploaded_files = upload_cols[0].file_uploader(
            "PDF-файлы (до 50 файлов, до 100 МБ каждый)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        upload_topic = upload_cols[1].selectbox(
            "Тема загружаемых файлов",
            [topic.slug for topic in TOPICS],
            format_func=lambda slug: next(
                (topic.label for topic in TOPICS if topic.slug == slug), slug
            ),
            index=len(TOPICS) - 1,
        )
        if st.button("⬆️ Загрузить в корпус", disabled=not uploaded_files):
            payload = [(file.name, file.getvalue()) for file in uploaded_files]
            try:
                result = api_client.upload(payload, upload_topic)
                st.session_state.upload_results = result
                clear_cache()
                st.success(
                    f"Загружено: {len(result.get('uploaded', []))}. "
                    "Теперь запустите индексацию загруженных документов."
                )
                if result.get("errors"):
                    for error in result["errors"]:
                        st.warning(error)
            except api_client.APIError as exc:
                st.error(f"Ошибка загрузки: {exc}")

        upload_results = st.session_state.get("upload_results") or {}
        uploaded_docs = upload_results.get("uploaded", [])
        if uploaded_docs:
            st.caption("Загруженные документы: " + ", ".join(doc["title"] for doc in uploaded_docs))
            if st.button("🧩 Проиндексировать загруженные", type="primary"):
                try:
                    api_client.start_index(
                        rebuild=False, doc_ids=[doc["doc_id"] for doc in uploaded_docs]
                    )
                    progress_placeholder = st.empty()
                    poll_index(progress_placeholder)
                    clear_cache()
                    st.rerun()
                except api_client.APIError as exc:
                    st.error(f"Не удалось запустить индексацию: {exc}")

        st.markdown("#### Полная переиндексация")
        st.caption(
            "Пересобирает индекс по всем документам манифеста. Может занимать десятки минут; "
            "прогресс сохраняется на сервере, вкладку можно закрыть."
        )
        confirm = st.checkbox("Подтверждаю полную переиндексацию")
        if st.button("🔄 Переиндексировать всё", disabled=not confirm):
            try:
                api_client.start_index(rebuild=True)
                progress_placeholder = st.empty()
                poll_index(progress_placeholder)
                clear_cache()
                st.rerun()
            except api_client.APIError as exc:
                st.error(f"Не удалось запустить индексацию: {exc}")

        st.markdown("#### Документы корпуса")
        if docs:
            filter_cols = st.columns([2, 1, 2])
            search = filter_cols[0].text_input("Поиск по названию", "")
            topic_filter = filter_cols[1].selectbox(
                "Тема",
                ["all"] + [topic.slug for topic in TOPICS],
                format_func=lambda slug: "Все" if slug == "all" else next(
                    (topic.label for topic in TOPICS if topic.slug == slug), slug
                ),
                key="docs_topic_filter",
            )
            status_filter = filter_cols[2].multiselect(
                "Статус",
                ["indexed", "sparse_only", "not_indexed", "scanned", "error", "duplicate"],
                default=["indexed", "sparse_only", "not_indexed", "scanned", "error"],
            )
            filtered = [
                doc
                for doc in docs
                if (not search or search.lower() in doc["title"].lower())
                and (topic_filter == "all" or doc["topic_slug"] == topic_filter)
                and doc["status"] in status_filter
            ]
            table = pd.DataFrame(filtered)
            if not table.empty:
                table = table[
                    ["title", "topic", "pages", "text_pages", "chunks", "status", "size_bytes"]
                ].rename(
                    columns={
                        "title": "Документ",
                        "topic": "Тема",
                        "pages": "Страниц",
                        "text_pages": "С текстом",
                        "chunks": "Чанков",
                        "status": "Статус",
                        "size_bytes": "Байт",
                    }
                )
                st.dataframe(table, use_container_width=True, hide_index=True, height=320)
            else:
                st.info("Ничего не найдено по фильтрам.")

            st.markdown("#### Удаление документа")
            doc_options = {f"{doc['title']} · {doc['topic']} · {doc['status']}": doc for doc in docs}
            selected_label = st.selectbox("Документ", list(doc_options.keys()))
            selected_doc = doc_options[selected_label]
            delete_file = st.checkbox(
                "Удалить PDF-файл (только для загруженных материалов)", value=False
            )
            if st.button("🗑 Удалить из индекса", type="secondary"):
                try:
                    api_client.delete_document(selected_doc["doc_id"])
                    if delete_file:
                        st.info("Файл загруженного документа удалён, если он находился в data/uploads.")
                    clear_cache()
                    st.success("Документ удалён из манифеста и индекса.")
                    st.rerun()
                except api_client.APIError as exc:
                    st.error(f"Не удалось удалить: {exc}")
        else:
            st.info("Документов нет.")

# ---------------- Metrics ----------------
with metrics_tab:
    st.subheader("Значения работы агента")
    chats = chat_store.list_chats()
    rows = []
    for chat in chats:
        for message in chat.get("messages", []):
            if message.get("role") != "assistant":
                continue
            meta = message.get("meta") or {}
            usage = meta.get("usage") or {}
            timings = meta.get("timings") or {}
            rows.append(
                {
                    "chat": chat["title"],
                    "request_id": meta.get("request_id", ""),
                    "provider": meta.get("provider", ""),
                    "model": meta.get("model", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "tokens_per_second": usage.get("tokens_per_second", 0),
                    "retrieval_ms": timings.get("retrieval_ms", 0),
                    "generation_ms": timings.get("generation_ms", 0),
                    "total_ms": timings.get("total_ms", 0),
                    "sources": len(meta.get("sources") or []),
                    "refusal": bool(meta.get("refusal")),
                    "error": bool(meta.get("error")),
                }
            )
    if not rows:
        st.info("Пока нет ответов агента. Задайте вопрос во вкладке «Чат».")
    else:
        frame = pd.DataFrame(rows)
        total_tokens = int(frame["total_tokens"].sum())
        total_completion = int(frame["completion_tokens"].sum())
        total_prompt = int(frame["prompt_tokens"].sum())
        avg_tps = float(frame.loc[frame["tokens_per_second"] > 0, "tokens_per_second"].mean() or 0)
        avg_latency = float(frame["total_ms"].mean())
        cols = st.columns(5)
        cols[0].metric("Запросов", len(frame))
        cols[1].metric("Всего токенов", fmt_int(total_tokens))
        cols[2].metric("Вход / выход", f"{fmt_int(total_prompt)} / {fmt_int(total_completion)}")
        cols[3].metric("Средняя скорость", f"{avg_tps:.1f} ток/с")
        cols[4].metric("Средний ответ", fmt_ms(avg_latency))

        chart_cols = st.columns(3)
        chart_cols[0].caption("Токены на запрос")
        chart_cols[0].bar_chart(frame["total_tokens"], height=200)
        chart_cols[1].caption("Скорость, ток/с")
        chart_cols[1].bar_chart(frame["tokens_per_second"], height=200)
        chart_cols[2].caption("Задержка, мс")
        chart_cols[2].bar_chart(frame["total_ms"], height=200)

        st.markdown("#### Журнал запросов")
        st.dataframe(
            frame[
                [
                    "chat",
                    "provider",
                    "model",
                    "total_tokens",
                    "tokens_per_second",
                    "retrieval_ms",
                    "generation_ms",
                    "total_ms",
                    "sources",
                    "refusal",
                    "error",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Скачать CSV",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="studymate_metrics.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("#### Индекс и провайдеры")
    if stats_data:
        index_info = stats_data.get("index", {})
        cols = st.columns(4)
        cols[0].metric("Плотных векторов", fmt_int(index_info.get("chunks", 0)))
        cols[1].metric("BM25-чанков", fmt_int(index_info.get("meta", {}).get("bm25_chunks", 0)))
        cols[2].metric("Модель эмбеддингов", stats_data.get("embedding_model", "—"))
        cols[3].metric("Chunk size / overlap", f"{stats_data.get('chunk_size','—')} / {stats_data.get('chunk_overlap','—')}")
        with st.expander("Полный ответ /stats"):
            st.json(stats_data)
    else:
        st.info("Статистика API недоступна.")

# ---------------- About ----------------
with about_tab:
    st.subheader("Как устроен StudyMate")
    st.markdown(
        """
**Сценарий.** Студент задаёт вопрос по материалам курса. Система ищет релевантные фрагменты
в векторном индексе, передаёт их LLM вместе с вопросом и возвращает ответ со ссылками на
документ и страницу.

**Пайплайн:**
`PDF → PyMuPDF → очистка → LlamaIndex SentenceSplitter → USER2-base → ChromaDB → top-k/MMR + BM25 → промпт → Llama-3.1-70B → ответ + источники`.

**Что можно делать:**
- задавать вопросы и видеть цитаты, страницы и оценку релевантности;
- фильтровать поиск по теме;
- смотреть историю чатов, экспортировать её;
- загружать новые PDF, запускать индексацию и удалять документы;
- смотреть токены, скорость (ток/с), задержки поиска и генерации, размер контекста;
- переключаться между OpenRouter, локальной GGUF-моделью и mock-провайдером.
        """
    )
    st.markdown("#### Ограничения")
    st.markdown(
        """
- Ответ ограничен найденным контекстом; если материала нет, ассистент отказывается отвечать.
- PDF-сканы без текстового слоя не индексируются (видны в статусе `scanned`).
- Формулы, таблицы и листинги кода из PDF могут терять структуру.
- LLM может ошибаться; источники нужно проверять.
- Полный корпус из 41 PDF не входит в публичный репозиторий по условиям авторских прав.
        """
    )
    st.markdown("#### API")
    st.code(
        "GET  /health          — статус сервиса\n"
        "GET  /ready           — готовность индекса\n"
        "POST /chat            — вопрос → ответ + источники\n"
        "GET  /documents       — документы корпуса\n"
        "GET  /topics          — темы и количество документов\n"
        "POST /corpus/upload   — загрузка PDF\n"
        "POST /index           — запуск индексации\n"
        "GET  /index/status    — прогресс индексации\n"
        "GET  /stats           — статистика индекса и провайдеров\n"
        "GET  /metrics         — Prometheus-метрики\n"
        "GET  /docs            — OpenAPI/Swagger",
        language="text",
    )
    st.caption("Репозиторий: github.com/evangreen165/studymate-rag · Лицензия MIT")
