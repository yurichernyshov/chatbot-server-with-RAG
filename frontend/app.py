import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://chatbot:8000")

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.title("RAG Chatbot с LLM")

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Боковая панель
with st.sidebar:
    st.header("Статистика")
    
    if st.button("Обновить статистику"):
        try:
            response = requests.get(f"{CHATBOT_URL}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                st.metric("Документов", stats.get("documents_count", 0))
                st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
                st.metric("LLM Model", stats.get("llm_model", "N/A"))
                st.success(f"Статус: {stats.get('status', 'unknown')}")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    
    st.divider()
    
    st.header("Загрузка документов")
    uploaded_files = st.file_uploader(
        "Загрузите TXT файлы",
        type=["txt"],
        accept_multiple_files=True
    )
    
    if st.button("Загрузить и индексировать"):
        if uploaded_files:
            try:
                files = [("files", (f.name, f.read(), "text/plain")) for f in uploaded_files]
                response = requests.post(
                    f"{CHATBOT_URL}/upload-documents",
                    files=files,
                    timeout=60
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Загружено файлов: {result['files_uploaded']}")
                    st.success(f"Проиндексировано чанков: {result['chunks_indexed']}")
                else:
                    st.error(f"Ошибка: {response.text}")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
        else:
            st.warning("Выберите файлы для загрузки")
    
    st.divider()
    
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

# Отображение истории чата
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("Источники"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Источник {i}:**")
                    st.text(source.get("content", "N/A")[:300])

# Поле ввода
if prompt := st.chat_input("Задайте вопрос по документам..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                response = requests.post(
                    f"{CHATBOT_URL}/query",
                    json={"question": prompt},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "Нет ответа")
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("Источники"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Источник {i}:**")
                                st.text(source.get("content", "N/A")[:300])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"Ошибка: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа")
            except Exception as e:
                st.error(f"Ошибка: {e}")

# Футер
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>RAG Chatbot powered by LLM + LangChain + ChromaDB + Ollama</small>
</div>
""", unsafe_allow_html=True)
