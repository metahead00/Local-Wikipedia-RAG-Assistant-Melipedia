import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.database import get_connection, init_db, get_all_entities, get_chunks_by_entity
from src.embedder import check_ollama_running
from src.vector_store import get_collection, collection_count
from src.retriever import retrieve, classify_query
from src.generator import generate_answer
from src.entities import PEOPLE, PLACES

st.set_page_config(page_title="Wikipedia RAG", page_icon="🧠", layout="wide")

@st.cache_resource
def load_resources():
    init_db()
    conn = get_connection()
    collection = get_collection()
    return conn, collection

conn, collection = load_resources()

# Sidebar
with st.sidebar:
    st.title("⚙️ System Status")

    ollama_ok = check_ollama_running()
    if ollama_ok:
        st.success("✅ Ollama running")
    else:
        st.error("❌ Ollama offline — start with: `ollama serve`")

    st.divider()
    st.subheader("📊 Database Stats")
    entities = get_all_entities(conn)
    n_chunks = collection_count(collection)
    st.metric("Entities", len(entities))
    st.metric("Chunks indexed", n_chunks)

    st.divider()
    st.subheader("🔧 Settings")
    show_context = st.toggle("Show retrieved context", value=False)
    st.caption("Model: tinyllama · Embedder: nomic-embed-text")

# Header
col1, col2 = st.columns([5, 1])
with col1:
    st.title("🧠 Wikipedia RAG")
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_context" not in st.session_state:
    st.session_state.show_context = False

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_context and msg.get("chunks"):
            with st.expander("📄 Retrieved context"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}] {chunk['entity_title']}** _(distance: {chunk['distance']:.3f})_")
                    st.text(chunk["chunk_text"][:400])
                    st.divider()

# Chat input
query = st.chat_input("Ask about a famous person or place...")

if query and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        query_type = classify_query(query, PEOPLE, PLACES)
        type_label = {"person": "👤 person", "place": "📍 place", "both": "🔍 both"}.get(query_type, query_type)

        with st.spinner(f"Searching {type_label}..."):
            chunks = retrieve(query, collection, conn)

        with st.spinner("Generating answer..."):
            result = generate_answer(query, chunks)

        answer = result["answer"]
        st.markdown(answer)

        if show_context and chunks:
            with st.expander("📄 Retrieved context"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**[{i}] {chunk['entity_title']}** _(distance: {chunk['distance']:.3f})_")
                    st.text(chunk["chunk_text"][:400])
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks
        })
