"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    # TODO: implement
    # 1. st.sidebar.file_uploader(
    #        "Upload study materials",
    #        type=["pdf", "md"],
    #        accept_multiple_files=True
    #    )
    #
    # 2. "Ingest Documents" button — only enabled when files are selected
    #
    # 3. On button click:
    #    a. Save uploaded files to a temp directory
    #    b. chunker.chunk_files(file_paths)
    #    c. store.ingest(chunks) → IngestionResult
    #    d. Display result: st.success / st.warning / st.error
    #       Show: "{result.ingested} chunks added, {result.skipped} duplicates skipped"
    #    e. Refresh ingested documents list in session_state
    #
    # 4. Render ingested documents list below the uploader
    #    For each document: show source name, topic, chunk count
    #    Add a small "🗑 Remove" button per document that calls store.delete_document()

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["md"],
        accept_multiple_files=True,
    )

    ingest_clicked = st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:
        try:
            all_chunks = []

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for uploaded_file in uploaded_files:
                    file_path = temp_path / uploaded_file.name
                    file_path.write_bytes(uploaded_file.getvalue())
                    all_chunks.extend(chunker.chunk_file(file_path))

            result = store.ingest(all_chunks)
            st.session_state.last_ingestion_result = result

            if result.errors:
                st.sidebar.error(
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped, "
                    f"{len(result.errors)} errors"
                )
            elif result.ingested > 0:
                st.sidebar.success(
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped"
                )
            else:
                st.sidebar.warning(
                    f"{result.ingested} chunks added, "
                    f"{result.skipped} duplicates skipped"
                )

            st.session_state.ingested_documents = store.list_documents()

        except Exception as exc:
            st.sidebar.error(f"Ingestion failed: {exc}")

    st.session_state.ingested_documents = store.list_documents()

    if st.session_state.ingested_documents:
        st.sidebar.subheader("Ingested Documents")

        for document in st.session_state.ingested_documents:
            st.sidebar.write(
                f"**{document['source']}** — "
                f"{document['topic']} "
                f"({document['chunk_count']} chunks)"
            )

            if st.sidebar.button(
                f"🗑 Remove {document['source']}",
                key=f"remove_{document['source']}",
                use_container_width=True,
            ):
                deleted = store.delete_document(document["source"])
                st.sidebar.success(
                    f"Removed {deleted} chunks from {document['source']}"
                )
                st.session_state.ingested_documents = store.list_documents()
                st.rerun()
    else:
        st.sidebar.info("Upload .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    stats = store.get_collection_stats()

    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    else:
        st.sidebar.write("Topics: none yet")

    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox("Select document", options=[doc["source"] for doc in docs])
    #    Store selection in st.session_state["selected_document"]
    #
    # 3. On selection change: store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge: topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was used in last response
    #
    # 5. Display chunk count and coverage summary below viewer

    docs = store.list_documents()

    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in docs]

    selected_source = st.selectbox(
        "Select document",
        options=sources,
        index=(
            sources.index(st.session_state.selected_document)
            if st.session_state.selected_document in sources
            else 0
        ),
    )

    st.session_state.selected_document = selected_source
    chunks = store.get_document_chunks(selected_source)

    content_container = st.container(height=420)

    with content_container:
        for index, chunk in enumerate(chunks, start=1):
            st.caption(
                f"{chunk.metadata.topic} | "
                f"{chunk.metadata.difficulty} | "
                f"{chunk.metadata.type}"
            )
            st.markdown(chunk.chunk_text)

            if index < len(chunks):
                st.divider()

    st.caption(
        f"{len(chunks)} chunks loaded from {selected_source}"
    )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        # TODO: st.selectbox for topic filter
        topic_options = [
            "All",
            "ANN",
            "CNN",
            "RNN",
            "LSTM",
            "Seq2Seq",
            "Autoencoder",
            "SOM",
            "BoltzmannMachine",
            "GAN",
        ]
        selected_topic = st.selectbox(
            "Topic",
            topic_options,
        )
        st.session_state.topic_filter = (
            None if selected_topic == "All" else selected_topic
        )

    with col_diff:
        # TODO: st.selectbox for difficulty filter
        difficulty_options = [
            "All",
            "beginner",
            "intermediate",
            "advanced",
        ]
        selected_difficulty = st.selectbox(
            "Difficulty",
            difficulty_options,
        )
        st.session_state.difficulty_filter = (
            None
            if selected_difficulty == "All"
            else selected_difficulty
        )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    # TODO: implement
    # 1. query = st.chat_input("Ask about a deep learning topic...")
    #
    # 2. On submit:
    #    a. Append user message to chat_history
    #    b. Display user message immediately (st.rerun or direct render)
    #    c. Build LangGraph input:
    #       {"messages": [HumanMessage(content=query)]}
    #    d. config = {"configurable": {"thread_id": st.session_state.thread_id}}
    #    e. result = graph.invoke(input, config=config)
    #    f. response = result["final_response"]
    #    g. Append assistant message with answer, sources, no_context_found flag
    #
    # STRETCH GOAL — streaming:
    # Replace graph.invoke with graph.stream() and use st.write_stream()
    # to display tokens as they arrive. Significant "wow factor" in Hour 3.

    query = st.chat_input("Ask about a deep learning topic...")

    if query:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query,
            }
        )

        graph_input = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state.topic_filter,
            "difficulty_filter": st.session_state.difficulty_filter,
        }

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:
            result = graph.invoke(
                graph_input,
                config=config,
            )

            response: AgentResponse | None = result.get("final_response")

            if response is None:
                raise RuntimeError(
                    "The RAG agent did not return a final response."
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "no_context_found": response.no_context_found,
                }
            )

        except Exception as exc:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"Error generating response: {exc}",
                    "sources": [],
                    "no_context_found": False,
                }
            )

        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()