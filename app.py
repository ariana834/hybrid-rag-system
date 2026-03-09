"""
streamlit run app.py
"""

import streamlit as st
from core.parser import DocumentParser
from core.pipeline import RAGPipeline, PipelineConfig, PipelineMode
from core.llm_service import LLMService
from config import OPENAI_API_KEY, OPENAI_MODEL

st.set_page_config(
    page_title="documind-ai",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background-color: #0e0e11; color: #e8e8e8; }
      [data-testid="stSidebar"] {
        background-color: #16161a;
        border-right: 1px solid #2a2a32;
    }
    [data-testid="stChatMessage"] {
        background-color: #1a1a20 !important;
        border: 1px solid #2a2a32;
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .stButton > button {
        background-color: #2a2a38;
        color: #e8e8e8;
        border: 1px solid #3a3a48;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #7c6af7;
        border-color: #7c6af7;
        color: white;
    }

    .source-card {
        background-color: #1a1a20;
        border: 1px solid #2a2a32;
        border-left: 3px solid #7c6af7;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 6px;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #a0a0b0;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 500;
        font-family: 'DM Mono', monospace;
    }
    .badge-mode    { background: #2a2a38; color: #7c6af7; border: 1px solid #3a3a48; }
    .badge-rerank  { background: #1a2a1a; color: #4caf7c; border: 1px solid #2a4a2a; }
    .badge-tokens  { background: #1a1a2a; color: #6a9af7; border: 1px solid #2a2a4a; }

    .app-title {
        font-weight: 600;
        font-size: 1.3rem;
        color: #e8e8e8;
        letter-spacing: -0.02em;
    }

    hr { border-color: #2a2a32 !important; }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0e0e11; }
    ::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        "messages":      [],
        "uploaded_docs": [],
        "mode":          "hybrid_rerank",
        "parser":        DocumentParser(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

MODE_MAP = {
    "semantic_only": PipelineMode.SEMANTIC_ONLY,
    "bm25_only":     PipelineMode.BM25_ONLY,
    "hybrid":        PipelineMode.HYBRID,
    "hybrid_rerank": PipelineMode.HYBRID_RERANK,
}

MODE_LABELS = {
    "semantic_only": "🔵 Semantic Only",
    "bm25_only":     "🟡 BM25 Only",
    "hybrid":        "🟠 Hybrid",
    "hybrid_rerank": "🟣 Hybrid + Rerank",
}

@st.cache_resource(show_spinner=False)
def load_pipeline(mode_key: str) -> RAGPipeline:
    return RAGPipeline(
        config=PipelineConfig(
            mode=MODE_MAP[mode_key],
            retrieval_top_k=20,
            final_top_k=5,
        )
    ).setup()


@st.cache_resource(show_spinner=False)
def load_llm() -> LLMService:
    return LLMService(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)

def render_sources(sources, mode, reranked, elapsed, tokens, cost):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} sources · {mode} · {elapsed:.2f}s"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<span class="badge badge-mode">{mode}</span>', unsafe_allow_html=True)
        with c2:
            if reranked:
                st.markdown('<span class="badge badge-rerank">reranked ✓</span>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<span class="badge badge-tokens">{tokens} tokens · ${cost:.5f}</span>', unsafe_allow_html=True)
        st.markdown("---")
        for s in sources:
            st.markdown(
                f'<div class="source-card">'
                f'[{s["citation_number"]}] &nbsp; score: {s["score"]:.4f}<br><br>'
                f'{s["text_excerpt"]}...'
                f'</div>',
                unsafe_allow_html=True,
            )

with st.sidebar:
    st.markdown('<div class="app-title">📄 documind-ai</div>', unsafe_allow_html=True)
    st.caption("Semantic · BM25 · Hybrid · Rerank")
    st.divider()

    # Retrieval mode
    st.markdown("**Retrieval Mode**")
    selected_mode = st.selectbox(
        label="mode",
        options=list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
        index=list(MODE_LABELS.keys()).index(st.session_state.mode),
        label_visibility="collapsed",
    )
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Upload
    st.markdown("**Upload Document**")
    uploaded_file = st.file_uploader(
        label="upload",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        already = any(d["filename"] == uploaded_file.name for d in st.session_state.uploaded_docs)
        if not already:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    pipeline = load_pipeline(st.session_state.mode)
                    result = pipeline.ingest(uploaded_file, parser=st.session_state.parser)
                    if result.success:
                        st.session_state.uploaded_docs.append({
                            "filename":  result.filename,
                            "chunks":    result.num_chunks,
                            "sentences": result.num_sentences,
                            "elapsed":   result.elapsed_seconds,
                        })
                        st.success(f"✓ {result.num_chunks} chunks · {result.elapsed_seconds:.1f}s")
                    else:
                        st.error(f"✗ {result.error}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Already uploaded.")

    st.divider()

    # Documents list
    if st.session_state.uploaded_docs:
        st.markdown("**Documents**")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"📄 **{doc['filename']}**")
            st.caption(f"{doc['chunks']} chunks · {doc['sentences']} sentences")
        st.divider()

    # Clear chat
    if st.session_state.messages:
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption(f"mode: `{st.session_state.mode}`")
    st.caption("documind-ai v1.0")

st.markdown("## Chat with your documents")
st.divider()

# Afișăm istoricul
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(
                sources=msg["sources"],
                mode=msg.get("mode", ""),
                reranked=msg.get("reranked", False),
                elapsed=msg.get("elapsed", 0),
                tokens=msg.get("tokens", 0),
                cost=msg.get("cost", 0.0),
            )

# Empty state
if not st.session_state.uploaded_docs:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px; color:#555;">
        <div style="font-size:3rem; margin-bottom:16px;">📄</div>
        <div style="font-size:1.1rem; color:#888; margin-bottom:8px;">No documents uploaded</div>
        <div style="font-size:0.85rem; color:#555;">Upload a PDF, DOCX, or TXT from the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask something about your documents..."):
    if not st.session_state.uploaded_docs:
        st.warning("⚠ Upload a document first!")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                pipeline = load_pipeline(st.session_state.mode)
                llm = load_llm()

                query_result = pipeline.query(prompt)

                if not query_result.chunks:
                    answer  = "I couldn't find relevant information in the uploaded documents."
                    sources = []
                    tokens  = 0
                    cost    = 0.0
                else:
                    llm_response = llm.answer(query_result)
                    answer  = llm_response.answer_with_citations
                    sources = [
                        {
                            "citation_number": s.citation_number,
                            "text_excerpt":    s.text_excerpt,
                            "score":           round(s.relevance_score, 4),
                        }
                        for s in llm_response.sources
                    ]
                    tokens = llm_response.total_tokens
                    cost   = llm_response.cost_usd

                st.markdown(answer)
                render_sources(
                    sources=sources,
                    mode=query_result.mode.value,
                    reranked=query_result.reranked,
                    elapsed=query_result.elapsed_seconds,
                    tokens=tokens,
                    cost=cost,
                )

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  answer,
                    "sources":  sources,
                    "mode":     query_result.mode.value,
                    "reranked": query_result.reranked,
                    "elapsed":  query_result.elapsed_seconds,
                    "tokens":   tokens,
                    "cost":     cost,
                })

            except Exception as e:
                st.error(f"Error: {e}")