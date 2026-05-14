"""
NeoStats AI Chatbot — Multi-Domain Intelligent Assistant
Main Streamlit Application Entry Point

Features:
  ✅ Multi-provider LLM (OpenAI / Groq / Gemini) incl. llama-3.3-70b-versatile
  ✅ RAG with FAISS vector store
  ✅ Live web search (Serper / Tavily / Brave)
  ✅ Concise / Detailed response modes
  ✅ Streaming responses
  ✅ Dynamic domain theming (Healthcare / Legal / Finance / Education / General)
  ✅ Sidebar open/close toggle button always visible
  ✅ Session analytics dashboard
  ✅ Chat history export
"""

import logging
import time
import streamlit as st

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="NeoStats AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import llm_config, rag_config, web_search_config, app_config
from models.llm import get_llm
from models.embeddings import EmbeddingModel, FAISSVectorStore
from utils.document_processor import process_uploaded_file, build_rag_context
from utils.web_search import web_search, format_search_results, should_search_web
from utils.prompt_utils import get_system_prompt, format_chat_history, build_source_footer
from utils.analytics import SessionAnalytics, export_chat_history


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONFIG — All per-domain theming and content lives here
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_CONFIG = {
    "Healthcare": {
        "icon": "🩺", "label": "MediBot Pro",
        "subtitle": "Healthcare Intelligence · RAG-Powered · Live Web Search",
        "accent": "#0ea5e9", "accent_rgb": "14,165,233",
        "accent2": "#06b6d4", "accent_light": "#38bdf8",
        "bg": "linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#0a1628 100%)",
        "sb_bg": "linear-gradient(180deg,#0d1b2a 0%,#091220 100%)",
        "sb_border": "rgba(56,189,248,0.15)",
        "placeholder": "Ask a healthcare question…",
        "prompts": [("💊","What are the side effects of metformin?"),
                    ("🫀","Explain hypertension treatment guidelines"),
                    ("🧬","Latest research on GLP-1 receptor agonists?")],
        "doc_hint": "Upload clinical guidelines, drug references or research PDFs.",
        "bot_avatar": "🩺",
    },
    "Legal": {
        "icon": "⚖️", "label": "LexBot",
        "subtitle": "Legal Information Assistant · Document Analysis · Case Research",
        "accent": "#8b5cf6", "accent_rgb": "139,92,246",
        "accent2": "#7c3aed", "accent_light": "#a78bfa",
        "bg": "linear-gradient(135deg,#0d0a1a 0%,#130d2a 50%,#0f0a20 100%)",
        "sb_bg": "linear-gradient(180deg,#130d2a 0%,#0a0812 100%)",
        "sb_border": "rgba(167,139,250,0.15)",
        "placeholder": "Ask a legal information question…",
        "prompts": [("📜","What is the difference between civil and criminal law?"),
                    ("🏛️","Explain contract formation requirements"),
                    ("🔍","What are my rights during a police stop?")],
        "doc_hint": "Upload contracts, legal briefs, statutes or case documents.",
        "bot_avatar": "⚖️",
    },
    "Finance": {
        "icon": "📈", "label": "FinBot",
        "subtitle": "Financial Intelligence · Market Research · Portfolio Insights",
        "accent": "#10b981", "accent_rgb": "16,185,129",
        "accent2": "#059669", "accent_light": "#34d399",
        "bg": "linear-gradient(135deg,#0a1a0f 0%,#0d2a18 50%,#0a1a10 100%)",
        "sb_bg": "linear-gradient(180deg,#0d2a18 0%,#071209 100%)",
        "sb_border": "rgba(52,211,153,0.15)",
        "placeholder": "Ask a finance or investment question…",
        "prompts": [("💰","Explain dollar-cost averaging strategy"),
                    ("📊","What is the difference between stocks and bonds?"),
                    ("🏦","Latest Fed interest rate decisions?")],
        "doc_hint": "Upload annual reports, financial statements or market research.",
        "bot_avatar": "📈",
    },
    "Education": {
        "icon": "🎓", "label": "EduBot",
        "subtitle": "Learning Assistant · Concept Explanation · Study Support",
        "accent": "#f59e0b", "accent_rgb": "245,158,11",
        "accent2": "#d97706", "accent_light": "#fbbf24",
        "bg": "linear-gradient(135deg,#1a150a 0%,#2a1e0d 50%,#1a1508 100%)",
        "sb_bg": "linear-gradient(180deg,#2a1e0d 0%,#120e04 100%)",
        "sb_border": "rgba(251,191,36,0.15)",
        "placeholder": "Ask anything — I'll explain it clearly…",
        "prompts": [("🧮","Explain calculus derivatives with examples"),
                    ("🧬","How does DNA replication work?"),
                    ("🌍","What caused World War I?")],
        "doc_hint": "Upload textbooks, lecture notes, syllabi or study materials.",
        "bot_avatar": "🎓",
    },
    "General": {
        "icon": "🤖", "label": "NeoBot",
        "subtitle": "General Purpose AI Assistant · Powered by NeoStats",
        "accent": "#ec4899", "accent_rgb": "236,72,153",
        "accent2": "#db2777", "accent_light": "#f472b6",
        "bg": "linear-gradient(135deg,#1a0a14 0%,#2a0d1e 50%,#1a0a14 100%)",
        "sb_bg": "linear-gradient(180deg,#2a0d1e 0%,#12040e 100%)",
        "sb_border": "rgba(244,114,182,0.15)",
        "placeholder": "Ask me anything…",
        "prompts": [("💡","Explain quantum computing in simple terms"),
                    ("✍️","Help me write a professional email"),
                    ("🔍","What's happening in AI research lately?")],
        "doc_hint": "Upload any documents you'd like to chat about.",
        "bot_avatar": "🤖",
    },
}


def get_domain() -> str:
    return st.session_state.get("domain", "Healthcare")


def dc() -> dict:
    return DOMAIN_CONFIG[get_domain()]


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    d = dc()
    a, ar, a2, al = d["accent"], d["accent_rgb"], d["accent2"], d["accent_light"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

    html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;}}
    .stApp{{background:{d["bg"]};color:#e2e8f0;}}

    /* ── Sidebar always-visible toggle (collapsed state) ── */
    [data-testid="collapsedControl"]{{
        display:flex!important;visibility:visible!important;opacity:1!important;
        background:rgba({ar},0.18)!important;
        border:1px solid rgba({ar},0.45)!important;
        border-radius:8px!important;color:{al}!important;
    }}
    [data-testid="collapsedControl"]:hover{{
        background:rgba({ar},0.28)!important;
    }}
    [data-testid="collapsedControl"] svg{{fill:{al}!important;stroke:{al}!important;}}

    /* ── Sidebar ── */
    [data-testid="stSidebar"]{{
        background:{d["sb_bg"]};
        border-right:1px solid {d["sb_border"]};
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stToggle label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3{{color:#e2e8f0!important;}}

    /* ── Header ── */
    .bot-header{{
        background:rgba({ar},0.10);border:1px solid rgba({ar},0.25);
        border-radius:16px;padding:20px 28px;margin-bottom:18px;
        display:flex;align-items:center;gap:18px;backdrop-filter:blur(10px);
    }}
    .bot-header h1{{
        font-family:'Space Grotesk',sans-serif;font-size:1.85rem;font-weight:700;
        background:linear-gradient(135deg,{al},{a},{a2});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;
    }}
    .bot-header p{{color:#94a3b8;font-size:0.86rem;margin:4px 0 0;}}

    /* ── Chat ── */
    .chat-message{{
        display:flex;gap:14px;padding:14px 0;
        border-bottom:1px solid rgba(255,255,255,0.04);
        animation:fadeIn 0.3s ease;
    }}
    @keyframes fadeIn{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}
    .chat-avatar{{
        width:36px;height:36px;border-radius:9px;
        display:flex;align-items:center;justify-content:center;
        font-size:17px;flex-shrink:0;margin-top:2px;
    }}
    .user-avatar{{background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);}}
    .bot-avatar {{background:rgba({ar},0.2);border:1px solid rgba({ar},0.4);}}
    .chat-content{{flex:1;line-height:1.7;font-size:0.94rem;color:#e2e8f0;}}
    .user-text{{color:#c7d2fe;}}

    /* ── Badges ── */
    .badge{{display:inline-block;padding:2px 10px;border-radius:20px;
            font-size:0.71rem;font-weight:600;margin-right:6px;letter-spacing:0.4px;}}
    .badge-domain{{background:rgba({ar},0.15);color:{al};border:1px solid rgba({ar},0.35);font-weight:700;}}
    .badge-rag   {{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);}}
    .badge-web   {{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);}}
    .badge-mode  {{background:rgba({ar},0.12);color:{al};border:1px solid rgba({ar},0.3);}}

    /* ── Metric cards ── */
    .metric-card{{background:rgba({ar},0.06);border:1px solid rgba({ar},0.15);
                  border-radius:12px;padding:14px;text-align:center;}}
    .metric-value{{font-family:'Space Grotesk',sans-serif;font-size:1.75rem;font-weight:700;color:{al};}}
    .metric-label{{font-size:0.76rem;color:#64748b;margin-top:4px;}}

    /* ── Input ── */
    .stChatInput>div{{border-radius:14px!important;}}
    .stChatInput textarea{{
        background:rgba(15,23,42,0.8)!important;
        border:1px solid rgba({ar},0.25)!important;
        color:#e2e8f0!important;font-family:'DM Sans',sans-serif!important;
        border-radius:14px!important;
    }}
    .stChatInput textarea:focus{{
        border-color:rgba({ar},0.55)!important;
        box-shadow:0 0 0 3px rgba({ar},0.1)!important;
    }}

    /* ── Status cards ── */
    .status-card{{background:rgba({ar},0.06);border:1px solid rgba({ar},0.2);
                  border-radius:10px;padding:10px 14px;margin:6px 0;font-size:0.83rem;}}
    .s-online {{border-left:3px solid #34d399;}}
    .s-warning{{border-left:3px solid #fbbf24;}}
    .s-offline{{border-left:3px solid #f87171;}}

    /* ── Buttons ── */
    .stButton>button{{
        background:linear-gradient(135deg,{a},{a2});color:white;border:none;
        border-radius:8px;font-family:'DM Sans',sans-serif;font-weight:600;
        transition:all 0.2s ease;
    }}
    .stButton>button:hover{{transform:translateY(-1px);box-shadow:0 4px 16px rgba({ar},0.35);}}

    /* ── Section titles ── */
    .sec-title{{font-family:'Space Grotesk',sans-serif;font-size:0.7rem;font-weight:600;
               letter-spacing:1.4px;text-transform:uppercase;color:#475569;margin:18px 0 6px;}}

    /* ── Welcome prompt chips ── */
    .prompt-chip{{background:rgba({ar},0.08);border:1px solid rgba({ar},0.2);
                  border-radius:8px;padding:9px 15px;font-size:0.81rem;color:#94a3b8;
                  display:inline-block;margin:5px;}}

    #MainMenu,footer{{visibility:hidden;}}
    .block-container{{padding-top:1rem;}}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "chat_history": [],
        "analytics": SessionAnalytics(),
        "vector_store": None,
        "llm_instance": None,
        "current_provider": None,
        "current_model": None,
        "uploaded_docs": [],
        "rag_enabled": True,
        "web_search_enabled": False,
        "response_mode": "Detailed",
        "domain": "Healthcare",
        "show_sources": True,
        "temperature": 0.4,
        "show_analytics": False,
        "serper_key": "",
        "tavily_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    return EmbeddingModel(model_name)


def get_or_create_vector_store():
    if st.session_state.vector_store is None:
        try:
            em = get_embedding_model(rag_config.embedding_model)
            vs = FAISSVectorStore(em, store_path=rag_config.vector_store_path)
            vs.load()
            st.session_state.vector_store = vs
        except Exception as e:
            logger.error(f"Vector store init failed: {e}")
    return st.session_state.vector_store


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    d = dc()
    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div style="text-align:center;padding:14px 0 6px;">
            <div style="font-size:2rem;">{d['icon']}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;
                        font-weight:700;color:{d['accent_light']};margin-top:3px;">{d['label']}</div>
            <div style="font-size:0.68rem;color:#475569;margin-top:2px;letter-spacing:1px;">
                POWERED BY NEOSTATS AI
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Domain selector ───────────────────────────────────────────────
        st.markdown('<div class="sec-title">🌐 Domain / Use Case</div>', unsafe_allow_html=True)
        domain_list = list(DOMAIN_CONFIG.keys())
        prev = st.session_state.domain
        chosen = st.selectbox(
            "Active Domain",
            options=domain_list,
            format_func=lambda x: f"{DOMAIN_CONFIG[x]['icon']}  {x}",
            index=domain_list.index(prev),
            key="domain_select",
        )
        if chosen != prev:
            st.session_state.domain = chosen
            st.session_state.chat_history = []   # fresh chat when domain changes
            st.rerun()

        st.divider()

        # ── LLM ───────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🤖 Language Model</div>', unsafe_allow_html=True)
        provider = st.selectbox("Provider", ["OpenAI", "Groq", "Gemini"], index=1,
                                key="provider_select")
        model_map = {
            "OpenAI": llm_config.openai_models,
            "Groq":   llm_config.groq_models,
            "Gemini": llm_config.gemini_models,
        }
        model = st.selectbox("Model", model_map[provider], key="model_select")

        env_map = {"OpenAI": ("OPENAI_API_KEY", llm_config.openai_api_key),
                   "Groq":   ("GROQ_API_KEY",   llm_config.groq_api_key),
                   "Gemini": ("GEMINI_API_KEY",  llm_config.gemini_api_key)}
        env_var, stored_key = env_map[provider]
        api_key = st.text_input(f"{provider} API Key", value=stored_key, type="password",
                                placeholder=f"Enter {provider} key…",
                                help=f"Or set env var {env_var}")

        if st.button("⚡ Connect Model", use_container_width=True):
            if not api_key:
                st.error("Please provide an API key.")
            else:
                with st.spinner("Connecting…"):
                    try:
                        max_tok = (app_config.concise_max_tokens
                                   if st.session_state.response_mode == "Concise"
                                   else app_config.detailed_max_tokens)
                        llm = get_llm(provider=provider, model=model,
                                      temperature=st.session_state.temperature,
                                      max_tokens=max_tok, api_key=api_key)
                        st.session_state.llm_instance = llm
                        st.session_state.current_provider = provider
                        st.session_state.current_model = model
                        st.success(f"✅ {provider} / {model}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        if st.session_state.llm_instance:
            st.markdown(f"""<div class="status-card s-online">
                ✅ <b>{st.session_state.current_provider}</b> — {st.session_state.current_model}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card s-offline">❌ No model connected</div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Response mode ─────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📝 Response Mode</div>', unsafe_allow_html=True)
        st.session_state.response_mode = st.radio(
            "Mode", ["Concise", "Detailed"], index=1, horizontal=True,
            help="Concise: ≤3 sentences · Detailed: comprehensive with structure")
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.0,
            float(st.session_state.get("temperature", 0.4)), 0.05,
            help="Higher = more creative")

        st.divider()

        # ── RAG ───────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📚 Knowledge Base (RAG)</div>', unsafe_allow_html=True)
        st.session_state.rag_enabled = st.toggle("Enable RAG",
                                                  value=st.session_state.rag_enabled)
        if st.session_state.rag_enabled:
            st.caption(d["doc_hint"])
            uploaded = st.file_uploader("Upload Documents",
                                        type=["pdf","txt","md","docx","csv","xlsx","xls"],
                                        accept_multiple_files=True)
            if uploaded:
                vs = get_or_create_vector_store()
                if vs:
                    for f in uploaded:
                        if f.name not in st.session_state.uploaded_docs:
                            with st.spinner(f"Indexing {f.name}…"):
                                res = process_uploaded_file(
                                    file_bytes=f.read(), filename=f.name, vector_store=vs,
                                    chunk_size=rag_config.chunk_size,
                                    chunk_overlap=rag_config.chunk_overlap)
                            if res["success"]:
                                st.session_state.uploaded_docs.append(f.name)
                                st.success(f"✅ {f.name} — {res['chunks_added']} chunks")
                            else:
                                st.error(f"❌ {f.name}: {res['error']}")
                else:
                    st.warning("⚠️ Install sentence-transformers & faiss-cpu for RAG.")

            vs = st.session_state.vector_store
            if vs and vs.total_chunks > 0:
                st.markdown(f"""<div class="status-card s-online">
                    📂 <b>{vs.total_chunks}</b> chunks · <b>{len(vs.sources)}</b> doc(s)
                    </div>""", unsafe_allow_html=True)
                if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                    vs.clear(); st.session_state.uploaded_docs = []; st.rerun()

        st.divider()

        # ── Web search ────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🌐 Live Web Search</div>', unsafe_allow_html=True)
        st.session_state.web_search_enabled = st.toggle(
            "Enable Web Search", value=st.session_state.web_search_enabled)
        if st.session_state.web_search_enabled:
            st.session_state.serper_key = st.text_input(
                "Serper API Key", value=st.session_state.serper_key,
                type="password", placeholder="Free at serper.dev…")
            st.session_state.tavily_key = st.text_input(
                "Tavily API Key", value=st.session_state.tavily_key,
                type="password", placeholder="Free at tavily.com…")
            if st.session_state.serper_key or st.session_state.tavily_key:
                st.markdown('<div class="status-card s-online">🌐 Web search active</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card s-warning">⚠️ Add a search API key</div>',
                            unsafe_allow_html=True)

        st.divider()

        # ── Options ───────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">⚙️ Options</div>', unsafe_allow_html=True)
        st.session_state.show_sources = st.toggle("Show Sources",
                                                   value=st.session_state.show_sources)
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.analytics = SessionAnalytics()
                st.rerun()
        with c2:
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.show_analytics = not st.session_state.show_analytics
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def render_analytics():
    d = dc()
    s = st.session_state.analytics.summary()
    st.markdown(f"### 📊 Session Analytics — {d['icon']} {d['label']}")
    cols = st.columns(5)
    items = [("💬", str(s["total_messages"]), "Messages"),
             ("⚡", f"{s['avg_response_time_s']}s", "Avg Response"),
             ("📚", str(s["rag_hits"]), "RAG Hits"),
             ("🌐", str(s["web_searches"]), "Web Searches"),
             ("⏱️", f"{s['session_duration_mins']}m", "Session Time")]
    for col, (icon, val, label) in zip(cols, items):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:1.3rem">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
    if s["providers_used"]:
        st.markdown("**Provider usage:**")
        for p, c in s["providers_used"].items():
            st.progress(c / max(s["total_messages"], 1), text=f"{p}: {c}")
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(user_message: str):
    llm = st.session_state.llm_instance
    if not llm:
        return ("⚠️ No model connected. Open the sidebar (**☰** top-left) and click "
                "**⚡ Connect Model**."), False, False

    rag_chunks, web_results = [], []
    rag_context = web_context = ""

    # RAG
    if st.session_state.rag_enabled:
        vs = st.session_state.vector_store
        if vs and vs.total_chunks > 0:
            try:
                rag_chunks = vs.search(user_message, top_k=rag_config.top_k_results,
                                       threshold=rag_config.similarity_threshold)
                if rag_chunks:
                    rag_context = build_rag_context(rag_chunks)
            except Exception as e:
                logger.error(f"RAG error: {e}")

    # Web search
    if st.session_state.web_search_enabled:
        if (st.session_state.serper_key or st.session_state.tavily_key) \
                and should_search_web(user_message):
            try:
                web_results = web_search(
                    query=user_message,
                    serper_key=st.session_state.serper_key,
                    tavily_key=st.session_state.tavily_key,
                    max_results=web_search_config.max_results,
                    timeout=web_search_config.search_timeout,
                )
                if web_results:
                    web_context = format_search_results(web_results)
            except Exception as e:
                logger.error(f"Web search error: {e}")

    system_prompt = get_system_prompt(
        domain=st.session_state.domain,
        rag_context=rag_context,
        web_context=web_context,
        response_mode=st.session_state.response_mode,
    )
    messages = format_chat_history(
        st.session_state.chat_history, max_turns=app_config.max_history_turns
    ) + [{"role": "user", "content": user_message}]

    # Stream
    response_text = ""
    placeholder = st.empty()
    try:
        llm.max_tokens = (app_config.concise_max_tokens
                          if st.session_state.response_mode == "Concise"
                          else app_config.detailed_max_tokens)
        for token in llm.stream(messages, system_prompt=system_prompt):
            response_text += token
            placeholder.markdown(
                f'<div class="chat-content">{response_text}▌</div>',
                unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Stream error: {e}")
        try:
            response_text = llm.chat(messages, system_prompt=system_prompt)
        except Exception as e2:
            response_text = f"❌ Error: {e2}"
    placeholder.empty()

    footer = build_source_footer(rag_chunks, web_results, st.session_state.show_sources)
    return (response_text + footer,
            bool(rag_chunks),
            bool(web_results))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    inject_css()
    render_sidebar()

    d = dc()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="bot-header">
        <div style="font-size:2.3rem">{d['icon']}</div>
        <div>
            <h1>{d['label']}</h1>
            <p>{d['subtitle']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status bar ────────────────────────────────────────────────────────
    rag_active = (st.session_state.rag_enabled
                  and st.session_state.vector_store
                  and st.session_state.vector_store.total_chunks > 0)

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.2, 3.5])
    with c1:
        st.markdown(f'<span class="badge badge-domain">{d["icon"]} {get_domain()}</span>',
                    unsafe_allow_html=True)
    with c2:
        rag_lbl = "✅ RAG Active" if rag_active else "⭕ RAG Off"
        st.markdown(f'<span class="badge badge-rag">{rag_lbl}</span>', unsafe_allow_html=True)
    with c3:
        web_lbl = "✅ Web Search" if st.session_state.web_search_enabled else "⭕ No Web"
        st.markdown(f'<span class="badge badge-web">{web_lbl}</span>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<span class="badge badge-mode">📝 {st.session_state.response_mode}</span>',
                    unsafe_allow_html=True)
    with c5:
        if st.session_state.chat_history:
            txt = export_chat_history(st.session_state.chat_history, format="txt")
            st.download_button("📥 Export Chat", data=txt,
                               file_name=f"{get_domain().lower()}_chat.txt",
                               mime="text/plain")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analytics ─────────────────────────────────────────────────────────
    if st.session_state.show_analytics:
        render_analytics()

    # ── Chat history ──────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""<div class="chat-message">
                <div class="chat-avatar user-avatar">👤</div>
                <div class="chat-content user-text">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-message">
                <div class="chat-avatar bot-avatar">{d['bot_avatar']}</div>
                <div class="chat-content">""", unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Welcome screen ────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        chips = "".join(
            f'<div class="prompt-chip">{icon}&nbsp;&nbsp;"{txt}"</div>'
            for icon, txt in d["prompts"]
        )
        st.markdown(f"""
        <div style="text-align:center;padding:36px 20px;opacity:0.88;">
            <div style="font-size:3.2rem;margin-bottom:10px">{d['icon']}</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;color:#94a3b8;">
                Welcome to <span style="color:{d['accent_light']};font-weight:700;">{d['label']}</span>
            </div>
            <div style="font-size:0.83rem;color:#475569;margin-top:8px;
                        max-width:480px;margin-left:auto;margin-right:auto;">
                {d['subtitle']}<br><br>
                Connect a model from the sidebar — use the
                <b style="color:{d['accent_light']};">☰</b> button at top-left if it's closed.
            </div>
            <div style="margin-top:22px;display:flex;flex-wrap:wrap;justify-content:center;">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────
    if prompt := st.chat_input(d["placeholder"]):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        st.markdown(f"""<div class="chat-message">
            <div class="chat-avatar user-avatar">👤</div>
            <div class="chat-content user-text">{prompt}</div>
            </div>""", unsafe_allow_html=True)

        with st.chat_message("assistant", avatar=d["bot_avatar"]):
            t0 = time.time()
            status = st.empty()
            if st.session_state.rag_enabled:
                status.markdown("🔍 *Searching knowledge base…*")
            if st.session_state.web_search_enabled:
                status.markdown("🌐 *Running web search…*")
            status.markdown("🧠 *Generating response…*")

            response, rag_used, web_used = generate_response(prompt)
            status.empty()
            elapsed = time.time() - t0

            st.markdown(response)
            st.markdown(
                f'<div style="font-size:0.71rem;color:#334155;margin-top:6px;">'
                f'⏱ {elapsed:.2f}s'
                f'{" · 📚 RAG" if rag_used else ""}'
                f'{" · 🌐 Web" if web_used else ""}'
                f' · {st.session_state.response_mode} · {d["icon"]} {get_domain()}'
                f'</div>',
                unsafe_allow_html=True)

        st.session_state.analytics.log_message(
            query=prompt, response_time=elapsed,
            provider=st.session_state.current_provider or "unknown",
            rag_used=rag_used, web_used=web_used,
            response_mode=st.session_state.response_mode)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
