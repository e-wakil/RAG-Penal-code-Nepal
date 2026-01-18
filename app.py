import streamlit as st
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import hashlib
import warnings

warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL = "llama-3.1-8b-instant"
DEFAULT_K = 6
TEMPERATURE = 0.0  # Zero temperature for strict legal accuracy

# ==============================
# CACHED RESOURCE LOADING (Optimized from your working code)
# ==============================
@st.cache_resource
def load_embeddings():
    """Load embeddings and metadata - optimized version"""
    try:
        emb = np.load("final_legal_embeddings.npy")
        with open("final_legal_laws_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        dim = emb.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb)
        
        return emb, meta, index
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def load_sbert():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ==============================
# RETRIEVAL LOGIC (Enhanced)
# ==============================
def search_law(query, sbert, index, metadata, k=DEFAULT_K):
    """Retrieve relevant legal sections - optimized from your working code"""
    q_emb = sbert.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    
    # Store retrieved items with their metadata
    retrieved_items = []
    seen_sections = set()
    
    for idx in indices[0]:
        if idx < len(metadata):
            item = metadata[idx]
            section_key = f"{item.get('section', '')}-{item.get('subsection', '')}"
            
            # Avoid duplicates
            if section_key not in seen_sections:
                seen_sections.add(section_key)
                retrieved_items.append(item)
    
    # Format for display
    chunks = []
    for item in retrieved_items:
        # Create citation
        citation_parts = []
        if item.get('chapter'):
            citation_parts.append(f"Chapter {item['chapter']}")
        if item.get('section'):
            citation_parts.append(f"Section {item['section']}")
        if item.get('subsection') and item['subsection'] not in [None, '', 'null']:
            citation_parts.append(f"Sub-section {item['subsection']}")
        
        citation = ", ".join(citation_parts)
        chunks.append(f"[{citation}] {item.get('text', '')}")
    
    # Store in session state for later use
    st.session_state.retrieved_items = retrieved_items
    st.session_state.llm_context = "\n\n".join(chunks)
    
    return "\n\n".join(chunks)

# ==============================
# LLM INTERACTION (Strict version from your working code)
# ==============================
def ask_llm(client, context, question):
    """Strict legal assistant - exact same as your working code"""
    prompt = f"""
You are an AI LEGAL ASSISTANT whose sole authority is the
National Penal Code of Nepal, 2017.

This system operates under a Retrieval-Augmented Generation (RAG) framework.
The provided Law Text is the ONLY source of truth.

======================
ABSOLUTE LEGAL RULES
======================

1. You MUST answer strictly and exclusively from the provided Law Text.
2. You MUST NOT rely on prior knowledge, general law principles, or assumptions.
3. You MUST NOT add, infer, simplify, reinterpret, or generalize the law.
4. If the Law Text does NOT explicitly contain the answer, you MUST respond with a refusal.
5. Partial answers are NOT allowed.
6. Every legal statement MUST be directly supported by the Law Text.
7. You MUST preserve all legal conditions, exceptions, and provisos.
8. You MUST maintain a formal, neutral, legal tone.
9. You MUST cite the exact Chapter, Section, and Subsection if available.
10. Hallucination of law is STRICTLY PROHIBITED.

======================
REFUSAL POLICY (MANDATORY)
======================

You MUST refuse to answer if:
- The relevant legal provision is absent
- The Law Text is incomplete
- The question exceeds the scope of the provided sections
- The question asks for punishment, procedure, or interpretation not stated

Refusal MUST use EXACT wording:

"The provided sections of the National Penal Code, 2017 do not mention this."

No alternative phrasing is permitted.

======================
AUTHORITATIVE LAW TEXT
======================

{context}

======================
USER QUESTION
======================

{question}

======================
RESPONSE FORMAT (STRICT)
======================

Answer:
<Precise, faithful legal explanation strictly grounded in the Law Text>

Source:
<Exact Chapter / Section / Subsection OR "Not specified in provided text">
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=350,
    )
    return response.choices[0].message.content

# ==============================
# STREAMLIT UI (Enhanced version)
# ==============================
def setup_page():
    """Configure Streamlit page with custom styling"""
    st.set_page_config(
        page_title="Nepal Penal Code 2017 — RAG Legal Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1e3a8a;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #1e3a8a;
    }
    .legal-answer {
        background-color: #f8f9fa;
        border-left: 4px solid #1e3a8a;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 5px;
        font-family: 'Georgia', serif;
    }
    .source-box {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #172554;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render enhanced sidebar"""
    with st.sidebar:
        st.markdown("## ⚙️ **Settings**")
        
        # Retrieval settings
        k_value = st.slider(
            "Number of legal sections to retrieve:",
            min_value=3,
            max_value=10,
            value=DEFAULT_K,
            help="Higher values provide more context but may be slower"
        )
        
        st.markdown("---")
        
        # Dataset info
        if 'metadata' in st.session_state and st.session_state.metadata:
            st.markdown("## 📊 **Dataset Info**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Provisions", len(st.session_state.metadata))
            with col2:
                chapters = len(set([item.get('chapter', '') for item in st.session_state.metadata]))
                st.metric("Chapters", chapters)
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("## 💡 **Sample Questions**")
        
        sample_questions = [
            "What is the definition of 'public servant'?",
            "What constitutes murder?",
            "What is the punishment for theft?",
            "Explain the right of private defense",
            "What are the types of punishment available?",
            "What is considered sedition?",
            "How is imprisonment for life computed?"
        ]
        
        for q in sample_questions:
            if st.button(f"• {q}", key=f"sample_{hashlib.md5(q.encode()).hexdigest()[:6]}"):
                st.session_state.user_query = q
                st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("""
        ## ℹ️ **About**
        
        **National Penal Code 2017 RAG System**
        
        This system answers questions **strictly** from the provided legal text using Retrieval-Augmented Generation.
        
        **Key Features:**
        - Zero hallucination policy
        - Exact legal citations
        - Strict refusal when information is absent
        
        *Always verify with official sources for critical legal matters.*
        """)

def format_citation(item):
    """Format a clean citation for display"""
    parts = []
    if item.get('chapter'):
        parts.append(f"Chapter {item['chapter']}")
    if item.get('section'):
        parts.append(f"Section {item['section']}")
    if item.get('subsection') and item['subsection'] not in [None, '', 'null']:
        parts.append(f"Sub-section {item['subsection']}")
    
    if parts:
        return ", ".join(parts)
    return "Citation not available"

def main():
    """Main application function"""
    setup_page()
    
    # Header
    st.markdown("""
    <div class="main-title">
        <h1>🇳🇵 Nepal Penal Code 2017 — RAG Legal Assistant</h1>
        <p>Ask legal questions. Get answers strictly from the National Penal Code, 2017.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""
    if 'retrieved_items' not in st.session_state:
        st.session_state.retrieved_items = []
    if 'llm_context' not in st.session_state:
        st.session_state.llm_context = ""
    
    # Load resources
    with st.spinner("🔄 Loading legal database..."):
        emb, metadata, index = load_embeddings()
        if emb is None or index is None:
            st.error("❌ Failed to load legal database. Check if 'final_legal_embeddings.npy' and 'final_legal_laws_metadata.json' exist.")
            st.stop()
        
        # Store in session state
        st.session_state.metadata = metadata
    
    # Load models
    sbert = load_sbert()
    client = load_llm()
    
    if not os.environ.get("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.stop()
    
    if client is None:
        st.error("❌ Failed to initialize Groq client")
        st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "**Enter your legal question:**",
            value=st.session_state.user_query,
            placeholder="e.g., What is the punishment for murder under Section 41?",
            help="Be specific about sections or legal concepts"
        )
        
    with col2:
        st.write("")  # Spacing
        st.write("")
        search_clicked = st.button("🔍 **Search Legal Sections**", type="primary", use_container_width=True)
    
    # Process query
    if search_clicked and query:
        st.session_state.user_query = query
        
        with st.spinner("🔍 Searching legal sections..."):
            # Retrieve relevant sections
            context = search_law(query, sbert, index, metadata, k=st.session_state.get('k_value', DEFAULT_K))
            
            # Display retrieved sections
            with st.expander(f"📚 Retrieved Legal Sections ({len(st.session_state.retrieved_items)} found)", expanded=True):
                if st.session_state.retrieved_items:
                    for item in st.session_state.retrieved_items:
                        with st.container():
                            citation = format_citation(item)
                            st.markdown(f"**{citation}**")
                            st.markdown(f"*{item.get('section_title', '')}*")
                            st.markdown(f"> {item.get('text', '')[:300]}..." if len(item.get('text', '')) > 300 else f"> {item.get('text', '')}")
                            st.markdown("---")
                else:
                    st.warning("No relevant legal sections found.")
        
        # Generate answer
        with st.spinner("🤔 Generating legal answer..."):
            answer = ask_llm(client, st.session_state.llm_context, query)
        
        # Display answer
        st.markdown("### 📝 **Legal Answer**")
        st.markdown('<div class="legal-answer">', unsafe_allow_html=True)
        
        # Parse answer for better display
        if "Answer:" in answer and "Source:" in answer:
            answer_text = answer.split("Answer:")[1].split("Source:")[0].strip()
            source_text = answer.split("Source:")[1].strip() if "Source:" in answer else ""
            
            st.markdown(answer_text)
            
            if source_text and "Not specified" not in source_text:
                st.markdown("---")
                st.markdown(f"**Source:** {source_text}")
        else:
            st.markdown(answer)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show retrieved citations
        if st.session_state.retrieved_items:
            st.markdown("### 📋 **Retrieved Citations**")
            cols = st.columns(3)
            for idx, item in enumerate(st.session_state.retrieved_items[:6]):
                with cols[idx % 3]:
                    citation = format_citation(item)
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>{citation}</strong><br>
                    <small>{item.get('section_title', '')[:50]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif search_clicked and not query:
        st.warning("⚠️ Please enter a legal question first.")
    
    # Quick stats at bottom
    st.markdown("---")
    st.markdown("### 📊 **System Information**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h3>Legal Provisions</h3>
        <h2>{:,}</h2>
        </div>
        """.format(len(metadata)), unsafe_allow_html=True)
    
    with col2:
        chapters = len(set([item.get('chapter', '') for item in metadata]))
        st.markdown(f"""
        <div class="metric-box">
        <h3>Chapters</h3>
        <h2>{chapters}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
        <h3>Embedding Model</h3>
        <h4>{EMBEDDING_MODEL}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
        <h3>LLM Model</h3>
        <h4>{LLM_MODEL}</h4>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
