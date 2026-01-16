import streamlit as st
import json
import faiss
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Static file paths
FAISS_FILE = "index/MSME_Handbook_FAISS.faiss"
IDS_FILE = "index/MSME_Handbook_ids.json"
CHUNKS_FILE = "index/MSME_Handbook_Chunks.json"


# ==================== CONVERSATION MANAGER ====================
class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.MAX_TOKENS = 500
        self.SESSION_TTL = timedelta(hours=1)

    def _approximate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 0.75)

    def _cleanup_expired(self):
        now = datetime.now()
        expired = [sid for sid, data in self.sessions.items()
                   if now - data["last_access"] > self.SESSION_TTL]
        for sid in expired:
            del self.sessions[sid]

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str):
        if len(self.sessions) > 100:
            self._cleanup_expired()

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_access": datetime.now(),
                "token_count": 0
            }

        session = self.sessions[session_id]
        session["last_access"] = datetime.now()

        new_tokens = self._approximate_tokens(user_msg + assistant_msg)

        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": assistant_msg})
        session["token_count"] += new_tokens

        while session["token_count"] > self.MAX_TOKENS and len(session["history"]) > 2:
            removed = session["history"].pop(0)
            removed_assistant = session["history"].pop(0)
            session["token_count"] -= self._approximate_tokens(
                removed["content"] + removed_assistant["content"]
            )

    def get_context(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return ""

        history = self.sessions[session_id]["history"]
        if not history:
            return ""

        return "Previous conversation:\n" + "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        ) + "\n\n"

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_stats(self, session_id: str):
        if session_id not in self.sessions:
            return {"exists": False, "message_count": 0, "token_count": 0}

        session = self.sessions[session_id]
        return {
            "exists": True,
            "message_count": len(session["history"]),
            "token_count": session["token_count"]
        }


# ==================== INITIALIZE COMPONENTS ====================
@st.cache_resource
def load_models():
    """Load LLM and embeddings (cached)"""
    # Try getting from OS environment, otherwise try Streamlit secrets
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )

    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    return llm, embedder


@st.cache_resource
def load_index_and_chunks():
    """Load FAISS index and chunks (cached)"""
    try:
        # Load FAISS index
        index = faiss.read_index(FAISS_FILE)

        # Load chunk IDs
        with open(IDS_FILE, "r") as f:
            chunk_ids = json.load(f)

        # Load chunks data
        with open(CHUNKS_FILE, "r") as f:
            chunks = json.load(f)

        return index, chunk_ids, chunks
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e.filename}")
        st.info("Please ensure these files exist:\n- index.faiss\n- index_ids.json\n- index_chunks.json")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        st.stop()


@st.cache_resource
def get_conversation_manager():
    """Get conversation manager (singleton)"""
    return ConversationManager()


# ==================== RETRIEVAL FUNCTION ====================
def retrieve_documents(query: str, embedder, index, chunk_ids, chunks, k=3):
    """Retrieve relevant documents"""
    # Embed query
    query_embedding = embedder.embed_query(query)
    vec = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(vec)

    # Search
    scores, ids = index.search(vec, k)

    # Build results
    results = []
    for idx, i in enumerate(ids[0]):
        if i != -1 and i < len(chunk_ids):
            chunk_id = chunk_ids[i]
            chunk = chunks[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": float(scores[0][idx]),
                "title": chunk["title"],
                "summary": chunk["summary"],
                "evidence": chunk["propositions"]
            })

    return results


# ==================== ANSWER GENERATION ====================
def generate_answer(query: str, context: str, retrieved_docs, llm):
    """Generate answer using LLM"""
    if not retrieved_docs:
        return "I couldn't find any relevant information in the knowledge base."

    # Build evidence
    evidence_text = "\n\n".join(
        f"SOURCE: {doc['chunk_id']}\n" + "\n".join(f"- {p}" for p in doc["evidence"][:5])
        for doc in retrieved_docs[:3]
    )

    # Create prompt
    CONV_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "Answer using conversation context and evidence. Cite sources. Be concise."),
        ("user", "{context}Q: {query}\n\nEvidence:\n{evidence}")
    ])

    # Generate answer
    runnable = CONV_PROMPT | llm | StrOutputParser()
    answer = runnable.invoke({
        "context": context,
        "query": query,
        "evidence": evidence_text
    })

    return answer


# ==================== LOAD ALL COMPONENTS ====================
llm, embedder = load_models()
index, chunk_ids, chunks = load_index_and_chunks()
conv_manager = get_conversation_manager()

# ==================== SESSION STATE ====================
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🤖 RAG Chat Assistant")
    st.markdown("---")

    # Session Info
    st.subheader("📊 Session Info")
    stats = conv_manager.get_stats(st.session_state.session_id)
    st.metric("Messages", stats["message_count"])
    st.metric("Context Tokens", stats["token_count"])
    st.metric("Token Limit", 500)

    # Progress bar
    token_percentage = min(stats["token_count"] / 500, 1.0)
    st.progress(token_percentage)

    st.markdown("---")

    # Knowledge Base Status
    st.subheader("📁 Knowledge Base")
    st.success(f"✅ {len(chunks)} chunks loaded")
    st.info(f"📊 {index.ntotal} vectors indexed")

    st.markdown("---")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            conv_manager.clear(st.session_state.session_id)
            st.rerun()

    with col2:
        if st.button("🔄 New", use_container_width=True):
            import uuid

            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Made with Streamlit & Google AI")

# ==================== MAIN CONTENT ====================
st.title("💬 Conversational RAG Assistant")
st.markdown("Ask questions about the knowledge base. I'll remember our conversation!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", f"{message['metadata'].get('score', 0):.3f}")
                with col2:
                    st.metric("Source", message['metadata'].get('title', 'N/A'))

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get conversation context
                context = conv_manager.get_context(st.session_state.session_id)

                # Retrieve documents
                retrieved_docs = retrieve_documents(prompt, embedder, index, chunk_ids, chunks)

                # Generate answer
                answer = generate_answer(prompt, context, retrieved_docs, llm)

                # Display answer
                st.markdown(answer)

                # Show metadata
                if retrieved_docs:
                    with st.expander("📊 Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Score", f"{retrieved_docs[0]['score']:.3f}")
                        with col2:
                            st.metric("Source", retrieved_docs[0]['title'])

                    metadata = {
                        "score": retrieved_docs[0]['score'],
                        "title": retrieved_docs[0]['title']
                    }
                else:
                    metadata = {}

                # Update conversation history
                conv_manager.add_exchange(st.session_state.session_id, prompt, answer)

                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 Session-based")
with col2:
    st.caption("🧠 500-token context")
with col3:
    st.caption("⚡ Optimized retrieval")