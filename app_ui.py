import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    min-height: 100vh;
}

/* Header */
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.app-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e94560, #f5a623, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-header p {
    color: #aaaacc;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* Upload box */
.upload-box {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #e94560;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Chat container */
.chat-container {
    height: 460px;
    overflow-y: auto;
    padding: 1rem;
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* Bubbles */
.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(233,69,96,0.3);
    word-wrap: break-word;
}
.bubble-bot {
    align-self: flex-start;
    background: linear-gradient(135deg, #0f3460, #533483);
    color: #f0f0ff;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(83,52,131,0.3);
    word-wrap: break-word;
}
.bubble-label {
    font-size: 0.7rem;
    font-weight: 600;
    margin-bottom: 3px;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.bubble-user .bubble-label { text-align: right; color: #ffcccc; }
.bubble-bot .bubble-label { color: #aaddff; }

/* Empty state */
.empty-state {
    text-align: center;
    color: #aaaacc;
    padding: 3rem 1rem;
    font-size: 0.95rem;
}
.empty-state span { font-size: 3rem; display: block; margin-bottom: 0.5rem; }

/* Input bar */
.stTextInput > div > div > input {
    background: white !important;
    color: black !important;
    border: 2px solid rgba(233,69,96,0.4) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    font-family: 'Poppins', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e94560 !important;
    box-shadow: 0 0 0 2px rgba(233,69,96,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: #888 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #f5a623) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.4) !important;
}

/* Success/info boxes */
.stSuccess {
    background: rgba(0,212,170,0.1) !important;
    border: 1px solid #00d4aa !important;
    border-radius: 10px !important;
    color: #00d4aa !important;
}
.stInfo {
    background: rgba(245,166,35,0.1) !important;
    border: 1px solid #f5a623 !important;
    border-radius: 10px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] label { color: #aaaacc !important; }

/* Sidebar-like left panel label */
.panel-label {
    color: #f5a623;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

/* Scrollbar */
.chat-container::-webkit-scrollbar { width: 5px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: #e94560; border-radius: 10px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = "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"

@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        st.error("❌ This PDF has no readable text. Please upload a text-based PDF.")
        st.stop()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if not chunks:
        st.error("❌ Could not extract content from this PDF.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def get_answer(vectorstore, query, chat_history):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [{"role": "system", "content": "Answer questions based only on the PDF context provided. Be concise and helpful. If not found, say 'I could not find that in the document.'"}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.3
    )
    return response.choices[0].message.content

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Header
st.markdown("""
<div class="app-header">
    <h1>🤖 PDF Chatbot</h1>
    <p>Upload a PDF and chat with it instantly — powered by Groq AI</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="panel-label">📂 Upload PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("⚡ Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name
                st.session_state.vectorstore = process_pdf(file_path)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"📄 Size: {uploaded_file.size / 1024:.1f} KB")
        st.caption(f"💬 Messages: {len(st.session_state.chat_history) // 2}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("""
        <div style='color:#aaaacc; font-size:0.85rem; padding: 0.5rem 0;'>
        📌 Supports text-based PDFs<br>
        ⚡ Powered by Groq LLaMA3<br>
        💬 Chat with any document
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel-label">💬 Chat</div>', unsafe_allow_html=True)

    bubbles_html = '<div class="chat-container">'
    if not st.session_state.chat_history:
        bubbles_html += '''
        <div class="empty-state">
            <span>💬</span>
            Upload a PDF and start asking questions!
        </div>'''
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                bubbles_html += f'''
                <div class="bubble-user">
                    <div class="bubble-label">You</div>
                    {msg["content"]}
                </div>'''
            else:
                bubbles_html += f'''
                <div class="bubble-bot">
                    <div class="bubble-label">🤖 AI</div>
                    {msg["content"]}
                </div>'''
    bubbles_html += '</div>'
    st.markdown(bubbles_html, unsafe_allow_html=True)

    if st.session_state.vectorstore:
        query = st.text_input("", placeholder="Ask anything about your PDF...", label_visibility="collapsed")
        if st.button("Send ➤") and query.strip():
            with st.spinner("🤔 Thinking..."):
                answer = get_answer(st.session_state.vectorstore, query, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.markdown('<p style="color:#aaaacc; font-size:0.9rem;">👈 Upload a PDF first to start chatting</p>', unsafe_allow_html=True)
