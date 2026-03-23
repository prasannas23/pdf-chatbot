import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
import tempfile

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot (FAST ⚡)")

# 🔥 CACHE: Load + process PDF only once
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# 🔥 CACHE: Load model only once
@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ PDF uploaded & processed!")

    # ⚡ Cached vector DB
    vectorstore = process_pdf(file_path)

    # ⚡ Cached LLM
    llm = load_llm()

    # Input
    query = st.text_input("💬 Ask a question:")
    ask = st.button("Ask")

    if ask and query:
        with st.spinner("Thinking... 🤖"):

            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            response = llm.invoke(
                f"""
                Answer ONLY from the context.
                If not found, say "Not found".

                Context:
                {context}

                Question:
                {query}
                """
            )

            answer = response.content if hasattr(response, "content") else str(response)

            if not answer.strip():
                answer = "⚠️ No response from model"

            st.markdown("### 💡 Answer:")
            st.write(answer)