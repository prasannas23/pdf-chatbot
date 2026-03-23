import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Page config
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot (Online 🚀)")

# 🔐 Load API key
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add OPENAI_API_KEY in Streamlit secrets")
    st.stop()

# 🔥 Cache PDF processing
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# 🔥 Cache LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key
    )

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ PDF uploaded & processed!")

    vectorstore = process_pdf(file_path)
    llm = load_llm()

    # Input + Button
    query = st.text_input("💬 Ask a question:")
    ask = st.button("Ask")

    if ask and query:
        with st.spinner("Thinking... 🤖"):

            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            response = llm.invoke(
                f"""
                You are a helpful assistant.
                Answer ONLY from the given context.
                If not found, say "Not found in document".

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