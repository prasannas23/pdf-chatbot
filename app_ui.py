import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="PDF Chatbot FREE", layout="wide")
st.title("📄 PDF Chatbot (100% FREE 🆓)")

# 🔥 CACHE PDF PROCESSING
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # FREE embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# 🔥 CACHE MODEL (FIXED VERSION)
@st.cache_resource
def load_llm():
    pipe = pipeline(
        task="text2text-generation",   # ✅ supported for flan-t5
        model="google/flan-t5-base",  # ✅ stable model
        max_length=256
    )
    return HuggingFacePipeline(pipeline=pipe)

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("✅ PDF uploaded & processed!")

    vectorstore = process_pdf(file_path)
    llm = load_llm()

    query = st.text_input("💬 Ask a question:")
    ask = st.button("Ask")

    if ask and query:
        with st.spinner("Thinking... 🤖"):

            # 🔥 Reduce noise (important)
            docs = vectorstore.similarity_search(query, k=1)
            context = docs[0].page_content

            # 🔥 Strong prompt
            prompt = f"""
You are an intelligent assistant.

Answer the question ONLY using the context below.
Give a SHORT and DIRECT answer.
Do NOT repeat the context.
If answer is not found, say "Not found".

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)

            # 🔥 Clean output
            answer = str(response).strip()

            if context in answer:
                answer = answer.replace(context, "").strip()

            if not answer:
                answer = "⚠️ No response from model"

            st.markdown("### 💡 Answer:")
            st.write(answer)
