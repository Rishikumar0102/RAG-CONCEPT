import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------
# Load Environment Variables
# -------------------------

import os
from dotenv import load_dotenv

load_dotenv()

print("GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))
# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -------------------------
# Embedding Model
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-small",
    model_kwargs={"device": "cpu"}
)

# -------------------------
# Chroma Cloud
# -------------------------
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

vectorstore = Chroma(
    client=client,
    collection_name="python",
    embedding_function=embeddings,
)

# -------------------------------------------------
# Run this ONLY ONCE when creating the vector DB.
# Comment it after the first successful upload.
# -------------------------------------------------
UPLOAD_DATA = False

if UPLOAD_DATA:
    loader = PyPDFLoader("Python Programming.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)

    print(f"Uploaded {len(chunks)} chunks.")

# -------------------------
# Retriever
# -------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful Python tutor.

Answer ONLY using the provided context.

If the answer is not available in the context, reply exactly:

"Not found in the book."

Context:
{context}
"""
    ),
    (
        "user",
        "{question}"
    ),
])

# -------------------------
# Format Documents
# -------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------
# RAG Chain
# -------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)