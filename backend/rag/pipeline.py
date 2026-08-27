"""
Core RAG pipeline — deliberately kept independent of any UI framework.

Flow (Level 1):
    PDF path -> load & split into pages -> chunk into smaller pieces
    -> embed chunks -> store in FAISS -> build a retriever
    -> wrap retriever + LLM into a question-answering chain
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Config (Level 1 asks us to experiment with these two) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 5

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2-preview"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a document.
Answer the question using ONLY the following retrieved context. Do not use any
outside knowledge, even if you know the answer from elsewhere.

If the retrieved context does not contain enough information to answer the
question, say clearly that you don't know based on the uploaded document.
Do not guess or make up an answer.

Context:
{context}

Question: {question}

Answer:"""


def load_and_split_pdf(pdf_path: str):
    """Load a PDF into per-page Documents. PyPDFLoader auto-attaches
    page-number metadata to each Document — needed later for Level 4 citations."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def chunk_documents(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split page-level Documents into smaller chunks. Embeddings work best
    over focused text; a whole page can mix multiple ideas together.
    Overlap prevents cutting an idea cleanly in half at a chunk boundary.
    Metadata (including page) carries over onto every chunk automatically."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks):
    """Embed each chunk and store vectors in FAISS — lets us find chunks
    whose *meaning* is close to a question, not just shared keywords.
    In-memory, no DB needed for a small 10-page doc."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain(vectorstore, k: int = RETRIEVAL_K):
    """Wrap retriever + LLM into one runnable chain:
    question -> retriever -> {context, question} -> prompt -> LLM -> text"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_pipeline_from_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE,
                             chunk_overlap: int = CHUNK_OVERLAP, k: int = RETRIEVAL_K):
    """Runs the whole Level 1 pipeline end to end — the one function
    Gradio (and later FastAPI) needs to call."""
    pages = load_and_split_pdf(pdf_path)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    vectorstore = build_vectorstore(chunks)
    return build_qa_chain(vectorstore, k)