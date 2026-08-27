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

# --- Level 5: condense a follow-up question into a standalone one ---
CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite
the follow-up question as a standalone question that includes any context needed to
understand it on its own (for example, resolve pronouns like "it" or phrases like
"the last one" using the history).

If the follow-up question is already standalone, return it unchanged. Return only the
rewritten question, nothing else.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""


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


def build_pipeline_from_pdf_v2(pdf_path: str, pages_needing_fix=None,
                                chunk_size: int = CHUNK_SIZE,
                                chunk_overlap: int = CHUNK_OVERLAP, k: int = RETRIEVAL_K):
    from backend.rag.extraction import load_pdf_with_layout_fix  # avoids circular import
    pages = load_pdf_with_layout_fix(pdf_path, pages_needing_fix=pages_needing_fix)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    vectorstore = build_vectorstore(chunks)
    return build_qa_chain(vectorstore, k)


def build_retriever(vectorstore, k: int = RETRIEVAL_K):
    """Level 4: expose the retriever on its own so the UI can inspect
    which chunks were actually retrieved (needed for citations),
    instead of only getting back the final answer text."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_answer_chain():
    """LLM-only chain: takes {"context", "question"} and streams the
    answer. Retrieval is handled separately via build_retriever(), so we
    can reuse the same retrieved docs both for the LLM's context and for
    building citations from their metadata."""
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    return prompt | llm | StrOutputParser()


def build_condense_chain():
    """Level 5: rewrites a follow-up question into a standalone one using
    conversation history, so retrieval can find the right chunks even
    when the question uses a pronoun or reference like "the last one"."""
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)
    return prompt | llm | StrOutputParser()


def format_chat_history(history):
    """Formats Gradio's message-format history into plain text for the
    condense-question prompt. Strips the Sources section so citation
    text doesn't clutter what the condense LLM sees, and skips the
    empty assistant placeholder used while an answer is still streaming.

    Gradio's `content` field is usually a string, but can come through as
    a list (e.g. for multimodal messages) depending on version/component
    state — so we normalize it to a string first.
    """
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        raw_content = turn["content"]

        if isinstance(raw_content, list):
            raw_content = "".join(
                part if isinstance(part, str) else str(part) for part in raw_content
            )

        content = raw_content.split("\n\n**Sources:**")[0]
        if content.strip():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def format_citations(docs):
    """Level 4: build a 'Sources' line from the page numbers of the
    chunks actually retrieved for this answer -- never the whole
    document by default. Uses page_label metadata (human-readable,
    1-indexed) that PyPDFLoader attaches to every chunk, and the
    actual source filename so citations are correct regardless of
    which PDF was uploaded."""
    if not docs:
        return ""

    # Get the actual filename from metadata instead of hardcoding it
    source_path = docs[0].metadata.get("source", "document")
    source_name = os.path.splitext(os.path.basename(source_path))[0]

    pages = sorted(set(int(d.metadata.get("page_label", d.metadata.get("page", 0))) for d in docs))
    if not pages:
        return ""

    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))

    parts = [str(a) if a == b else f"{a}\u2013{b}" for a, b in ranges]
    label = "pp." if len(pages) > 1 else "p."
    return f"**Sources:**\n- {source_name} \u2014 {label} " + ", ".join(parts)


def build_qa_components(pdf_path, pages_needing_fix=None, chunk_size: int = CHUNK_SIZE,
                         chunk_overlap: int = CHUNK_OVERLAP, k: int = RETRIEVAL_K):
    """Returns (retriever, answer_chain, condense_chain) so the UI can:
    retrieve once, use the docs for citations, stream the answer from
    that same retrieved context, and (Level 5) rewrite follow-up
    questions into standalone ones before retrieval even happens."""
    if pages_needing_fix:
        from backend.rag.extraction import load_pdf_with_layout_fix
        pages = load_pdf_with_layout_fix(pdf_path, pages_needing_fix=pages_needing_fix)
    else:
        pages = load_and_split_pdf(pdf_path)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    vectorstore = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore, k)
    answer_chain = build_answer_chain()
    condense_chain = build_condense_chain()
    return retriever, answer_chain, condense_chain