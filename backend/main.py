"""
FastAPI backend — Level 6.

Reuses the exact same RAG core from backend/rag/pipeline.py.
Stateless per-request except for the single currently-loaded PDF,
which is kept in memory as a global (fine for a single-user demo).
"""

import os
import shutil
import time
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.rag.pipeline import (
    build_qa_components,
    format_docs,
    format_citations,
    format_chat_history,
)

load_dotenv()

app = FastAPI(title="LSPP RAG Chatbot API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Same layout-fix page as the Gradio app
PAGES_NEEDING_LAYOUT_FIX = [2]

# Holds (retriever, answer_chain, condense_chain) for the currently loaded PDF
qa_components = None


class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[ChatTurn] = []


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_components

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        qa_components = build_qa_components(dest_path, pages_needing_fix=PAGES_NEEDING_LAYOUT_FIX)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "ok", "filename": file.filename}


@app.post("/ask")
async def ask_question(req: AskRequest):
    if qa_components is None:
        return StreamingResponse(iter(["Please upload a PDF first."]), media_type="text/plain")

    retriever, answer_chain, condense_chain = qa_components

    history_dicts = [{"role": turn.role, "content": turn.content} for turn in req.history]
    chat_history_text = format_chat_history(history_dicts)

    if chat_history_text.strip():
        standalone_question = condense_chain.invoke(
            {"chat_history": chat_history_text, "question": req.question}
        )
    else:
        standalone_question = req.question

    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)
    
    def generate():
        for chunk in answer_chain.stream({"context": context, "question": standalone_question}):
            yield chunk
            time.sleep(0.15)  # slows it down enough to see clearly
        citations = format_citations(docs)
        if citations:
            yield "\n\n" + citations

    return StreamingResponse(generate(), media_type="text/plain")

    return StreamingResponse(generate(), media_type="text/plain")