"""
Gradio UI — Levels 1-5 (upload, chat, streaming, citations, conversational).

This file only handles the UI: file upload, chat display, wiring button
clicks to functions. All RAG logic lives in backend/rag/pipeline.py
(and backend/rag/extraction.py for the Level 2 layout fix).
"""

import os
import shutil
import gradio as gr
from dotenv import load_dotenv

from backend.rag.pipeline import (
    build_qa_components,
    format_docs,
    format_citations,
    format_chat_history,
)

load_dotenv()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page 3 (0-indexed: 2) is the "Learn. Lead. Grow." diagram where numbers
# are visually separated from their labels — see backend/rag/extraction.py.
PAGES_NEEDING_LAYOUT_FIX = [2]


def handle_upload(file):
    """
    Runs when the user uploads a PDF.

    We don't hardcode a filename anywhere — `file.name` is whatever path
    Gradio gave the uploaded temp file, and we copy it into our own
    uploaded_pdfs/ folder using its original name so re-uploads are easy
    to trace during testing.
    """
    if file is None:
        return None, "No file uploaded yet."

    dest_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    shutil.copy(file.name, dest_path)

    try:
        qa_components = build_qa_components(dest_path, pages_needing_fix=PAGES_NEEDING_LAYOUT_FIX)
    except Exception as e:
        return None, f"Failed to process PDF: {e}"

    return qa_components, f"Loaded '{os.path.basename(file.name)}'. Ask a question below."


def answer_question(qa_components, question, history):
    """
    Generator function (uses yield) — this is what makes Gradio stream
    the response chunk-by-chunk into the chat UI instead of waiting for
    the whole answer to finish.

    Level 5: before retrieval, prior conversation history is used to
    rewrite the raw question into a standalone one (resolving pronouns
    like "it" or references like "the last one"). Both retrieval and the
    final answer use this rewritten question — the raw question is only
    shown to the user in the chat log.
    """
    if qa_components is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload a PDF first."},
        ]
        yield history, ""
        return

    if not question.strip():
        yield history, ""
        return

    retriever, answer_chain, condense_chain = qa_components

    chat_history_text = format_chat_history(history)  # prior turns only
    if chat_history_text.strip():
        standalone_question = condense_chain.invoke(
            {"chat_history": chat_history_text, "question": question}
        )
    else:
        standalone_question = question

    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    yield history, ""

    for chunk in answer_chain.stream({"context": context, "question": standalone_question}):
        history[-1]["content"] += chunk
        yield history, ""

    citations = format_citations(docs)
    if citations:
        history[-1]["content"] += "\n\n" + citations
        yield history, ""


with gr.Blocks(title="LSPP RAG Chatbot") as demo:
    gr.Markdown("## LSPP RAG Chatbot (Levels 1-5: Upload, Streaming, Citations, Conversational)")

    qa_state = gr.State(None)  # holds (retriever, answer_chain, condense_chain) for the current PDF

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat")
    question_box = gr.Textbox(label="Ask a question about the PDF", placeholder="Type your question...")

    pdf_upload.change(
        fn=handle_upload,
        inputs=[pdf_upload],
        outputs=[qa_state, status],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[qa_state, question_box, chatbot],
        outputs=[chatbot, question_box],
    )


if __name__ == "__main__":
    demo.launch()