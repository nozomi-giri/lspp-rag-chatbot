"""
Gradio UI — Level 1. Only handles upload/chat wiring.
All RAG logic lives in backend/rag/pipeline.py.
"""

import os
import shutil
import gradio as gr
from dotenv import load_dotenv

from backend.rag.pipeline import build_pipeline_from_pdf

load_dotenv()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def handle_upload(file):
    """No hardcoded filename — file.name is whatever Gradio's temp path is."""
    if file is None:
        return None, "No file uploaded yet."

    dest_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    shutil.copy(file.name, dest_path)

    try:
        chain = build_pipeline_from_pdf(dest_path)
    except Exception as e:
        return None, f"Failed to process PDF: {e}"

    return chain, f"Loaded '{os.path.basename(file.name)}'. Ask a question below."


def answer_question(chain, question, history):
    if chain is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload a PDF first."},
        ]
        return history, ""
    if not question.strip():
        return history, ""
    answer = chain.invoke(question)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, ""

with gr.Blocks(title="LSPP RAG Chatbot — Level 1") as demo:
    gr.Markdown("## LSPP RAG Chatbot (Level 1: Bring Your Own PDF)")

    chain_state = gr.State(None)  # holds the built QA chain for the current PDF

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat")
    question_box = gr.Textbox(label="Ask a question about the PDF", placeholder="Type your question...")

    pdf_upload.change(fn=handle_upload, inputs=[pdf_upload], outputs=[chain_state, status])
    question_box.submit(fn=answer_question, inputs=[chain_state, question_box, chatbot], outputs=[chatbot, question_box])


if __name__ == "__main__":
    demo.launch()