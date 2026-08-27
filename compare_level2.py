from dotenv import load_dotenv
load_dotenv()
from backend.rag.pipeline import build_pipeline_from_pdf, build_pipeline_from_pdf_v2

PDF_PATH = "uploaded_pdfs/The LSPP Playbook _ LSPP 2026.pdf"
QUESTION = "How is the LSPP program structured, and what activities are associated with each part?"

print("=== BASELINE (Level 1) ===")
baseline_chain = build_pipeline_from_pdf(PDF_PATH)
print(baseline_chain.invoke(QUESTION))

print("\n=== FIXED (Level 2, page 3 spatial reconstruction) ===")
fixed_chain = build_pipeline_from_pdf_v2(PDF_PATH, pages_needing_fix=[2])
print(fixed_chain.invoke(QUESTION))