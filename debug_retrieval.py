from dotenv import load_dotenv
load_dotenv()
from backend.rag.pipeline import build_vectorstore, load_and_split_pdf, chunk_documents

pages = load_and_split_pdf("uploaded_pdfs/The LSPP Playbook _ LSPP 2026.pdf")
chunks = chunk_documents(pages)
vs = build_vectorstore(chunks)

retriever = vs.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What are the three phases of the LSPP program?")
for d in docs:
    print("PAGE:", d.metadata.get("page"))
    print(d.page_content[:200])
    print("---")