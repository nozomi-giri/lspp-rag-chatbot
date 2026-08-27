# LSPP RAG Chatbot

Live demo: https://lspp-rag-chatbot.vercel.app
Backend API docs: https://lspp-rag-chatbot.onrender.com/docs

## What document did I use?

The LSPP Playbook (LSPP 2026), a 10 page guide that explains the program's three phases (Learn, Lead, Grow), the activities in each one, and the monthly checkpoints participants need to hit.

## What levels did I reach?

All of them, 1 through 6.

- Level 1: upload a PDF and chat with it
- Level 2: fixed how messy tables/layouts get extracted (more on this below)
- Level 3: streaming responses instead of one big dump of text
- Level 4: citations with page numbers under every answer
- Level 5: conversational memory, so follow up questions with pronouns actually get resolved
- Level 6: split into a FastAPI backend and a React frontend, deployed for real (backend on Render, frontend on Vercel)

## One thing that surprised me

Page 3 of the playbook has a small diagram where numbers (like "6 monthly meetups") sit visually apart from their labels. With the plain PyPDFLoader, that page got flattened into text and the bot just lost that information completely, no error, it just answered confidently without mentioning the numbers at all.

I only caught this because I ran the same question through the baseline pipeline and a fixed one side by side (in compare_level2.py) and compared the two answers directly. The fixed version recovered the numbers correctly. What got me was how quiet the failure was. If I hadn't specifically gone looking for it, I would've assumed the bot was working fine, since it never threw an error and the rest of the answer looked completely normal.
