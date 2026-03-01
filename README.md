# revou-gen-ai-tutorial

## Mini Project: FAQ Agent

The FAQ Agent answers frequently asked questions about Dexa Medica using RAG over the FAQ document.

### Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env_template` to `.env` and fill in `OPENAI_API_KEY`, `DB_PATH`, and optionally `LANGSMITH_API_KEY`
3. Download FAQ PDF: `python scripts/download_faq_pdf.py` (or place `docs/FAQ Dexa Medica.pdf` manually)

### Run

- **FAQ Chat (standalone)**: Run Streamlit and open "Mini Project" in the sidebar
- **Multi-agent (Lab 8)**: Run Streamlit and open "Lab 8" – asks DB (DBQNA), company profile (RAG), or FAQ questions

### LangSmith Tracing

Set in `.env`:
```
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=<your-key>
```

### ROUGE Evaluation

```bash
python scripts/evaluate_faq.py
```

Update `scripts/faq_ground_truth.json` with Q&A pairs from the FAQ document for accurate evaluation.

