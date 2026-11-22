### Semantic Job Matcher

Paste a resume and get the most relevant jobs, ranked by meaning rather than keyword overlap.

## How it works

Each job (title + description + skills) is embedded with all-MiniLM-L6-v2 (runs locally).
Embeddings are stored in a ChromaDB collection using cosine distance.
Your resume is embedded the same way, and the nearest jobs are retrieved (this is the retrieval step of RAG).
Optional: Gemini reads the resume plus each retrieved job and explains the fit and one skill gap (the generation step).
resume -> embed -> ChromaDB top-k search -> ranked jobs -> Gemini explanation (optional)

Run locally
bash
pip install -r requirements.txt
streamlit run app.py

Optional: set GEMINI_API_KEY in your environment, or paste the key into the sidebar.

## Ideas to extend :- 

Add a reranking step and compare ranking quality before/after

Evaluate retrieval with a small labelled set (precision@k)

Swap the JSON file for a PostgreSQL table with pgvector
