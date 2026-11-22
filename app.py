import json
import os
import chromadb
import streamlit as st
import time
from sentence_transformers import SentenceTransformer # sentence transformer is used for generating embeddings for the text data
# example if text data is "Hello World", the embedding will be a vector of numbers that represents the meaning of the text data




# ---------- config ----------
EMBED_MODEL = "all-MiniLM-L6-v2"  # runs locally, free,
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # change if Google renames it
TOP_K = 5

# this is the function that loads the sentence transformer model for generating embeddings
# @st.cache_resource is used to cache the model so that it doesn't have to be loaded every time the app is run
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def build_index():
    """Embed every job once and store it in an in-memory ChromaDB collection."""
    # jobs .json is a file that contains the job data in JSON format. It is loaded into a Python dictionary using the json.load() function.
    # encoding="utf-8" is used to ensure that the file is read correctly, especially if it contains non-ASCII characters.

    with open("jobs.json", encoding="utf-8") as f:
        jobs = json.load(f)

        # we load the embedder model and create a ChromaDB client and collection to store the embeddings
        # collection is a ChromaDB collection that will store the embeddings for the job data

    embedder = load_embedder()
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="jobs", metadata={"hnsw:space": "cosine"}
    )


    # if count 0 , then empty
    if collection.count() == 0:

        # with docs we create a list of documents that will be used to generate embeddings. Each document is a string that contains the job title, description, and skills required for the job.

        docs = [f"{j['title']}. {j['description']} Skills: {', '.join(j['skills'])}" for j in jobs]

        # in collection we add ids which is a list of job ids, documents which is a list of job descriptions, embeddings which is a list of embeddings generated from the documents, and metadatas which is a list of dictionaries containing the job title, company, and skills required for the job.

        collection.add(
            ids=[str(j["id"]) for j in jobs],
            documents=docs,
            embeddings=embedder.encode(docs).tolist(),
            metadatas=[
                {"title": j["title"], "company": j["company"], "skills": ", ".join(j["skills"])}
                for j in jobs
            ],
        )
    return collection

# this function takes in resume and returns the top k jobs that match the resume based on the embeddings generated from the resume and the job data. The function uses cosine similarity to calculate the distance between the resume embedding and the job embeddings, and returns the jobs with the highest similarity scores.

def search_jobs(resume_text: str, k: int = TOP_K):
    embedder = load_embedder()
    collection = build_index()

    # query_vec is the embedding generated from the resume text using the embedder model. The embedding is a vector of numbers that represents the meaning of the resume text. The query_vec is then used to query the ChromaDB collection to find the top k jobs that match the resume based on cosine similarity.
    query_vec = embedder.encode([resume_text]).tolist()

    # res = reslt of query to chromadb
    res = collection.query(query_embeddings=query_vec, n_results=k)

    # results is a list of dictionary that contains the job title, company, skills required for the job, job description, and similarity score for each of the top k jobs that match the resume. The similarity score is calculated as (1 - distance) * 100, where distance is the cosine distance between the resume embedding and the job embedding. A score of 100 indicates that the resume and job are identical, while a score of 0 indicates that they are completely different.

    results = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]  # cosine distance: 0 = identical
        results.append(
            {
                "title": res["metadatas"][0][i]["title"],
                "company": res["metadatas"][0][i]["company"],
                "skills": res["metadatas"][0][i]["skills"],
                "description": res["documents"][0][i],
                "score": round((1 - distance) * 100, 1),
            }
        )
    return results

# this function takes in an API key, resume text, and a job dictionary, and returns a string that explains why the job matches (or doesn't match) the candidate's resume. The function uses the Google Gemini API to generate the explanation based on the resume and job data. The explanation is returned as a string containing three bullet points that highlight the match and one skill gap.
def explain_match(api_key: str, resume_text: str, job: dict, retries: int = 3) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = (
        "You are a career assistant. In 3 short bullet points, explain why this job "
        "matches the candidate's resume, and name one skill gap. Be specific and honest.\n\n"
        f"RESUME:\n{resume_text}\n\n"
        f"JOB: {job['title']} at {job['company']}\n{job['description']}\nSkills: {job['skills']}"
    )
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
        except Exception as e:
            # 503 / overloaded: wait 1s, then 2s, then give up
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

# this is for UI
st.set_page_config(page_title="Job Matcher", page_icon="🎯")
st.title("Semantic Job Matcher")
st.caption("Paste a resume. Jobs are ranked by meaning (embeddings + cosine similarity), not keywords.")

api_key = st.sidebar.text_input(
    "Gemini API key (optional)",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Only used for the 'why this match' explanations. Search itself is fully local.",
)

resume = st.text_area("Resume text", height=240, placeholder="Paste your resume here...")

# if the user clicks the "Find matching jobs" button, the app checks if the resume text is at least 40 characters long. If it is, the app generates embeddings for the resume and searches for matching jobs using the search_jobs function. The top matching jobs are then displayed in a container with their title, company, match score, and required skills. If an API key is provided, the app also calls the explain_match function to provide an explanation of why each job matches (or doesn't match) the candidate's resume.
if st.button("Find matching jobs", type="primary"):
    if len(resume.strip()) < 40:
        st.warning("Paste a bit more of your resume first.")
    else:
        with st.spinner("Embedding and searching..."):
            matches = search_jobs(resume)

        for job in matches:
            with st.container(border=True):
                st.subheader(f"{job['title']} · {job['company']}")
                st.progress(min(int(job["score"]), 100), text=f"Match: {job['score']}%")
                st.write(f"**Skills:** {job['skills']}")
                if api_key:
                    try:
                        st.info(explain_match(api_key, resume, job))
                    except Exception as e:  # keep the app usable if the API call fails
                        st.error(f"Gemini call failed: {e}")
