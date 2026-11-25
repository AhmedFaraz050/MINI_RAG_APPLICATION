import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Mini RAG Application", layout="wide")
st.title("📘 Mini RAG Application")
st.write("Build a Retrieval-Augmented Generation (RAG) system to answer questions from PDF documents.")

# ----------------- LOAD ENV & CONFIG GEMINI -----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please create a .env file with GEMINI_API_KEY.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.success("✅ Gemini API configured successfully!")
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {e}")

# ----------------- FUNCTIONS -----------------
def extract_text_pypdf2(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def create_faiss_index(chunks, _model):
    embeddings = _model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_top_chunks(question, chunks, index, model, top_k=3):
    q_emb = model.encode([question])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0]]

def ask_gemini(prompt, model_name="gemini-2.0-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def semantic_similarity(a, b, model):
    emb1 = model.encode([a])
    emb2 = model.encode([b])
    return cosine_similarity(emb1, emb2)[0][0]

# ----------------- COMMON QUESTIONS -----------------
common_questions = [
    "What is the main topic of this document?",
    "Summarize the introduction section.",
    "What conclusions are drawn in the PDF?",
    "Mention any statistics or data shared.",
    "Who are the authors or organizations mentioned?",
    "What methodology was used?",
    "What are the key findings?",
    "What challenges or limitations are discussed?",
    "What is the future work or recommendations?",
    "Provide a brief overall summary."
]

# ----------------- STREAMLIT UI -----------------
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # ---- Extraction ----
    st.subheader("1️⃣ PDF Text Extraction")
    pypdf2_text = extract_text_pypdf2(pdf_path)
    pdfplumber_text = extract_text_pdfplumber(pdf_path)

    if len(pdfplumber_text) > len(pypdf2_text):
        text = pdfplumber_text
        best_method = "pdfplumber"
    else:
        text = pypdf2_text
        best_method = "PyPDF2"

    st.write(f"✅ Using **{best_method}** (Extracted {len(text.split())} words).")

    # ---- Chunking ----
    st.subheader("2️⃣ Text Chunking & Embeddings")
    chunk_size = st.slider("Select chunk size (words per chunk)", 200, 1000, 500, 100)
    chunks = chunk_text(text, chunk_size)
    st.write(f"🧩 Created **{len(chunks)}** text chunks.")

    model = load_embedding_model()
    index, embeddings = create_faiss_index(chunks, model)
    st.success("✅ FAISS vector database created successfully!")

    # ---- Question Section ----
    st.subheader("3️⃣ Ask a Question")

    st.markdown("### 💬 Commonly Asked Questions")
    selected_question = None
    cols = st.columns(2)

    for i, q in enumerate(common_questions):
        if cols[i % 2].button(q):
            selected_question = q

    custom_q = st.text_input("Or type your own question:")

    # Determine final query
    query = custom_q if custom_q else selected_question

    if query:
        st.info(f"🔍 Searching for: **{query}**")

        # ---- Retrieve Context ----
        top_chunks = retrieve_top_chunks(query, chunks, index, model)
        context = "\n\n".join(top_chunks)

        # ---- RAG ----
        rag_prompt = f"Use the following context from the PDF to answer:\n\n{context}\n\nQuestion: {query}"
        rag_answer = ask_gemini(rag_prompt)
        nonrag_answer = ask_gemini(query)

        # ---- Display ----
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧠 RAG Answer")
            st.write(rag_answer)
        with col2:
            st.markdown("### 🤖 Non-RAG Answer")
            st.write(nonrag_answer)

        # ---- Context Chunks ----
        st.markdown("### 📄 Top Relevant Chunks Used")
        for i, ch in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {ch[:400]}...")

        # ---- Save Comparison (CSV) ----
        df = pd.DataFrame([[query, rag_answer, nonrag_answer]], columns=["Question", "RAG Answer", "Non-RAG Answer"])
        csv_path = "comparison_results.csv"

        # Append to existing CSV without overwriting
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        # ---- Semantic Comparison ----
        rag_sim = semantic_similarity(rag_answer, context, model)
        nonrag_sim = semantic_similarity(nonrag_answer, context, model)

        if rag_sim > nonrag_sim:
            better = "✅ RAG performed better — more aligned with document context."
        elif nonrag_sim > rag_sim:
            better = "🤖 Non-RAG performed better — question might not require document context."
        else:
            better = "⚖️ Both performed similarly."

        st.subheader("4️⃣ Comparison Analysis")
        st.write(better)

        # ---- Markdown Report (Append Mode) ----
        md_path = "comparison_analysis.md"
        # Add header if file is empty or doesn’t exist
        if not os.path.exists(md_path) or os.stat(md_path).st_size == 0:
            with open(md_path, "a", encoding="utf-8") as f:
                f.write("# 📊 Mini RAG Comparison Analysis\n\n")

        with open(md_path, "a", encoding="utf-8") as f:
            f.write(f"### Question\n{query}\n\n")
            f.write(f"**RAG Answer:**\n{rag_answer}\n\n")
            f.write(f"**Non-RAG Answer:**\n{nonrag_answer}\n\n")
            f.write(f"**Semantic Similarity**\n- RAG vs Context: {rag_sim:.3f}\n- Non-RAG vs Context: {nonrag_sim:.3f}\n\n")
            f.write(f"**Verdict:** {better}\n\n---\n\n")

        st.success("💾 Results appended to comparison_results.csv and comparison_analysis.md")

else:
    st.warning("📤 Please upload a PDF to begin.")
