
---

# 🧠 Mini RAG Application

### 🎯 Objective

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user questions based on the content of a PDF.
It retrieves the most relevant chunks of text, sends them (with the question) to an LLM (Gemini), and compares the RAG-generated answers with non-RAG (plain LLM) answers — saving results in **CSV** and **Markdown** files.

---

## 🚀 Features

✅ Extract text from PDF using **PyPDF2** and **pdfplumber** (choose which works better)

✅ Chunk text into manageable pieces for better retrieval

✅ Create and search **FAISS** vector database using **sentence-transformers** embeddings

✅ Retrieve top-k similar chunks for any question

✅ Query an LLM (Gemini or OpenAI) with and without context (RAG vs non-RAG)

✅ Compare answers based on **semantic similarity**, not just length

✅ Automatically save and **append** results in:

* `comparison_results.csv`
* `comparison_analysis.md`
* 
✅ Streamlit-based clean UI for interactive usage

✅ Built-in **FAQ section** to guide new users

---

## 🧩 Project Structure

```
MINI_RAG_APPLICATION/
│
├── mini_rag.py                # Main Streamlit app
├── comparison_results.csv     # Appends all question comparisons
├── comparison_analysis.md     # Human-readable comparison log
├── requirements.txt           # Dependencies
└── .env                       # Contains API key
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone or download this repo

```bash
git clone https://github.com/AhmedFaraz050/mini-rag-app.git
cd mini-rag-app
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate     # On Windows
source .venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit PyPDF2 pdfplumber faiss-cpu sentence-transformers google-generativeai python-dotenv
```

### 4️⃣ Create a `.env` file

Create a file named `.env` in your project root and add your API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ Make sure you don’t share or commit your `.env` file.

---

## ▶️ How to Run

  Start the Streamlit app:

```bash
 streamlit run mini_rag.py
```

Once it starts:

1. Upload your PDF
2. Enter a question
3. Click “Run RAG Analysis”
4. See answers side-by-side
5. Download `.csv` or `.md` comparison report

---

## 🧠 Comparison Analysis Logic

| Step | Description                                                                 |
| ---- | --------------------------------------------------------------------------- |
| 1    | User uploads PDF and enters a question                                      |
| 2    | System splits document into text chunks                                     |
| 3    | FAISS retrieves top 3–5 similar chunks                                      |
| 4    | LLM generates two answers: RAG (with context) and Non-RAG (without context) |
| 5    | Semantic similarity (cosine) compares both answers                          |
| 6    | Results are stored in CSV and Markdown                                      |
| 7    | Similarity score indicates when RAG helped                                  |

---

## 📄 Output Example

**comparison_results.csv**

| Question    | RAG Answer | Non-RAG Answer                | Similarity | Better | Timestamp        |
| ----------- | ---------- | ----------------------------- | ---------- | ------ | ---------------- |
| What is AI? | AI is...   | Artificial Intelligence is... | 0.89       | RAG    | 2025-10-11 20:31 |

**comparison_analysis.md**

```
## Question 1
**Question:** What is AI?
**RAG Answer:** AI is...
**Non-RAG Answer:** Artificial Intelligence is...
**Similarity:** 0.89
**Better Answer:** RAG
---
```

---

## 🧾 Libraries Used

| Library               | Purpose                      |
| --------------------- | ---------------------------- |
| Streamlit             | Frontend app interface       |
| PyPDF2 / pdfplumber   | PDF text extraction          |
| sentence-transformers | Text embeddings              |
| FAISS                 | Vector database              |
| google-generativeai   | LLM API (Gemini)             |
| python-dotenv         | Environment variable loading |

---

## 🧠 Challenges & Solutions

| Challenge                   | Solution                                                   |
| --------------------------- | ---------------------------------------------------------- |
| Inconsistent PDF extraction | Used both PyPDF2 and pdfplumber, tested which works better |
| Slow LLM responses          | Cached FAISS index and reused embeddings                   |
| Overwriting results         | Modified CSV/MD saving to **append** instead               |
| Weak comparisons            | Used **semantic similarity (cosine)** instead of length    |

---

## 🏁 Evaluation Criteria (50 pts)

| Criteria                     | Points |
| ---------------------------- | ------ |
| PDF extraction works         | 8      |
| Chunking & embeddings        | 8      |
| Vector database setup        | 8      |
| End-to-end RAG pipeline      | 10     |
| Comparison analysis          | 10     |
| Code quality & documentation | 6      |

---

