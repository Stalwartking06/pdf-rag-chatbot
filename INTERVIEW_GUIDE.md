# AI Engineer Interview Preparation Guide: PDF RAG Assistant

> **Companion guide for Akshat**  
> Use this document to master every architectural decision, technical formula, and interview talking point for the **PDF RAG Assistant** project.

---

## Table of Contents

1. [The 60-Second Elevator Pitch](#1-the-60-second-elevator-pitch)
2. [End-to-End Pipeline Walkthrough](#2-end-to-end-pipeline-walkthrough)
3. [Key Architectural Tradeoffs (The "Why" Questions)](#3-key-architectural-tradeoffs-the-why-questions)
4. [Real-World Edge Cases & Bugs Solved](#4-real-world-edge-cases--bugs-solved)
5. [Information Retrieval Evaluation Explained](#5-information-retrieval-evaluation-explained)
6. [Top 20 Technical Interview Questions & Ready Answers](#6-top-20-technical-interview-questions--ready-answers)
7. [Live Screen-Share Demo Script](#7-live-screen-share-demo-script)

---

## 1. The 60-Second Elevator Pitch

When the interviewer asks:  
**"Walk me through your recent AI project"** or **"Tell me about yourself"**, deliver this exact response:

> *"Recently, I designed and built an enterprise-grade PDF RAG Assistant from scratch without using black-box wrapper libraries like LangChain.*  
>  
> *The system indexes a 109-page company policy manual and uses a **two-stage hybrid retrieval pipeline**: it combines dense vector search with FAISS and lexical keyword search with BM25 using **Reciprocal Rank Fusion (RRF $k=60$)**, followed by neural re-ranking using a **Cross-Encoder transformer** (`ms-marco-MiniLM-L-6-v2`).*  
>  
> *I also added strict negative constraints to prevent hallucinations on missing policy details (like unstated annual leave quotas), implemented an Information Retrieval evaluation benchmark achieving **0.9091 MRR**, wrote **35 automated pytest unit tests**, and wrapped everything into a **FastAPI backend** and a modern **React 19 TypeScript** glassmorphism dashboard optimized to run locally within 8 GB of RAM."*

---

## 2. End-to-End Pipeline Walkthrough

```
1. INGESTION & BOUNDARY CHUNKING
   company_policy.pdf (109 pages)
         │  pypdf extraction with page metadata
         ▼
   285 Text Chunks (1000 chars, 200 overlap)
   Boundary-aware splitting on paragraph ('\n\n') and sentence ('. ') ends.

2. DUAL-INDEXING (Startup Lifespan)
   ├── Dense: sentence-transformers (all-MiniLM-L6-v2) ──► FAISS IndexFlatIP (384-dim)
   └── Sparse: Tokenized lowercase text minus stopwords ──► BM25Okapi Index

3. USER QUERY PREPROCESSING
   User Question: "What is the leave policy?"
         │  Spelling correction (difflib with document vocabulary)
         │  Vague query interception ("employ" ──► clarify without LLM)
         ▼  Query expansion with domain synonyms

4. HYBRID RETRIEVAL & FUSION
   ├── FAISS Dense Search ──► Top-20 Candidates
   └── BM25 Lexical Search ──► Top-20 Candidates
         │
         ▼ Reciprocal Rank Fusion: Score = ∑ 1 / (60 + rank)
   Fused Top-35 Unique Candidates

5. NEURAL RE-RANKING
   Cross-Encoder (ms-marco-MiniLM-L-6-v2)
   Joint self-attention on Query-Passage pairs
         │
         ▼ Select Top-3 highest logit passages

6. GROUNDED ANSWER GENERATION
   Local Ollama LLM (qwen2.5:1.5b, temp=0.0)
   Negative prompt constraints: refuse to hallucinate unstated leave numbers or manager names.
         │
         ▼
   Final Answer + Page & Chunk Citations + Retrieval Telemetry
```

---

## 3. Key Architectural Tradeoffs (The "Why" Questions)

### Q1: Why Hybrid Search (FAISS + BM25) instead of Vector Search alone?
- **Vector Search (Dense):** Understands synonyms and concepts (e.g., matching *"mission"* to *"our commitment and goals"*), but fails on exact proper nouns, acronyms, and product codes.
- **BM25 (Sparse Keyword):** Matches exact keywords via Term Frequency / Inverse Document Frequency (TF-IDF), but has zero semantic understanding.
- **Combined Benefit:** Maximizes recall. If an employee asks for a specific title or acronym, BM25 catches it; if they ask a conceptual question, FAISS catches it.

### Q2: Why Reciprocal Rank Fusion (RRF) instead of Weighted Score Sum?
- **The Problem:** BM25 scores range from $0$ to $25+$, while Cosine similarities range from $-1$ to $+1$. Adding them directly or normalizing them is brittle and changes with query length.
- **The Solution:** RRF only considers **rank positions**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{FAISS}, \text{BM25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- It gives top scores to items appearing near the top in both systems, completely scale-agnostic.

### Q3: Why Bi-Encoder + Cross-Encoder Two-Stage Retrieval?
- **Bi-Encoder (`all-MiniLM`):** Query and document are embedded separately. Fast ($O(N)$ dot products in FAISS), but no token-level cross-attention.
- **Cross-Encoder (`ms-marco-MiniLM`):** Query and document are concatenated and passed through transformer layers together. Extremely accurate token-level interaction, but too slow to run across thousands of chunks.
- **Two-Stage Pattern:** Bi-Encoder + BM25 quickly filters 285 chunks down to 35 candidates in ~5ms; Cross-Encoder scores those 35 candidates in ~20ms to pick the top-3.

### Q4: Why FAISS over Pinecone or ChromaDB?
1. **100% Local Data Privacy:** Sensitive corporate policy documents never leave the local environment.
2. **Zero Cost & Offline:** No SaaS cloud bills or internet dependencies.
3. **Minimal Memory:** Raw C++ FAISS CPU (`IndexFlatIP`) takes $< 5\text{MB}$ RAM for 285 chunks, unlike heavier database servers.

### Q5: Why FastAPI with Lifespan over Flask or Streamlit?
- **Lifespan Startup:** Models and indexes are loaded **once** in RAM on server boot. Requests reuse these in-memory singletons, avoiding out-of-memory crashes on 8 GB RAM machines.
- **Decoupled Frontend:** Clean separation between REST API backend and a production React 19 TypeScript UI.

---

## 4. Real-World Edge Cases & Bugs Solved

Interviewers love to hear about actual bugs you identified and fixed:

### 1. Executive Salary Intent Leak
- **Symptom:** Asking *"What is the salary of the CEO?"* previously answered *"Dr. Charu Raheja, PhD – Chair and CEO"*.
- **Root Cause:** Naive regex matching checked `if "ceo" in query:` and immediately returned the executive's identity.
- **The Fix:** Added an intent disambiguation filter in `text_utils.py` checking for `NON_NAME_INTENTS` (`salary`, `compensation`, `pay`, `contact`). If detected, it bypasses the shortcut and routes into the full hybrid RAG pipeline, which correctly returns that salaries are confidential.

### 2. Cross-Topic Chunk Pollution
- **Symptom:** On Page 71 of the PDF, the leave policy sentence is physically adjacent to prescription drug safety guidelines and salary confidentiality. Asking *"What is the leave policy?"* caused the LLM to summarize medication and salary rules.
- **The Fix:** Instructed system prompt with explicit topic isolation:
  *"When answering questions about leave, explain ONLY leave and call-in-sick procedures. NEVER include unrelated rules such as prescription medication guidelines or salary confidentiality, even if they appear in the same excerpt."*

### 3. Vague Single-Word Query Interception
- **Symptom:** Queries like `"employ"` caused arbitrary chunk retrieval.
- **The Fix:** Built `is_vague_question()` and `get_vague_question_suggestion()`. Before calling FAISS or LLM, it politely asks:
  *"Could you please be more specific? For example, you can ask about: Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies."*

### 4. Anti-Hallucination Guardrail (Leave Days & Manager Names)
- **Symptom:** The PDF specifies leave procedures (call in sick, notify supervisor), but **never defines numerical annual vacation day allocations**. Naive LLMs tend to invent numbers (e.g. "14 days").
- **The Fix:** Constrained prompt with: *"If exact leave counts are not stated in the document, explicitly state that they are not specified. NEVER invent leave numbers."*

---

## 5. Information Retrieval Evaluation Explained

Most freshers never measure retrieval mathematically. Highlighting these numbers will immediately set you apart:

| Metric | Formula | Value Achieved | Meaning |
| :--- | :--- | :---: | :--- |
| **Precision@5** | $\frac{\text{Relevant Chunks in Top-5}}{5}$ | **0.4000** | On average, 2 out of the top 5 retrieved chunks were directly relevant. |
| **Recall@5** | $\frac{\text{Relevant Chunks in Top-5}}{\text{Total Ground Truth Chunks}}$ | **0.6121** | Over 61% of all ground-truth relevant passages were captured in the top 5. |
| **Mean Reciprocal Rank (MRR)** | $\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$ | **0.9091** | In **10 out of 11 benchmark queries**, the most relevant passage was ranked at **Position #1**! |

---

## 6. Top 20 Technical Interview Questions & Ready Answers

### Q1: What is RAG?
> *"Retrieval-Augmented Generation is a technique that supplements an LLM prompt with relevant external context retrieved from a custom knowledge base at inference time, ensuring factual, up-to-date, and auditable answers."*

### Q2: Why not just fine-tune an LLM?
> *"Fine-tuning adjusts model weights for tone, syntax, or domain style, but it is expensive, prone to hallucination, cannot provide exact page citations, and requires re-training every time a company policy changes. RAG updates in seconds by simply re-indexing."*

### Q3: What is chunk overlap and why is it needed?
> *"Overlap (e.g. 200 characters) ensures semantic continuity. If a key sentence splits across chunk boundaries, without overlap both chunks become incomplete, leading to retrieval misses."*

### Q4: What is boundary-aware chunking?
> *"Instead of cutting text at fixed character counts, boundary-aware chunking searches backwards for paragraph breaks (`\n\n`) or sentence endings (`. `) so sentences are never sliced in half."*

### Q5: How does FAISS IndexFlatIP work?
> *"It computes inner products (dot products) between query and document vectors. When vectors are L2-normalized, the inner product equals cosine similarity, giving exact, lightning-fast ranking."*

### Q6: How does BM25 work?
> *"BM25 is a probabilistic keyword ranking algorithm that scores documents based on Term Frequency (TF) with saturation limits and Inverse Document Frequency (IDF) with document length normalization."*

### Q7: What is Reciprocal Rank Fusion (RRF)?
> *"RRF combines rankings from multiple retrieval algorithms by summing their reciprocal rank scores: $\frac{1}{k + \text{rank}}$. It resolves score incompatibility without arbitrary weight tuning."*

### Q8: What is a Cross-Encoder?
> *"A transformer model that takes both query and passage simultaneously into its self-attention layers, computing cross-token relevance for higher ranking accuracy than bi-encoders."*

### Q9: Why use temperature = 0.0?
> *"Temperature controls randomness. Setting temperature to 0.0 enforces greedy decoding, making the LLM's generation deterministic, factual, and strictly context-bound."*

### Q10: How do you prevent hallucinations?
> *"By engineering negative system prompt constraints, providing high-precision retrieved context, setting temperature to 0.0, and instructing the model to admit when information is missing."*

### Q11: How do you measure RAG quality?
> *"Using Information Retrieval metrics: Precision@k, Recall@k, and Mean Reciprocal Rank (MRR) across a curated evaluation dataset with ground-truth phrase matching."*

### Q12: What is the dimensionality of `all-MiniLM-L6-v2`?
> *"384 dimensions. It provides an optimal balance between dense semantic expressiveness and minimal memory/compute requirements."*

### Q13: How is this application optimized for 8 GB RAM?
> *"We use singleton model loading via FastAPI lifespan, an in-memory FAISS index, and the lightweight 1.5B Qwen model via Ollama, keeping total RAM consumption under 2.2 GB."*

### Q14: What is the role of query expansion?
> *"Query expansion adds domain-specific synonyms to the query (e.g. expanding 'leave' with 'sick leave, call in sick, absence'). This bridges the vocabulary gap in BM25 search."*

### Q15: Why didn't you use LangChain?
> *"To avoid black-box abstraction and understand RAG from first principles. Writing raw FAISS, BM25, and Cross-Encoder code gave me full control over memory, latency, and scoring."*

### Q16: How do you handle vague queries?
> *"We intercept single broad terms like 'employ' before retrieval and return clarification choices (benefits, responsibilities, leave, conduct) without wasting LLM compute."*

### Q17: What does the FastAPI backend provide?
> *"`/api/health` for monitoring, `/api/document` for live metadata, `/api/chat` for RAG responses with citations, and `/api/chat/reset` for conversation management."*

### Q18: What features does your React frontend include?
> *"A responsive glassmorphism layout, 5 clickable suggested inquiry cards, expandable source citation cards, live retrieval telemetry drawer, and auto-scrolling message list."*

### Q19: What are the 3 pillars of the RAG Triad?
> *"Context Relevance (retrieval quality), Groundedness/Faithfulness (absence of hallucination), and Answer Relevance (answering the exact user question)."*

### Q20: How would you scale this to 100 concurrent enterprise users?
> *"Replace Ollama with vLLM for continuous batching and PagedAttention, use Redis Queue for asynchronous job management, and deploy multiple stateless FastAPI workers behind NGINX."*

---

## 7. Live Screen-Share Demo Script

When demonstrating the project during a technical interview:

1. **Open the React Web App (`http://127.0.0.1:5173/`):**
   - *"Here is the full-stack web interface. On the left sidebar, you can see the target document metadata: `company_policy.pdf`, 109 pages, 285 chunks, with live architecture badges for FAISS, BM25, and Ollama."*
2. **Click a Suggested Inquiry (e.g. *"What is the company mission?"*):**
   - Show how the answer streams back instantly.
   - Expand the **Source [1]** card: point out the Page number (Page 5), Chunk ID (Chunk #10), and Cross-Encoder score (`+2.9323`).
3. **Toggle Retrieval Telemetry:**
   - Click **"View Retrieval Telemetry"**: show the interviewer the expanded query, FAISS candidate count (20), BM25 candidate count (20), and fused candidate count (35).
4. **Demonstrate Anti-Hallucination Guardrails:**
   - Type: *"How many annual vacation days are allowed?"*
   - Show the model's honest response: *"The document does not specify the exact number of leaves or vacation days allowed."*
   - Explain to the interviewer: *"Notice the bot refuses to guess or invent numbers because of our negative system prompt constraints."*
5. **Demonstrate Executive Intent Disambiguation:**
   - Type: *"Who is the CEO?"* $\to$ Answers *"Dr. Charu Raheja"*.
   - Type: *"What is the salary of the CEO?"* $\to$ Shows how it routes to RAG and clarifies salary confidentiality instead of leaking the name.
6. **Open Swagger API Docs (`http://127.0.0.1:8000/docs`):**
   - Show the clean REST contract: `/api/health`, `/api/document`, `/api/chat`, and explain the FastAPI lifespan singleton architecture.
7. **Show Pytest Results in Terminal:**
   - Run `pytest -v`: show all 35 tests passing green in ~20 seconds.

---

*Keep this guide open during interview preparation, review the talking points, and speak with confidence!*
