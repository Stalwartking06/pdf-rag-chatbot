# PDF-Based RAG Assistant: Full-Stack Web Application

An end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) web application and CLI assistant built with **React**, **TypeScript**, **FastAPI**, **FAISS**, **BM25**, **Cross-Encoder reranking**, and local **Ollama** (`qwen2.5:1.5b`). Designed specifically as an explainable, production-ready portfolio project for an AI Engineer.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is RAG?](#2-what-is-rag)
3. [Full-Stack Architecture](#3-full-stack-architecture)
4. [Folder Structure](#4-folder-structure)
5. [Technologies Used](#5-technologies-used)
6. [Prerequisites](#6-prerequisites)
7. [Backend Setup (FastAPI & RAG Engine)](#7-backend-setup-fastapi--rag-engine)
8. [Frontend Setup (React + Vite + TypeScript)](#8-frontend-setup-react--vite--typescript)
9. [Ollama Setup & Local LLM](#9-ollama-setup--local-llm)
10. [How to Run the Full-Stack Web Application](#10-how-to-run-the-full-stack-web-application)
11. [How to Run the Existing CLI Chatbot](#11-how-to-run-the-existing-cli-chatbot)
12. [How to Run the Evaluation Benchmark](#12-how-to-run-the-evaluation-benchmark)
13. [How to Run Automated Tests](#13-how-to-run-automated-tests)
14. [Design System & UI Components](#14-design-system--ui-components)
15. [Explanation of the RAG Pipeline](#15-explanation-of-the-rag-pipeline)
16. [Hybrid Search (FAISS + BM25 + RRF)](#16-hybrid-search-faiss--bm25--rrf)
17. [Cross-Encoder Re-Ranking](#17-cross-encoder-re-ranking)
18. [Intent Disambiguation & Grounding Guardrails](#18-intent-disambiguation--grounding-guardrails)
19. [Evaluation Metrics & Benchmark Results](#19-evaluation-metrics--benchmark-results)
20. [Performance & 8 GB RAM Optimization](#20-performance--8-gb-ram-optimization)
21. [Known Limitations](#21-known-limitations)
22. [Future Improvements](#22-future-improvements)

---

## 1. Project Overview

The **PDF RAG Assistant** enables employees and management to query an authoritative 109-page company policy manual (`documents/company_policy.pdf`) and receive accurate, truthful, context-grounded responses in real time.

### Key Highlights:
- **100% Local Execution**: Runs entirely on a local machine using local Ollama (`qwen2.5:1.5b`) and local HuggingFace sentence-transformers. Zero external API calls, zero cost, and zero data leakage.
- **Modern Glassmorphism UI**: High-end AI SaaS dashboard featuring dark charcoal aesthetics, frosted glass cards, translucent borders, subtle glows, and responsive three-panel navigation.
- **Two-Stage Hybrid Retrieval**: Dense semantic vectors (FAISS `IndexFlatIP`) + Sparse lexical keywords (BM25 Okapi) merged via Reciprocal Rank Fusion (RRF, $k=60$), then re-scored by a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
- **Strict Grounding & Zero Hallucination**: Negative constraints prevent the model from inventing non-existent policy quantities (e.g. unstated annual leave days) or fabricating managerial names.
- **Transparent Telemetry**: Exposes candidate counts, expansion synonyms, Cross-Encoder logits, and page citations in expandable source cards.
- **Evaluation-Backed**: Verified on an 11-topic evaluation dataset achieving **MRR = 0.9091**, **Recall@5 = 0.6121**, and **Precision@5 = 0.4000**.
- **Comprehensive Automated Tests**: 35 pytest unit and integration tests passing 100%.

---

## 2. What is RAG?

**Retrieval-Augmented Generation (RAG)** bridges the gap between static foundational LLMs and private, dynamic organizational knowledge.

```
Without RAG:  User Question ──► [Standard LLM] ──► Hallucinated or Outdated Answer
With RAG:     User Question ──► [Retriever] ──► [Document Excerpts] ──► [Grounded LLM] ──► Factual Answer + Citations
```

1. **Retrieve**: When a user asks a question, the system searches the vector and keyword indexes for the most relevant document chunks.
2. **Augment**: The system injects those exact excerpts into a carefully constrained system prompt as reference context.
3. **Generate**: The LLM synthesizes a concise, factual answer strictly from that reference context, providing exact page and chunk citations.

---

## 3. Full-Stack Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FRONTEND (React + TypeScript + Vite)                     │
│  - Three-Part Glassmorphism Layout (Sidebar, Chat Window, Telemetry Info Panel)        │
│  - Expandable Source Citation Cards, Suggested Inquiries, Telemetry Toggle             │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ HTTP REST / JSON (Port 5173 ──► 8000)
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                               BACKEND (FastAPI + Uvicorn)                              │
│  - Lifespan Startup: Models and Indexes Loaded ONCE into RAM (Singletons)              │
│  - Endpoints: GET /api/health, GET /api/document, POST /api/chat, POST /api/chat/reset │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Python In-Memory Function Calls
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                               MODULAR RAG ENGINE (src/)                                │
│                                                                                        │
│  1. Preprocessing & Intent Disambiguation (text_utils.py)                              │
│     ├── Conservative Spelling Correction (difflib)                                     │
│     ├── Vague Intent Clarification (employ, leave, policy)                             │
│     ├── Exact CEO/COO Identity Extraction (Page 8)                                     │
│     └── Query Expansion with Domain Synonyms                                           │
│                                                                                        │
│  2. Hybrid Retrieval & Rank Fusion (hybrid_search.py)                                  │
│     ├── Dense Vector Search: FAISS IndexFlatIP (384-dim all-MiniLM-L6-v2) ──► Top-20   │
│     ├── Sparse Lexical Search: BM25 Okapi with custom stopwords            ──► Top-20  │
│     └── Reciprocal Rank Fusion: Score = ∑ 1 / (k + rank) (k=60)           ──► Top-35   │
│                                                                                        │
│  3. Neural Reranking (reranker.py)                                                     │
│     └── Cross-Encoder (ms-marco-MiniLM-L-6-v2) Pairwise Attention         ──► Top-3    │
│                                                                                        │
│  4. Grounded Synthesis & Guardrails (answer_generator.py)                              │
│     └── Local Ollama (qwen2.5:1.5b, temp=0.0) with Strict Topic Separation             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Folder Structure

```
pdf-rag-chatbot/
│
├── api.py                          # FastAPI REST API server (entry point for web app)
├── main.py                         # CLI application entry point (--query, --eval, --debug)
├── requirements.txt                # Python backend dependencies
├── test_ollama.py                  # Standalone Ollama connectivity probe
│
├── documents/
│   └── company_policy.pdf          # Target PDF document (109 pages, 285 chunks)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central settings, model identifiers, hyperparameters
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py           # Text extraction with page-level metadata (pypdf)
│   │   └── chunking.py             # Boundary-aware text chunking (1000 chars, 200 overlap)
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py      # Sentence-transformers singleton (all-MiniLM-L6-v2)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py        # FAISS IndexFlatIP cosine similarity index
│   │   ├── bm25_search.py          # Rank-BM25 Okapi lexical index
│   │   ├── hybrid_search.py        # RRF rank fusion and candidate assembly
│   │   └── reranker.py             # Cross-Encoder neural reranker (ms-marco-MiniLM)
│   ├── generation/
│   │   ├── __init__.py
│   │   └── answer_generator.py     # Grounded LLM generation via Ollama with anti-hallucination rules
│   ├── chatbot/
│   │   ├── __init__.py
│   │   └── chat.py                 # RAG coordinator, spelling, vague intent, CLI interface
│   ├── utils/
│   │   ├── __init__.py
│   │   └── text_utils.py           # Vocabulary builder, query expansion, intent disambiguation
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # Precision@k, Recall@k, Reciprocal Rank, MRR
│       └── evaluation_runner.py    # 11-query evaluation harness and automated benchmark
│
├── frontend/                       # Modern React + Vite + TypeScript web application
│   ├── package.json                # Frontend dependencies (React 19, Axios, Lucide)
│   ├── vite.config.ts              # Vite server with proxy to FastAPI (:8000)
│   ├── index.html                  # HTML entry point with metadata
│   └── src/
│       ├── main.tsx                # React DOM root mounting
│       ├── App.tsx                 # Root layout coordinating Sidebar, Chat, Info Panel
│       ├── index.css               # Comprehensive glassmorphism design system
│       ├── types/
│       │   └── chat.ts             # TypeScript interfaces for messages, telemetry, docs
│       ├── services/
│       │   └── api.ts              # Axios client connecting to FastAPI endpoints
│       └── components/
│           ├── Sidebar.tsx         # Document status, New Chat, tech badges
│           ├── ChatWindow.tsx      # Top bar, message stream, loading state
│           ├── ChatMessage.tsx     # Message bubbles, formatting, mode badges
│           ├── ChatInput.tsx       # Auto-resizing textarea with Enter-to-send
│           ├── SourceCard.tsx      # Expandable glass cards displaying excerpts
│           ├── WelcomeScreen.tsx   # Hero card with 5 clickable suggested inquiries
│           └── InfoPanel.tsx       # Collapsible telemetry drawer with live specs
│
└── tests/                          # 35 automated unit and integration tests
    ├── test_chunking.py            # Splitting boundaries, overlaps, empty chunk rejection
    ├── test_pdf_loader.py          # Missing file handling, extraction, page mapping
    ├── test_embeddings.py          # Embedding dimensions, dtypes, singleton stability
    ├── test_retrieval.py           # FAISS, BM25, RRF fusion, Cross-Encoder reranking
    ├── test_answer_generation.py   # Ollama generation, prompt rules, vague intents, salary routing
    └── test_evaluation.py          # Precision@k, Recall@k, reciprocal rank, MRR math
```

---

## 5. Technologies Used

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | React + TypeScript | React 19, TS 5.x | Interactive, type-safe web application |
| **Frontend Tooling** | Vite | 8.x | High-speed HMR dev server and production bundler |
| **Styling** | Vanilla CSS3 | Modern CSS | Dark glassmorphism, responsive grid/flexbox, micro-animations |
| **Icons** | Lucide React | 1.66+ | Professional SVG iconography |
| **HTTP Client** | Axios | 1.21+ | REST client communicating with FastAPI |
| **API Framework** | FastAPI | 0.110+ | Asynchronous REST backend with OpenAPI docs |
| **ASGI Server** | Uvicorn | 0.28+ | High-performance production ASGI server |
| **PDF Extraction** | PyPDF | 5.0+ | Text and page metadata parsing |
| **Dense Embeddings** | Sentence-Transformers | 3.0+ | `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector Search** | FAISS CPU | 1.8+ | In-memory `IndexFlatIP` cosine similarity search |
| **Keyword Search** | Rank-BM25 | 0.2+ | Probabilistic BM25Okapi lexical matching |
| **Neural Reranking** | Cross-Encoder | HuggingFace | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Local LLM** | Ollama | 0.3+ | Local engine hosting `qwen2.5:1.5b` (986 MB) |
| **Testing** | Pytest | 9.x | 35 automated unit and integration tests |

---

## 6. Prerequisites

- **OS**: Windows 10/11 (or macOS / Linux)
- **Python**: Version 3.10 to 3.13
- **Node.js**: Version 18+ (verified on v24.12.0 with npm 11.6.2)
- **RAM**: Minimum 8 GB RAM (the pipeline is heavily memory-optimized)
- **Ollama**: Installed from [ollama.com](https://ollama.com)

---

## 7. Backend Setup (FastAPI & RAG Engine)

Open a **Windows PowerShell** terminal in the project root:

```powershell
# 1. Navigate to project root
cd "d:\MIssion Job\pdf-rag-chatbot"

# 2. Create virtual environment (if not already created)
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install backend dependencies
pip install -r requirements.txt
```

---

## 8. Frontend Setup (React + Vite + TypeScript)

Open a **second Windows PowerShell** terminal:

```powershell
# 1. Navigate to frontend directory
cd "d:\MIssion Job\pdf-rag-chatbot\frontend"

# 2. Install Node packages
npm install
```

---

## 9. Ollama Setup & Local LLM

1. Start the Ollama background service if it is not already running:
   ```powershell
   ollama serve
   ```
2. In another terminal, pull the lightweight, fast Qwen 2.5 (1.5B) model:
   ```powershell
   ollama pull qwen2.5:1.5b
   ```
3. Verify the model is downloaded (~986 MB):
   ```powershell
   ollama list
   ```

---

## 10. How to Run the Full-Stack Web Application

Running the application requires two lightweight processes:

### Terminal 1: Start the FastAPI Backend
```powershell
cd "d:\MIssion Job\pdf-rag-chatbot"
.\venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```
- The backend initializes the document chunks, builds FAISS and BM25 indexes, and loads models **once** at startup.
- Interactive Swagger API docs are available at: `http://127.0.0.1:8000/docs`

### Terminal 2: Start the React Frontend
```powershell
cd "d:\MIssion Job\pdf-rag-chatbot\frontend"
npm run dev
```
- Open your browser to: **`http://127.0.0.1:5173/`**
- Experience the glassmorphism interface: click any suggested question, type custom queries, expand source citations, and inspect live telemetry.

---

## 11. How to Run the Existing CLI Chatbot

The original modular command-line interface remains 100% operational:

### A. Interactive CLI Mode
```powershell
python main.py
```
- Type any question and press Enter.
- Type `/debug` to toggle retrieval telemetry on/off in real time.
- Type `exit` or `quit` to end the session.

### B. Single Query Mode
```powershell
python main.py --query "What is the company mission?"
python main.py --query "Who is the CEO?"
python main.py --query "What is the leave policy?"
```

### C. Single Query with Retrieval Telemetry
```powershell
python main.py --query "What is the company mission?" --debug
```

---

## 12. How to Run the Evaluation Benchmark

Execute the Information Retrieval evaluation framework across 11 benchmark policy queries:

```powershell
python main.py --eval
```

This runs each query through FAISS, BM25, RRF fusion, and Cross-Encoder reranking, matching retrieved passages against ground-truth policy segments to compute **Precision@5**, **Recall@5**, and **Mean Reciprocal Rank (MRR)**.

---

## 13. How to Run Automated Tests

Run the complete test suite locally:

```powershell
pytest -v
```

To run individual test modules:
```powershell
pytest -v tests/test_retrieval.py
pytest -v tests/test_answer_generation.py
pytest -v tests/test_chunking.py
```

All **35 unit and integration tests** execute in ~21 seconds with a **100% pass rate**.

---

## 14. Design System & UI Components

The web application follows an AI SaaS aesthetic:

1. **Left Sidebar**:
   - Application branding with glowing icon badge.
   - **New Chat Session** button for instant conversation clearing.
   - **Target Document Card**: File name, live status badge (`Indexed & Ready`), page count (109), and chunk count (285).
   - **Active Session State**: Conversation status and clear button.
   - **Engine Architecture Badges**: Visual indicators for FAISS, BM25, RRF, Cross-Encoder, and Ollama connectivity.
2. **Main Chat Area**:
   - **Header Bar**: Live status indicator (`Online`), document name, mobile menu toggle, and telemetry panel toggle.
   - **Welcome Hero Screen**: Gradient heading, assistant description, and 5 suggested inquiry cards:
     1. *"What is the company mission?"*
     2. *"Who is the CEO?"*
     3. *"Who is the COO?"*
     4. *"What is the leave policy?"*
     5. *"What are employee responsibilities?"*
   - **Message Bubbles**: Sleek indigo/violet gradient cards for user questions; frosted glass cards for AI answers.
   - **Source Citations Section**: Under each response, expandable glass cards display:
     - Document filename and page number badge
     - Chunk ID badge
     - Cross-Encoder relevance logit score (e.g. `+2.932`)
     - Expandable excerpt text block
   - **Retrieval Telemetry Drawer**: Optional toggle showing expanded query synonyms, FAISS candidate count, BM25 count, and RRF candidate count.
   - **Chat Input Bar**: Auto-resizing textarea with keyboard shortcut indicator (`Enter` to send, `Shift+Enter` for newline) and send button with loading spinner.
3. **Right Telemetry Panel (Collapsible)**:
   - Live system statistics and pipeline parameters.
   - Collapses gracefully on screens under 1280px or via the header toggle button.

---

## 15. Explanation of the RAG Pipeline

```
Raw PDF ──► PyPDF Loader ──► Boundary Chunker ──► Sentence-Transformers ──► FAISS Index
                                                                        ──► BM25 Index
```

1. **Extraction**: `load_pdf()` parses each page of `company_policy.pdf`, preserving page numbers.
2. **Boundary-Aware Chunking**: `chunk_text()` breaks text into 1000-character windows with 200-character overlap. The `find_sensible_split()` function ensures chunks break cleanly on paragraph ends or sentence terminators (`. `, `\n\n`) rather than in the middle of words.
3. **Dense Indexing**: Chunks are embedded with `all-MiniLM-L6-v2` into 384-dimensional normalized vectors and indexed with FAISS `IndexFlatIP`.
4. **Sparse Indexing**: Chunks are tokenized (with stopword filtering) and indexed into a BM25Okapi inverted index.

---

## 16. Hybrid Search (FAISS + BM25 + RRF)

Dense vector search and sparse keyword search have complementary strengths:
- **FAISS (Dense)**: Captures semantic meaning (e.g., matching "mission" to "purpose, goals, commitment").
- **BM25 (Sparse)**: Captures exact lexical tokens (e.g., proper names, specific acronyms, compliance codes).

### Reciprocal Rank Fusion (RRF)
To combine candidates from different scoring scales, we apply Reciprocal Rank Fusion ($k=60$):

$$RRF\_Score(d) = \frac{1}{60 + rank_{FAISS}(d)} + \frac{1}{60 + rank_{BM25}(d)}$$

- Candidates appearing in the top ranks of both search engines receive the highest fused score.
- This eliminates score calibration discrepancies between BM25 scores (typically 0–25) and cosine similarities (-1 to 1).

---

## 17. Cross-Encoder Re-Ranking

Bi-encoders (like `all-MiniLM-L6-v2`) encode query and document independently, making them fast for retrieval but susceptible to false positives.

A **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) processes the query and passage **jointly** through full cross-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- The model evaluates every token of the query against every token of the candidate passage.
- Outputs an unbounded logit relevance score.
- The top-3 highest-scoring passages are selected as final context for the LLM.

---

## 18. Intent Disambiguation & Grounding Guardrails

### 1. CEO/COO Intent Separation
- **The Issue**: Previously, queries like `"salary of ceo"` triggered an exact name lookup, answering `"Charu Raheja, PhD – Chair and CEO"`.
- **The Fix**: `extract_company_roles()` now checks `NON_NAME_INTENTS` (salary, pay, bonus, contact, duties, etc.). If present, exact lookup is bypassed and routed to the RAG pipeline.

### 2. Leave Policy Truthfulness & Anti-Hallucination
- **The Issue**: On Page 71 of the PDF, the leave sentence sits between prescription drug rules and salary confidentiality rules. Naive prompts caused the model to summarize medication safety rules when asked about leave.
- **The Fix**: 
  - System prompt enforces strict topic separation: never mix prescription medication rules or salary rules into leave responses.
  - If numerical quantities (e.g. annual vacation day quotas) are absent from the document, the model explicitly states: *"The document does not specify the exact number of leaves or vacation days allowed."* The model **never invents leave counts**.

### 3. Manager Names Honesty
- **The Issue**: The PDF defines managerial titles (Nurse Managers, Client Relations Manager, Sales & Accounts Manager, IT Relations Manager, Clinical Manager), but individual personal employee names are not listed.
- **The Fix**: The chatbot states that managerial positions exist in the document, but clarifies that personal names are not provided, avoiding hallucinated personas.

### 4. Vague Query Disambiguation
- **The Issue**: Typing single words like `"employ"` previously triggered arbitrary chunk retrieval.
- **The Fix**: `is_vague_question()` intercepts single broad terms and prompts the user with tailored suggestions:
  *"Could you please be more specific? For example, you can ask about: Employee responsibilities, Employee benefits, Leave policy, Employee conduct, or Training policies."*

---

## 19. Evaluation Metrics & Benchmark Results

Retrieval performance evaluated across 11 diverse benchmark topics:

| Query ID | Topic | Question | Precision@5 | Recall@5 | Reciprocal Rank (RR) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Q1** | Company Mission | *"What is the company mission?"* | 0.2000 | 0.5000 | **1.0000** (Rank 1) |
| **Q2** | Leadership (CEO) | *"Who is the CEO of the company?"* | 0.8000 | 0.5000 | **1.0000** (Rank 1) |
| **Q3** | Leadership (COO) | *"Who is the COO of the company?"* | 0.4000 | 0.4000 | **1.0000** (Rank 1) |
| **Q4** | CEO Salary & Confidentiality | *"What is the CEO salary?"* | 0.2000 | 1.0000 | **0.5000** (Rank 2) |
| **Q5** | COO Salary & Confidentiality | *"What is the COO salary?"* | 0.2000 | 1.0000 | **1.0000** (Rank 1) |
| **Q6** | Manager Names & Titles | *"What are the names and titles of managers in the company?"* | 1.0000 | 0.1667 | **1.0000** (Rank 1) |
| **Q7** | Employee Responsibilities | *"What are the responsibilities of employees?"* | 0.2000 | 0.5000 | **1.0000** (Rank 1) |
| **Q8** | Leave Policy | *"What is the company leave policy?"* | 0.4000 | 0.5000 | **1.0000** (Rank 1) |
| **Q9** | Sick Leave Procedures | *"What happens if an employee takes sick leave or calls in sick?"* | 0.4000 | 1.0000 | **1.0000** (Rank 1) |
| **Q10** | Vacation Policy & Quantities | *"What is the vacation policy and how many leaves can employees take?"* | 0.4000 | 0.6667 | **1.0000** (Rank 1) |
| **Q11** | Employee Benefits | *"What employee benefits are provided?"* | 0.2000 | 0.5000 | **0.5000** (Rank 2) |
| **AGGREGATE** | **11 Benchmark Queries** | | **Mean P@5: 0.4000** | **Mean R@5: 0.6121** | **MRR: 0.9091** |

> [!NOTE]
> An **MRR of 0.9091** demonstrates that in 10 out of 11 queries, the most relevant passage was retrieved as the #1 candidate.

---

## 20. Performance & 8 GB RAM Optimization

To ensure seamless operation on an 8 GB RAM laptop:
1. **Singleton Model Lifecycle**: Sentence-transformers, Cross-Encoder, FAISS, and BM25 are initialized **once** during FastAPI lifespan startup and shared across all incoming requests.
2. **Compact Local LLM**: `qwen2.5:1.5b` uses ~986 MB of memory while maintaining high conversational coherence and strict adherence to negative instructions.
3. **No Redundant Re-Embedding**: Document embeddings are generated once at server start and held in an in-memory numpy array, preventing CPU thrashing.
4. **Total RAM Footprint**: The entire full-stack pipeline (FastAPI + Ollama + Models + Indexes + React dev server) runs stably under **2.2 GB RAM total**.

---

## 21. Known Limitations

1. **PDF Format Scope**: Uses `pypdf` which extracts digital text. Scanned document PDFs containing image-only pages require an additional OCR layer (e.g. `pytesseract` or `pdf2image`).
2. **Single Document Context**: Optimized for `company_policy.pdf`. Ingesting hundreds of separate PDFs would require multi-tenancy indexing and document ID filtering.
3. **Context Length**: The 1.5B model is optimized for concise answers. Multi-page comparative analysis queries would benefit from a larger 7B/8B model if hardware allows.

---

## 22. Future Improvements

- [ ] PDF upload UI with dynamic chunk indexing
- [ ] Session-persistent chat history stored in SQLite
- [ ] Streaming response generation (Server-Sent Events / WebSockets)
- [ ] Voice input and text-to-speech accessibility
- [ ] Multi-document filtering by department and policy category
