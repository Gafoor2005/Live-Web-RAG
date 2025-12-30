# Live Web RAG - Project Understanding

## Overview
This project implements a **Real-time RAG (Retrieval-Augmented Generation) system** that monitors dynamic web pages using Selenium and maintains a continuously updated knowledge base for LLM-powered Q&A.

## Core Problem Being Solved
Traditional RAG systems work with static documents. This system addresses the challenge of:
- **Dynamic web content** that changes frequently (e.g., dashboards, e-commerce sites, live feeds)
- **User interactions** (clicks, form submissions) that alter page state
- **Real-time context** for LLM queries based on the current state of a web page

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   DOMWatcher    │────▶│  KnowledgeBase   │────▶│  RAGOrchestrator   │
│   (Selenium)    │     │  (Vector Store)  │     │  (LLM Interface)   │
└─────────────────┘     └──────────────────┘     └────────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   - Monitor DOM            - Embed chunks           - User queries
   - Detect changes         - Upsert vectors         - Retrieve context
   - Extract content        - Prune stale data       - Generate answers
```

## Component Breakdown

### 1. DOMWatcher (dom_watcher.py)
**Purpose:** Manages Selenium browser automation and DOM change detection.

**Key Responsibilities:**
- Initialize and manage Selenium WebDriver
- Inject JavaScript MutationObserver to detect DOM changes
- Generate page fingerprints (hashes) to identify meaningful changes
- Convert HTML content to clean, structured Markdown for embedding
- Handle navigation, clicks, and page state transitions

**My Understanding:**
- This acts as the "eyes" of the system, watching the webpage
- Uses MutationObserver (JavaScript API) for efficient change detection
- Only triggers re-embedding when content meaningfully changes (using hash comparison)

### 2. KnowledgeBase (knowledge_base.py)
**Purpose:** Manages the vector database for storing and retrieving embedded content.

**Key Responsibilities:**
- Interface with vector DB (ChromaDB for local, Pinecone/Qdrant for production)
- Embed text chunks using an embedding model
- Upsert logic: add new or update existing vectors by source ID
- Prune outdated vectors when page state changes
- Support hybrid search (semantic + keyword)

**My Understanding:**
- Each chunk is tagged with metadata (timestamp, state_id, source_url)
- State-aware: knows which chunks belong to which "click state"
- Supports temporal queries ("What changed recently?")

### 3. RAGOrchestrator (rag_orchestrator.py)
**Purpose:** Bridges user queries with the knowledge base and LLM.

**Key Responsibilities:**
- Accept user natural language queries
- Retrieve relevant chunks from vector store
- Construct prompts with retrieved context
- Send to LLM and return generated responses
- Optional: Cache common query patterns

**My Understanding:**
- This is the user-facing interface
- Combines retrieval + generation in a single flow
- Context-aware: can include metadata about when/where content was captured

## Data Flow

1. **User navigates to a URL** → DOMWatcher loads the page
2. **Page content extracted** → Converted to Markdown chunks
3. **Chunks embedded** → Stored in KnowledgeBase with metadata
4. **DOM changes detected** → MutationObserver triggers update
5. **Content re-extracted** → Only changed sections re-embedded
6. **User asks question** → RAGOrchestrator retrieves relevant chunks
7. **LLM generates answer** → Response based on current page state

## Key Design Decisions (Please Confirm/Correct)

### Questions I Have:

1. **Single page or multi-page?**
   - Does the system monitor one page at a time, or crawl multiple pages?
   - Current assumption: Single page with state tracking

2. **User interaction scope:**
   - Should the system automatically click buttons/links, or just observe?
   - Or does the user manually interact while the system watches?

3. **Vector DB choice:**
   - ChromaDB (local, simple) vs Pinecone/Qdrant (cloud, scalable)?
   - Current plan: Start with ChromaDB for simplicity

4. **LLM provider:**
   - OpenAI API, local models (Ollama), or cloud alternatives?
   - Current plan: OpenAI API with option to swap

5. **Embedding model:**
   - OpenAI embeddings, or open-source (sentence-transformers)?
   - Current plan: OpenAI text-embedding-3-small

6. **Change detection granularity:**
   - Full page re-embed on any change, or incremental updates?
   - Current plan: Incremental (only changed sections)

7. **Real-time vs polling:**
   - Continuous MutationObserver or periodic polling?
   - Current plan: MutationObserver with polling fallback

## Tech Stack (Proposed)

| Component | Technology |
|-----------|------------|
| Browser Automation | Selenium WebDriver |
| Vector Database | ChromaDB (local) / Pinecone (prod) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-4 / GPT-4o |
| HTML to Markdown | Custom RAG-optimized converter (Node.js + jsdom) |
| Python Version | 3.10+ |

## File Structure (Proposed)

```
Live Web RAG/
├── src/
│   ├── __init__.py
│   ├── dom_watcher.py      # Selenium + DOM monitoring
│   ├── knowledge_base.py   # Vector store operations
│   ├── rag_orchestrator.py # Query + generation logic
│   ├── config.py           # Configuration settings
│   └── utils.py            # Helper functions
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── .env.example            # Environment variables template
└── PROJECT_UNDERSTANDING.md
```

