# Live Web RAG 🌐🤖

Real-time RAG (Retrieval-Augmented Generation) system that monitors dynamic web pages and answers questions about their current state.

## Features

- **Live DOM Monitoring** — Uses Selenium + MutationObserver to detect page changes in real-time
- **Smart Embeddings** — Only re-embeds content when meaningful changes occur
- **State-Aware RAG** — Tracks different page states (clicks, navigations) with metadata
- **Interactive Q&A** — Ask natural language questions about any webpage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Set up environment variables
cp .env.example .env  # Add your API keys

# Run
python main.py
```

## How It Works

```
Webpage → DOM Watcher → Knowledge Base → RAG Orchestrator → LLM Response
           (Selenium)    (ChromaDB)        (Your Question)
```

1. Navigate to any URL
2. Content is extracted and converted to RAG-optimized Markdown
3. Chunks are embedded and stored in a vector database
4. Ask questions — get answers based on the live page content

## Tech Stack

- **Browser Automation:** Selenium WebDriver
- **Vector Store:** ChromaDB
- **Embeddings & LLM:** NVIDIA AI Endpoints
- **HTML Processing:** [HTML-to-RAG-Optimized-Markdown](https://github.com/Gafoor2005/HTML-to-RAG-Optimized-Markdown) (MIT)

## License

MIT
