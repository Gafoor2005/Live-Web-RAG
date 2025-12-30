"""
RAGOrchestrator - Query processing and response generation.

This module handles:
- User query processing
- Context retrieval from KnowledgeBase
- LLM prompt construction and response generation
- Response caching for common queries

Uses NVIDIA API for LLM inference.
"""

import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from openai import OpenAI

from .knowledge_base import KnowledgeBase


@dataclass
class RAGResponse:
    """Structured response from the RAG system."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    timestamp: float
    cached: bool = False


class RAGOrchestrator:
    """
    Orchestrates the RAG pipeline: query → retrieve → generate.
    
    Uses NVIDIA API for LLM inference (OpenAI-compatible endpoint).
    
    Attributes:
        knowledge_base: KnowledgeBase instance for retrieval
        llm_model: NVIDIA model to use for generation
        system_prompt: Base system prompt for the LLM
    """
    
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided web content context.

Rules:
1. Only answer based on the provided context. If the context doesn't contain relevant information, say so.
2. Be concise and direct in your answers.
3. If you quote specific information (prices, dates, names), indicate which part of the context it came from.
4. If the context contains outdated or conflicting information, mention this.
5. Format your response in a clear, readable way using markdown when appropriate."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_model: str = "meta/llama-3.1-8b-instruct",
        system_prompt: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl: float = 300  # 5 minutes
    ):
        """
        Initialize the RAGOrchestrator.
        
        Args:
            knowledge_base: KnowledgeBase instance for retrieval
            llm_model: NVIDIA model name (meta/llama-3.1-8b-instruct, etc.)
            system_prompt: Custom system prompt (uses default if None)
            nvidia_api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
            enable_cache: Whether to cache responses
            cache_ttl: Cache time-to-live in seconds
        """
        self.knowledge_base = knowledge_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Get API key
        api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required. Set it in .env or pass nvidia_api_key")
        
        # Initialize NVIDIA client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.NVIDIA_BASE_URL
        )
            
        # Simple in-memory cache
        self._cache: Dict[str, tuple[RAGResponse, float]] = {}
        
        print(f"[RAGOrchestrator] Initialized with NVIDIA model: {llm_model}")
        
    def _get_cache_key(self, query: str, n_results: int) -> str:
        """Generate a cache key for a query."""
        content = f"{query}:{n_results}:{self.knowledge_base.collection.count()}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _check_cache(self, cache_key: str) -> Optional[RAGResponse]:
        """Check if a valid cached response exists."""
        if not self.enable_cache:
            return None
            
        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                response.cached = True
                return response
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
                
        return None
        
    def _add_to_cache(self, cache_key: str, response: RAGResponse) -> None:
        """Add a response to the cache."""
        if self.enable_cache:
            self._cache[cache_key] = (response, time.time())
            
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        use_hybrid: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the knowledge base.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            use_hybrid: Use hybrid search (semantic + keyword)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant chunks with scores
        """
        if use_hybrid:
            results = self.knowledge_base.hybrid_search(query, n_results)
        else:
            results = self.knowledge_base.query(query, n_results, filter_metadata)
            
        return results
        
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            Formatted context string for the LLM
        """
        if not chunks:
            return "No relevant context found."
            
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            url = metadata.get('url', 'Unknown source')
            title = metadata.get('title', '')
            timestamp = metadata.get('timestamp', '')
            
            # Format timestamp if present
            time_str = ""
            if timestamp:
                try:
                    time_str = f" (captured: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(timestamp)))})"
                except:
                    pass
                    
            header = f"[Source {i}: {title or url}{time_str}]"
            context_parts.append(f"{header}\n{chunk['text']}")
            
        return "\n\n---\n\n".join(context_parts)
        
    def _build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Build the message list for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            List of message dicts for OpenAI API
        """
        user_message = f"""Context from web pages:

{context}

---

Question: {query}

Please answer based on the context above."""

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks
            temperature: LLM temperature (lower = more focused)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        context = self._build_context(context_chunks)
        messages = self._build_prompt(query, context)
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    def ask(
        self,
        query: str,
        n_results: int = 5,
        use_hybrid: bool = True,
        temperature: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve context and generate response.
        
        This is the main method for interacting with the RAG system.
        
        Args:
            query: User's natural language question
            n_results: Number of context chunks to retrieve
            use_hybrid: Use hybrid search
            temperature: LLM temperature
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Check cache first
        cache_key = self._get_cache_key(query, n_results)
        cached_response = self._check_cache(cache_key)
        if cached_response:
            print(f"[RAGOrchestrator] Cache hit for query")
            return cached_response
            
        print(f"[RAGOrchestrator] Processing query: {query[:50]}...")
        
        # Retrieve relevant chunks
        chunks = self.retrieve(query, n_results, use_hybrid, filter_metadata)
        
        if not chunks:
            return RAGResponse(
                answer="I couldn't find any relevant information in the current web content to answer your question.",
                sources=[],
                query=query,
                timestamp=time.time()
            )
            
        # Generate response
        answer = self.generate_response(query, chunks, temperature)
        
        # Build response
        response = RAGResponse(
            answer=answer,
            sources=[{
                'text': c['text'][:200] + '...' if len(c['text']) > 200 else c['text'],
                'url': c.get('metadata', {}).get('url', 'Unknown'),
                'score': c.get('combined_score', c.get('score', 0))
            } for c in chunks],
            query=query,
            timestamp=time.time()
        )
        
        # Cache the response
        self._add_to_cache(cache_key, response)
        
        return response
        
    def ask_with_history(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        n_results: int = 5
    ) -> RAGResponse:
        """
        Ask a question with conversation history for context.
        
        Args:
            query: Current question
            conversation_history: List of previous {"role": "user/assistant", "content": "..."}
            n_results: Number of context chunks
            
        Returns:
            RAGResponse
        """
        # Retrieve context for the current query
        chunks = self.retrieve(query, n_results)
        context = self._build_context(chunks)
        
        # Build messages with history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current query with context
        user_message = f"""New context from web pages:

{context}

---

Follow-up question: {query}"""

        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        return RAGResponse(
            answer=answer,
            sources=[{
                'text': c['text'][:200] + '...',
                'url': c.get('metadata', {}).get('url', 'Unknown'),
                'score': c.get('score', 0)
            } for c in chunks],
            query=query,
            timestamp=time.time()
        )
        
    def get_recent_changes(self, minutes: int = 5) -> str:
        """
        Get a summary of recent changes in the knowledge base.
        
        Args:
            minutes: How far back to look
            
        Returns:
            Summary of recent changes
        """
        cutoff = time.time() - (minutes * 60)
        
        # Get all recent chunks
        all_results = self.knowledge_base.collection.get(include=["metadatas", "documents"])
        
        recent_chunks = []
        for i, metadata in enumerate(all_results['metadatas'] or []):
            timestamp = float(metadata.get('timestamp', 0))
            if timestamp > cutoff:
                recent_chunks.append({
                    'text': all_results['documents'][i],
                    'metadata': metadata
                })
                
        if not recent_chunks:
            return f"No new content captured in the last {minutes} minutes."
            
        # Generate summary using LLM
        context = self._build_context(recent_chunks)
        
        summary_prompt = f"""Summarize the following recently captured web content in a few bullet points:

{context}

Provide a brief summary of what information was captured."""

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You summarize web content concisely."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        print("[RAGOrchestrator] Cache cleared")


# Example usage
if __name__ == "__main__":
    from .knowledge_base import KnowledgeBase
    
    # Initialize components
    kb = KnowledgeBase(collection_name="test_rag")
    rag = RAGOrchestrator(knowledge_base=kb)
    
    # Example query
    response = rag.ask("What products are on sale?")
    
    print(f"Answer: {response.answer}")
    print(f"\nSources used: {len(response.sources)}")
    for source in response.sources:
        print(f"  - {source['url']} (score: {source['score']:.3f})")
