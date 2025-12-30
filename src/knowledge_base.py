"""
KnowledgeBase - Vector store management for RAG content.

This module handles:
- Text chunking and embedding
- Vector database operations (upsert, query, delete)
- Hybrid search (semantic + keyword)
- Metadata management and stale data pruning

Uses NVIDIA API for embeddings.
"""

import hashlib
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from openai import OpenAI


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class KnowledgeBase:
    """
    Manages the vector database for storing and retrieving embedded web content.
    
    Uses NVIDIA API for embeddings (OpenAI-compatible endpoint).
    
    Attributes:
        collection: ChromaDB collection instance
        embedding_model: Name of the embedding model to use
    """
    
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    
    def __init__(
        self,
        collection_name: str = "web_rag",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        nvidia_api_key: Optional[str] = None
    ):
        """
        Initialize the KnowledgeBase.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: NVIDIA embedding model name
            nvidia_api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)
        """
        self.embedding_model = embedding_model
        
        # Get API key
        api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required. Set it in .env or pass nvidia_api_key")
        
        # Initialize NVIDIA client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.NVIDIA_BASE_URL
        )
            
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"[KnowledgeBase] Initialized with NVIDIA embeddings ({embedding_model})")
        print(f"[KnowledgeBase] Collection '{collection_name}' has {self.collection.count()} existing chunks")
        
    def _generate_chunk_id(self, text: str, source_url: str, chunk_index: int = 0) -> str:
        """Generate a deterministic ID for a chunk based on content, source, and index."""
        # Use full text hash + chunk index for uniqueness
        content = f"{source_url}:{chunk_index}:{hashlib.md5(text.encode()).hexdigest()}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using NVIDIA API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Truncate text if too long
        text = text[:8000] if len(text) > 8000 else text
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"}
        )
        return response.data[0].embedding
        
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings
        
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Full text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        metadata = metadata or {}
        source_url = metadata.get("url", "unknown")
        
        # Simple chunking by character count with overlap
        # In production, consider sentence-aware chunking
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size * 0.5:  # Only if we're past halfway
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
                    
            if chunk_text.strip():  # Only add non-empty chunks
                chunk_id = self._generate_chunk_id(chunk_text, source_url, chunk_index)
                
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end
                }
                
                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    metadata=chunk_metadata
                ))
                
                chunk_index += 1
                
            start = end - overlap
            
        return chunks
        
    def sync_chunk(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Upsert a single chunk to the vector store.
        
        If a chunk with the same ID exists, it will be updated.
        
        Args:
            text: Chunk text content
            metadata: Chunk metadata (must include 'url')
            
        Returns:
            The chunk ID
        """
        source_url = metadata.get("url", "unknown")
        chunk_index = metadata.get("chunk_index", 0)
        chunk_id = self._generate_chunk_id(text, source_url, chunk_index)
        
        # Generate embedding
        embedding = self._embed_text(text)
        
        # Ensure metadata values are valid types for ChromaDB
        clean_metadata = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in metadata.items()
        }
        
        # Upsert to ChromaDB
        self.collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata]
        )
        
        print(f"[KnowledgeBase] Synced chunk {chunk_id[:8]}...")
        return chunk_id
        
    def sync_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Chunk and sync an entire document to the vector store.
        
        Args:
            text: Full document text
            metadata: Document metadata
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunk IDs
        """
        # First, prune any existing chunks from this URL
        source_url = metadata.get("url")
        if source_url:
            self.prune_by_url(source_url)
            
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size, overlap, metadata)
        
        if not chunks:
            print("[KnowledgeBase] No chunks to sync")
            return []
            
        # Deduplicate chunks by ID (keep first occurrence)
        seen_ids = set()
        unique_chunks = []
        for c in chunks:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique_chunks.append(c)
        chunks = unique_chunks
        
        if not chunks:
            print("[KnowledgeBase] No unique chunks to sync after deduplication")
            return []
            
        # Batch embed
        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)
        
        # Prepare for batch upsert
        ids = [c.id for c in chunks]
        metadatas = []
        for c in chunks:
            clean_meta = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in c.metadata.items()
            }
            metadatas.append(clean_meta)
            
        # Batch upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"[KnowledgeBase] Synced {len(chunks)} chunks from {source_url}")
        return ids
        
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (ChromaDB where clause)
            
        Returns:
            List of results with 'text', 'metadata', and 'score'
        """
        # Embed the query
        query_embedding = self._embed_text(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1 - results['distances'][0][i] if results['distances'] else 0  # Convert distance to similarity
                })
                
        return formatted
        
    def hybrid_search(
        self,
        query_text: str,
        n_results: int = 5,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        # Semantic search
        semantic_results = self.query(query_text, n_results=n_results * 2)
        
        # Simple keyword matching (can be enhanced with BM25)
        keywords = set(query_text.lower().split())
        
        for result in semantic_results:
            text_lower = result['text'].lower()
            keyword_score = sum(1 for kw in keywords if kw in text_lower) / len(keywords) if keywords else 0
            
            # Combine scores
            combined_score = (1 - keyword_weight) * result['score'] + keyword_weight * keyword_score
            result['combined_score'] = combined_score
            result['keyword_score'] = keyword_score
            
        # Sort by combined score
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return semantic_results[:n_results]
        
    def prune_by_url(self, url: str) -> int:
        """
        Remove all chunks from a specific URL.
        
        Args:
            url: Source URL to prune
            
        Returns:
            Number of chunks removed
        """
        # Query for chunks from this URL
        results = self.collection.get(
            where={"url": url},
            include=[]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"[KnowledgeBase] Pruned {len(results['ids'])} chunks from {url}")
            return len(results['ids'])
            
        return 0
        
    def prune_stale_vectors(self, max_age_seconds: float = 3600) -> int:
        """
        Remove vectors older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of chunks removed
        """
        cutoff_time = time.time() - max_age_seconds
        
        # Get all chunks (ChromaDB doesn't support < > operators well, so we fetch and filter)
        all_results = self.collection.get(include=["metadatas"])
        
        stale_ids = []
        for i, metadata in enumerate(all_results['metadatas'] or []):
            timestamp = float(metadata.get('timestamp', 0))
            if timestamp < cutoff_time:
                stale_ids.append(all_results['ids'][i])
                
        if stale_ids:
            self.collection.delete(ids=stale_ids)
            print(f"[KnowledgeBase] Pruned {len(stale_ids)} stale chunks")
            
        return len(stale_ids)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        count = self.collection.count()
        
        # Get unique URLs
        all_results = self.collection.get(include=["metadatas"])
        urls = set()
        for meta in (all_results['metadatas'] or []):
            if 'url' in meta:
                urls.add(meta['url'])
                
        return {
            "total_chunks": count,
            "unique_urls": len(urls),
            "urls": list(urls)
        }
        
    def clear(self) -> None:
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[KnowledgeBase] Collection cleared")


# Example usage
if __name__ == "__main__":
    # Initialize (requires NVIDIA_API_KEY environment variable)
    kb = KnowledgeBase(
        collection_name="test_rag",
        persist_directory="./test_chroma_db"
    )
    
    # Example document
    sample_text = """
    # Welcome to Our Store
    
    We have a wide variety of products available.
    
    ## Electronics
    Check out our latest smartphones, laptops, and accessories.
    The new iPhone 15 is now available for $999.
    
    ## Clothing
    Winter sale! 50% off all jackets and sweaters.
    New arrivals in our designer collection.
    """
    
    sample_metadata = {
        "url": "https://example-store.com",
        "title": "Example Store",
        "timestamp": time.time(),
        "state_id": "homepage_v1"
    }
    
    # Sync document
    chunk_ids = kb.sync_document(sample_text, sample_metadata)
    print(f"Created {len(chunk_ids)} chunks")
    
    # Query
    results = kb.query("What is the price of iPhone?")
    print("\nQuery results:")
    for r in results:
        print(f"  Score: {r['score']:.3f} - {r['text'][:100]}...")
        
    # Stats
    print(f"\nStats: {kb.get_stats()}")
