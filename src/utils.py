"""
Utility functions for Live Web RAG system.
"""

import re
import hashlib
from typing import List, Dict, Any
from urllib.parse import urlparse


def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that don't add meaning
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'\"\(\)\[\]\{\}]', '', text)
    
    return text.strip()


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc


def generate_state_id(url: str, timestamp: float) -> str:
    """Generate a unique state ID for a page snapshot."""
    content = f"{url}:{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def chunk_by_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks by sentences, respecting max size.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    """
    Format source information for display to user.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources available."
        
    lines = ["**Sources:**"]
    for i, source in enumerate(sources, 1):
        url = source.get('url', 'Unknown')
        score = source.get('score', 0)
        lines.append(f"{i}. [{extract_domain(url)}]({url}) (relevance: {score:.1%})")
        
    return "\n".join(lines)


def is_meaningful_change(old_hash: str, new_hash: str, old_length: int, new_length: int) -> bool:
    """
    Determine if a DOM change is meaningful enough to re-embed.
    
    Args:
        old_hash: Previous content hash
        new_hash: New content hash
        old_length: Previous content length
        new_length: New content length
        
    Returns:
        True if change is significant
    """
    if old_hash != new_hash:
        # Check if length changed significantly (more than 5%)
        if old_length == 0:
            return True
        change_ratio = abs(new_length - old_length) / old_length
        return change_ratio > 0.05 or new_hash != old_hash
        
    return False
