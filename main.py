"""
Live Web RAG - Main Entry Point

This script demonstrates the full RAG pipeline:
1. Start browser and navigate to a URL
2. Extract and embed page content
3. Watch for DOM changes
4. Answer questions about the page content
"""

import sys
import time
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.dom_watcher import DOMWatcher
from src.knowledge_base import KnowledgeBase
from src.rag_orchestrator import RAGOrchestrator
from src.config import Config


class LiveWebRAG:
    """
    Main class that ties together all components of the Live Web RAG system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Live Web RAG system.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config.from_env()
        
        # Initialize components
        self.knowledge_base = KnowledgeBase(
            collection_name=self.config.vector_db.collection_name,
            persist_directory=self.config.vector_db.persist_directory,
            embedding_model=self.config.embedding.model,
            nvidia_api_key=self.config.nvidia_api_key
        )
        
        self.rag = RAGOrchestrator(
            knowledge_base=self.knowledge_base,
            llm_model=self.config.llm.model,
            nvidia_api_key=self.config.nvidia_api_key,
            enable_cache=self.config.cache.enabled,
            cache_ttl=self.config.cache.ttl_seconds
        )
        
        self.watcher = DOMWatcher(
            headless=self.config.selenium.headless,
            on_change_callback=self._on_dom_change
        )
        
        self._is_running = False
        
    def _on_dom_change(self, markdown: str, metadata: dict) -> None:
        """Callback when DOM changes are detected."""
        print(f"\n[LiveWebRAG] DOM change detected at {metadata.get('url', 'unknown')}")
        
        # Sync to knowledge base
        chunk_ids = self.knowledge_base.sync_document(
            text=markdown,
            metadata=metadata,
            chunk_size=self.config.embedding.chunk_size,
            overlap=self.config.embedding.chunk_overlap
        )
        
        print(f"[LiveWebRAG] Synced {len(chunk_ids)} chunks to knowledge base")
        
    def start(self, url: str) -> None:
        """
        Start the RAG system with a target URL.
        
        Args:
            url: URL to monitor
        """
        print(f"\n{'='*60}")
        print("Live Web RAG System Starting")
        print(f"{'='*60}")
        
        self.watcher.start()
        self.watcher.navigate(url)
        self._is_running = True
        
        print(f"\n✓ Browser started and navigated to: {url}")
        print(f"✓ Knowledge base ready ({self.knowledge_base.get_stats()['total_chunks']} chunks)")
        print(f"\nReady to answer questions! Type 'quit' to exit.\n")
        
    def stop(self) -> None:
        """Stop the RAG system and cleanup resources."""
        self._is_running = False
        self.watcher.stop()
        print("\n[LiveWebRAG] System stopped")
        
    def ask(self, question: str) -> str:
        """
        Ask a question about the current page content.
        
        Args:
            question: Natural language question
            
        Returns:
            Generated answer
        """
        response = self.rag.ask(question)
        return response.answer
        
    def interactive_loop(self) -> None:
        """Run an interactive question-answer loop."""
        conversation_history = []
        
        while self._is_running:
            try:
                # Check for DOM changes periodically
                self.watcher.check_for_changes()
                
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if user_input.lower() == 'refresh':
                    print("Refreshing page content...")
                    self.watcher.navigate(self.watcher.current_url)
                    continue
                    
                if user_input.lower() == 'stats':
                    stats = self.knowledge_base.get_stats()
                    print(f"\nKnowledge Base Stats:")
                    print(f"  Total chunks: {stats['total_chunks']}")
                    print(f"  Unique URLs: {stats['unique_urls']}")
                    continue
                    
                if user_input.lower().startswith('goto '):
                    new_url = user_input[5:].strip()
                    print(f"Navigating to: {new_url}")
                    self.watcher.navigate(new_url)
                    continue
                    
                # Ask the question
                print("\nAssistant: ", end="", flush=True)
                
                if conversation_history:
                    response = self.rag.ask_with_history(
                        user_input, 
                        conversation_history
                    )
                else:
                    response = self.rag.ask(user_input)
                    
                print(response.answer)
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response.answer})
                
                # Keep history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                    
                # Show sources if available
                if response.sources:
                    print(f"\n  [Based on {len(response.sources)} source(s)]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
                
        self.stop()


def main():
    """Main entry point."""
    # Default URL to start with
    default_url = "https://gafoor.dev"
    default_url = "https://soundcloud.com/search?q=into dust"
    
    # Get URL from command line if provided
    url = sys.argv[1] if len(sys.argv) > 1 else default_url
    
    # Create and start the system
    rag_system = LiveWebRAG()
    
    try:
        rag_system.start(url)
        rag_system.interactive_loop()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag_system.stop()


if __name__ == "__main__":
    main()
