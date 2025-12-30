"""
DOMWatcher - Selenium-based DOM monitoring and content extraction.

This module handles:
- Browser automation via Selenium
- DOM change detection using MutationObserver
- Content extraction and conversion to RAG-optimized Markdown
"""

import hashlib
import time
import subprocess
import os
from typing import Optional, Callable, List, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class DOMWatcher:
    """
    Watches a web page for DOM changes and extracts content for RAG ingestion.
    
    Attributes:
        driver: Selenium WebDriver instance
        current_url: Currently loaded URL
        last_hash: Hash of the last captured DOM state
        on_change_callback: Function called when DOM changes are detected
    """
    
    def __init__(
        self,
        headless: bool = False,
        on_change_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the DOMWatcher with a Selenium WebDriver.
        
        Args:
            headless: Run browser in headless mode
            on_change_callback: Callback function when DOM changes (receives markdown, metadata)
        """
        self.driver: Optional[webdriver.Chrome] = None
        self.current_url: Optional[str] = None
        self.last_hash: Optional[str] = None
        self.on_change_callback = on_change_callback
        self.headless = headless
        self._mutation_detected = False
        
    def start(self) -> None:
        """Initialize and start the Selenium WebDriver."""
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=options)
        print("[DOMWatcher] Browser started")
        
    def stop(self) -> None:
        """Close the browser and cleanup resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("[DOMWatcher] Browser closed")
            
    def navigate(self, url: str, wait_for_load: bool = True) -> None:
        """
        Navigate to a URL and optionally wait for page load.
        
        Args:
            url: The URL to navigate to
            wait_for_load: Wait for page to fully load
        """
        if not self.driver:
            raise RuntimeError("Driver not started. Call start() first.")
            
        self.driver.get(url)
        self.current_url = url
        
        if wait_for_load:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
        # Inject MutationObserver after page load
        self._inject_mutation_observer()
        
        # Capture initial state
        self._capture_and_notify()
        print(f"[DOMWatcher] Navigated to: {url}")
        
    def _inject_mutation_observer(self) -> None:
        """Inject JavaScript MutationObserver to detect DOM changes."""
        observer_script = """
        if (!window.__ragMutationObserver) {
            window.__ragDomChanged = false;
            window.__ragMutationObserver = new MutationObserver((mutations) => {
                // Filter out trivial changes
                const significantChange = mutations.some(m => 
                    m.type === 'childList' && (m.addedNodes.length > 0 || m.removedNodes.length > 0) ||
                    m.type === 'characterData'
                );
                if (significantChange) {
                    window.__ragDomChanged = true;
                }
            });
            
            window.__ragMutationObserver.observe(document.body, {
                childList: true,
                subtree: true,
                characterData: true
            });
            console.log('[RAG] MutationObserver injected');
        }
        """
        self.driver.execute_script(observer_script)
        
    def check_for_changes(self) -> bool:
        """
        Check if DOM has changed since last check.
        
        Returns:
            True if changes detected, False otherwise
        """
        if not self.driver:
            return False
            
        # Check MutationObserver flag
        changed = self.driver.execute_script("return window.__ragDomChanged || false;")
        
        if changed:
            # Reset the flag
            self.driver.execute_script("window.__ragDomChanged = false;")
            
            # Verify with hash comparison
            current_hash = self.get_page_hash()
            if current_hash != self.last_hash:
                self._capture_and_notify()
                return True
                
        return False
        
    def get_page_hash(self) -> str:
        """
        Generate a hash fingerprint of the current DOM state.
        
        Returns:
            MD5 hash of the page's main content
        """
        if not self.driver:
            return ""
            
        # Get main content, excluding dynamic elements like timestamps
        content = self.driver.execute_script("""
            const clone = document.body.cloneNode(true);
            // Remove elements that change frequently but aren't meaningful
            clone.querySelectorAll('script, style, noscript, iframe').forEach(el => el.remove());
            return clone.innerText;
        """)
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
        
    def capture_semantic_markdown(self) -> tuple[str, Dict[str, Any]]:
        """
        Extract current page content and convert to RAG-optimized Markdown.
        
        Uses the HTML-to-RAG-Optimized-Markdown converter which preserves
        identifiers (class, id, aria attributes) for interactive elements.
        
        Returns:
            Tuple of (markdown_content, metadata_dict)
        """
        if not self.driver:
            raise RuntimeError("Driver not started.")
            
        # Get the full HTML including semantic elements
        html_content = self.driver.execute_script("""
            return document.documentElement.outerHTML;
        """)
        
        # Convert using the RAG-optimized markdown converter (Node.js script)
        markdown = self._convert_html_to_rag_markdown(html_content)
        
        # Build metadata
        metadata = {
            "url": self.current_url,
            "title": self.driver.title,
            "timestamp": time.time(),
            "hash": self.get_page_hash(),
            "state_id": f"{self.current_url}_{int(time.time())}"
        }
        
        return markdown.strip(), metadata
    
    def _convert_html_to_rag_markdown(self, html_content: str) -> str:
        """
        Convert HTML to RAG-optimized Markdown using Node.js converter.
        
        This converter preserves semantic structure and interactive element
        identifiers (class, id, aria attributes, href, etc.) which is crucial
        for enabling interactions based on RAG context.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            RAG-optimized Markdown string
        """
        # Get path to the JS converter script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        converter_path = os.path.join(script_dir, "html_to_rag_markdown.js")
        
        try:
            # Run Node.js converter, passing HTML via stdin
            result = subprocess.run(
                ["node", converter_path],
                input=html_content,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"[DOMWatcher] Converter error: {result.stderr}")
                # Fallback to basic text extraction
                return self._fallback_text_extraction()
                
            return result.stdout
            
        except subprocess.TimeoutExpired:
            print("[DOMWatcher] Converter timed out")
            return self._fallback_text_extraction()
        except FileNotFoundError:
            print("[DOMWatcher] Node.js not found. Please install Node.js.")
            return self._fallback_text_extraction()
        except Exception as e:
            print(f"[DOMWatcher] Converter exception: {e}")
            return self._fallback_text_extraction()
    
    def _fallback_text_extraction(self) -> str:
        """
        Fallback method for text extraction when JS converter fails.
        
        Returns:
            Basic text content from the page
        """
        if not self.driver:
            return ""
        
        try:
            return self.driver.execute_script("""
                const clone = document.body.cloneNode(true);
                clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                return clone.innerText;
            """)
        except Exception:
            return ""
        
    def _capture_and_notify(self) -> None:
        """Capture content and notify via callback if registered."""
        markdown, metadata = self.capture_semantic_markdown()
        self.last_hash = metadata["hash"]
        
        if self.on_change_callback:
            self.on_change_callback(markdown, metadata)
            
    def click_element(self, selector: str, by: By = By.CSS_SELECTOR) -> bool:
        """
        Click an element and wait for any resulting DOM changes.
        
        Args:
            selector: Element selector
            by: Selector type (CSS, XPATH, etc.)
            
        Returns:
            True if click successful, False otherwise
        """
        if not self.driver:
            return False
            
        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((by, selector))
            )
            element.click()
            
            # Wait a moment for potential DOM updates
            time.sleep(0.5)
            
            # Check and process any changes
            self.check_for_changes()
            
            return True
        except Exception as e:
            print(f"[DOMWatcher] Click failed: {e}")
            return False
            
    def watch_loop(self, interval: float = 1.0, duration: Optional[float] = None) -> None:
        """
        Continuously watch for DOM changes.
        
        Args:
            interval: Seconds between checks
            duration: Total duration to watch (None = indefinite)
        """
        start_time = time.time()
        print(f"[DOMWatcher] Starting watch loop (interval: {interval}s)")
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                    
                self.check_for_changes()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("[DOMWatcher] Watch loop stopped by user")
            
    def get_page_sections(self) -> List[Dict[str, str]]:
        """
        Extract page content as logical sections for chunking.
        
        Returns:
            List of dicts with 'heading' and 'content' keys
        """
        if not self.driver:
            return []
            
        sections = self.driver.execute_script("""
            const sections = [];
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            
            headings.forEach((heading, index) => {
                let content = '';
                let sibling = heading.nextElementSibling;
                
                while (sibling && !sibling.matches('h1, h2, h3, h4, h5, h6')) {
                    content += sibling.innerText + '\\n';
                    sibling = sibling.nextElementSibling;
                }
                
                sections.push({
                    heading: heading.innerText,
                    level: parseInt(heading.tagName.charAt(1)),
                    content: content.trim()
                });
            });
            
            return sections;
        """)
        
        return sections


# Example usage
if __name__ == "__main__":
    def on_change(markdown: str, metadata: dict):
        print(f"\n{'='*50}")
        print(f"[CHANGE DETECTED] {metadata['title']}")
        print(f"URL: {metadata['url']}")
        print(f"Hash: {metadata['hash']}")
        print(f"Content preview: {markdown[:200]}...")
        print(f"{'='*50}\n")
    
    watcher = DOMWatcher(headless=False, on_change_callback=on_change)
    
    try:
        watcher.start()
        watcher.navigate("https://gafoor.dev")
        
        # Watch for 30 seconds
        watcher.watch_loop(interval=2.0, duration=30)
        
    finally:
        watcher.stop()
