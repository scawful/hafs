"""UI Registry for HAFS.

Allows plugins to register new pages and widgets for the Web Hub.
"""
from typing import Callable, Dict

class PageRegistry:
    def __init__(self):
        self.pages: Dict[str, Callable] = {}
        
    def register_page(self, name: str, render_fn: Callable):
        """Register a new page to be displayed in the sidebar."""
        self.pages[name] = render_fn
        print(f"[UI Registry] Registered page: {name}")

# Global singleton
ui_registry = PageRegistry()
