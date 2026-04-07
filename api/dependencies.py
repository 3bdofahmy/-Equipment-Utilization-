"""
api/dependencies.py
────────────────────
Shared FastAPI dependencies injected into routers.
"""

from database.connection import get_db  # re-export for convenience

__all__ = ["get_db"]
