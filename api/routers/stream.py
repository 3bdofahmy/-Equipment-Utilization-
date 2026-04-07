"""
api/routers/stream.py
──────────────────────
GET /stream/latest-frame — returns the most recent annotated JPEG frame.
Frontend polls this endpoint every N seconds.

The latest frame is stored in a module-level cache updated by cv_service.
"""

import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# In-memory cache — updated by cv_service via update_latest_frame()
_latest_frame: dict | None = None


def update_latest_frame(payload: dict) -> None:
    """Called by cv_service each time a new annotated frame is ready."""
    global _latest_frame
    _latest_frame = payload


@router.get("/latest-frame")
async def latest_frame():
    if _latest_frame is None:
        return JSONResponse(status_code=204, content=None)
    return JSONResponse(content=_latest_frame)
