"""
api/routers/model.py
─────────────────────
GET /model/info
GET /model/performance
GET /model/status
"""

from fastapi import APIRouter
from api.schemas import ModelInfoOut, ModelPerformanceOut, ModelStatusOut
from inference.registry import ModelRegistry

router = APIRouter()


@router.get("/info", response_model=ModelInfoOut)
async def model_info():
    return ModelRegistry.get_info()


@router.get("/performance", response_model=ModelPerformanceOut)
async def model_performance():
    return ModelRegistry.get_performance()


@router.get("/status", response_model=ModelStatusOut)
async def model_status():
    return ModelRegistry.get_status()
