"""
api/routers/equipment.py
─────────────────────────
GET /equipment
GET /equipment/{equipment_id}
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas import EquipmentOut
from api.dependencies import get_db
from database.repository import equipment_repo

router = APIRouter()


@router.get("", response_model=list[EquipmentOut])
async def list_equipment(db: AsyncSession = Depends(get_db)):
    return await equipment_repo.get_all(db)


@router.get("/{equipment_id}", response_model=EquipmentOut)
async def get_equipment(equipment_id: str, db: AsyncSession = Depends(get_db)):
    eq = await equipment_repo.get_by_id(db, equipment_id)
    if not eq:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return eq
