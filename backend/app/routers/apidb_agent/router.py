
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (User, Agent, Transaction, Portfolio, 
                                     orm_get_transactions_by_id,
                                     orm_get_agents,
                                     orm_get_agent_by_id,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio)

from backend.app.configuration import (Server, 
                                       AgentResponse,
                                       AgentCreate,
                                       AgentTrade,
                                       UserResponse,
                                       OrderUpdateAmount,
                                       OrderResponse,
                                       OrderCreate,
                                       OrderCancel,
                                       OrderType,
                                       verify_authorization,
                                       verify_authorization_admin)

http_bearer = HTTPBearer(auto_error=False)

# Инициализация роутера
router = APIRouter(
    prefix="/api_db_agent",
    dependencies=[Depends(http_bearer)],
    tags=["Api db agent"]
)

@router.post("/create_agent/", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate, 
                       _: User = Depends(verify_authorization_admin), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:

        agent = Agent(
            name = agent_data.name,
            type = agent_data.type,
            path_model = agent_data.path_model,
            a_conficent = agent_data.a_conficent,
            active = agent_data.active,
            version = agent_data.version
        )

        db.add(agent)
        db.refresh(agent)

        await db.commit()  # Фиксируем изменения

        return agent
    
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")


@router.post("/agent/{id}", response_model=AgentResponse)
async def get_agent_by_id(agent_data: AgentResponse, 
                    _: User = Depends(verify_authorization), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_id(db, agent_data.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent

@router.get("/agents/", response_model=list[AgentResponse])
async def get_agents(_: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agents(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.post("/trade/", response_model=list[AgentResponse])
async def get_agent_trade(trade_data: AgentTrade,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agents(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents