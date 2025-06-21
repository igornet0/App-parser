
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (User, Agent, Transaction, Portfolio, 
                                     AgentFeature,
                                     orm_get_train_agent,
                                     orm_delete_agent,
                                     orm_get_agent_by_name,
                                     orm_get_train_agents,
                                     orm_add_agent,
                                     orm_get_agents,
                                     orm_get_agent_by_id,
                                     orm_get_agents_type,
                                     orm_get_features,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio)

from backend.app.configuration import (Server, 
                                       rabbit,
                                       AgentType,
                                       TrainData,
                                       FeatureTypeResponse,
                                       AgentTrainResponse,
                                       AgentTypeResponse,
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

@router.post("/train_new_agent/", response_model=AgentResponse)
async def create_agent_train(agent_data: AgentTrainResponse, 
                       _: User = Depends(verify_authorization_admin), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_name(db, agent_data.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    if agent:
        raise HTTPException(status_code=400, detail="Agent with this name already exists") 
    
    try:
        agent, train_data = await orm_add_agent(db, agent_data)

        await rabbit.send_message("process_queue", {"id": train_data.id})

        return agent
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

@router.post("/delete_agent/{agent_id}")
async def delete_agent(agent_id: int,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agent = await orm_get_agent_by_id(db, agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        await orm_delete_agent(db, agent_id)

        return {"message": "Agent deleted successfully"}
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Agent delete failed: {str(e)}")

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
async def get_agents(status: str | None = None,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agents(db, status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/train_agents/", response_model=list[TrainData])
async def get_agents(status: str | None = None,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_train_agents(db, status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/train_agent/{id}", response_model=list[TrainData])
async def get_agents(id: int,
                    _: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        agents = await orm_get_agent_by_id(db, id=id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
    if not agents:
        raise HTTPException(status_code=404, detail="Agents not found")
    
    return agents

@router.get("/agents_types/", response_model=list[AgentTypeResponse])
async def get_agents_type(_: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        get_agents_type = await orm_get_agents_type(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agents type failed: {str(e)}")
    
    if not get_agents_type:
        raise HTTPException(status_code=404, detail="Agents type not found")
    
    return get_agents_type

@router.get("/available_features/", response_model=list[FeatureTypeResponse])
async def get_agent_features(_: User = Depends(verify_authorization_admin), 
                    db: AsyncSession = Depends(Server.get_db)):
    try:
        feuters = await orm_get_features(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Features failed: {str(e)}")
    
    if not feuters:
        raise HTTPException(status_code=404, detail="Features not found")
    
    return feuters

# @router.post("/trade/", response_model=list[AgentResponse])
# async def get_agent_trade(trade_data: AgentTrade,
#                     _: User = Depends(verify_authorization_admin), 
#                     db: AsyncSession = Depends(Server.get_db)):
#     try:
#         agents = await orm_get_agents(db)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Agents failed: {str(e)}")
    
#     if not agents:
#         raise HTTPException(status_code=404, detail="Agents not found")
    
#     return agents

@router.get("/health/rabbitmq")
async def rabbitmq_health():
    try:
        connection = await rabbit.get_connection()
        if connection.is_closed:
            return {"status": "down", "detail": "Connection closed"}
        return {"status": "up"}
    except Exception as e:
        return {"status": "down", "detail": str(e)}