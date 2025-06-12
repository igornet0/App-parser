from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (Coin, User, Transaction, Portfolio, 
                                     orm_get_coins,
                                     orm_get_coin_by_id,
                                     orm_get_data_timeseries,
                                     paginate_coin_prices,
                                     orm_get_agents,
                                     orm_get_models,
                                     orm_get_transactions_by_id,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio)

from backend.app.configuration import (Server, 
                                       TimeLineCoin,
                                       CoinData,
                                       CoinResponse,
                                       CoinResponseData,
                                    #    CreateStrategyResponse,
                                    #    StrategyResponse,
                                       UserResponse,
                                       OrderUpdateAmount,
                                       OrderResponse,
                                       OrderCreate,
                                       OrderCancel,
                                       OrderType,
                                       verify_authorization)

# Инициализация роутера
router = APIRouter(
    prefix="/coins",
    dependencies=[Depends(Server.http_bearer), Depends(verify_authorization)],
    tags=["Strategy"]
)

@router.get("/get_coins/", response_model=list[CoinResponse])
async def get_coins(db: AsyncSession = Depends(Server.get_db)):
    try:
        coins = await orm_get_coins(db, parsed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coins: {str(e)}")

    if not coins:
        raise HTTPException(status_code=404, detail="No coins found")
    
    coins = sorted(coins, key=lambda x: x.price_now, reverse=True)

    return coins 
    
    
@router.get("/get_coin/", response_model=CoinResponseData)
async def get_coin_by_id(coin_id: int = Query(..., alias="coin_id"),
                        timeframe: str = Query("5m", alias="timeframe"),
                        size_page: int = Query(100, alias="size_page"),
                        last_timestamp: Optional[str] = Query(None, alias="last_timestamp"),
                        user: User = Depends(verify_authorization),
                        db: AsyncSession = Depends(Server.get_db)):
    
    time_line_coin = TimeLineCoin(coin_id=coin_id,
                                  timeframe=timeframe,
                                  last_timestamp=datetime.strptime(last_timestamp.split("+")[0], "%Y-%m-%dT%H:%M:%S") if last_timestamp else None,
                                  size_page=size_page)
    
    try:
        coin = await orm_get_coin_by_id(db, id=time_line_coin.coin_id, 
                                        parsed=user.role != "admin")
        records = await paginate_coin_prices(
            db, 
            coin_id=time_line_coin.coin_id, 
            timeframe=time_line_coin.timeframe,
            last_timestamp=time_line_coin.last_timestamp, 
            limit=time_line_coin.size_page,
            sort=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get coin: {str(e)}")

    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")
    
    if not records:
        raise HTTPException(status_code=404, detail="No records found")

    result = CoinResponseData(
        coin_id=time_line_coin.coin_id,
        price_now=coin.price_now,
        coin_data=list(map(lambda x: CoinData(
            datetime=x.datetime,
            open_price=x.open,
            close_price=x.close,
            max_price=x.max,
            min_price=x.min,
            volume=x.volume
        ), records)),
        last_timestamp=records[0].datetime
    )

    return result 
    

# @router.get("/get_agents/", response_model=list[AgentResponse])
# async def get_agents(db: AsyncSession = Depends(Server.get_db)):
#     try:
#         agents = await orm_get_agents(db, active=True)
        
#         if not agents:
#             raise HTTPException(status_code=404, detail="No agents found")
        
#         agents = sorted(agents, key=lambda x: x.version, reverse=True)

#         return agents 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")
    
# @router.get("/get_models/", response_model=list[ModelResponse])
# async def get_agents(db: AsyncSession = Depends(Server.get_db)):
#     try:
#         models = await orm_get_agents(db, active=True)
        
#         if not models:
#             raise HTTPException(status_code=404, detail="No agents found")
        
#         models = sorted(models, key=lambda x: int(x.version.replace(".", "")), reverse=True)

#         return models 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")

# @router.post("/create_strategy/", response_model=StrategyResponse)
# async def create_strategy(strategy_data: CreateStrategyResponse, 
#                        user: User = Depends(verify_authorization), 
#                        db: AsyncSession = Depends(Server.get_db)):
#     try:
#         orders = await orm_get_user_coin_transactions(db, user.id, order_data.coin_id, 
#                                                       status="!cancel",
#                                                       type_order=order_data.type)
#         if orders:
#             raise HTTPException(status_code=400, detail="Order already exists")

#         if order_data.type == OrderType.BUY:
#             total_cost = order_data.amount * order_data.price

#             if user.balance < total_cost:
#                 raise HTTPException(status_code=400, detail=f"Insufficient funds {user.balance}")
#             user.balance -= total_cost

#         else:
#             portfolio_item = await orm_get_coin_portfolio(db, user.id, order_data.coin_id)

#             if not portfolio_item or portfolio_item.amount < order_data.amount:
#                 raise HTTPException(status_code=400, detail="Insufficient assets")
        
#             # Блокируем активы
#             portfolio_item.amount -= order_data.amount

#         new_order = Transaction(
#             user_id=user.id,
#             coin_id=order_data.coin_id,
#             type=order_data.type,
#             amount_orig=order_data.amount,
#             amount=order_data.amount,
#             price=order_data.price,
#         )

#         db.add(new_order)

#         await db.commit()  # Фиксируем изменения
#         await db.refresh(new_order)  # Обновляем объект из БД

#         return new_order
    
#     except Exception as e:
#         await db.rollback()  # Откатываем при ошибке
#         raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")

# @router.post("/update_order_amount/", response_model=OrderResponse)
# async def cancel_order(order_data: OrderUpdateAmount, 
#                        user: User = Depends(verify_authorization), 
#                        db: AsyncSession = Depends(Server.get_db)):
#     try:
#         order: Transaction = await orm_get_transactions_by_id(db, order_data.id,
#                                                               status="open")

#         if not order:
#             raise HTTPException(status_code=404, detail="Order not found")
        
#         if user.role != "admin" and order.user_id != user.id:
#             raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
#         if order.amount < order_data.amount:
#             raise HTTPException(status_code=400, detail="Insufficient amount in order")
        
#         order.amount -= order_data.amount

#         coin_portfolio = await orm_get_coin_portfolio(db, user.id, order.coin_id)

#         if not coin_portfolio:

#             coin_portfolio = Portfolio(
#                 user_id=user.id,
#                 coin_id=order.coin_id,
#                 amount=0,
#                 price_avg=order.price,  # Можно установить цену, если нужно 224
#             )
#             db.add(coin_portfolio)

#         if order.type == OrderType.BUY:
#             coin_portfolio.price_avg = (coin_portfolio.price_avg * coin_portfolio.amount +
#                                     order.price * order_data.amount) / \
#                                     (coin_portfolio.amount + order_data.amount)
#             coin_portfolio.amount += order_data.amount

#         else:  # OrderType.SELL
#             if coin_portfolio.amount + order.amount != 0:
#                 coin_portfolio.price_avg = (coin_portfolio.price_avg * (coin_portfolio.amount + order.amount_orig) -
#                                 order.price * order_data.amount) / \
#                                 (coin_portfolio.amount + order.amount)
            
#             user.balance += order.price * order_data.amount
            
#         if order.amount == 0:
#             order.set_status(new_status="approve")
#             if coin_portfolio.amount == 0:
#                 await db.delete(coin_portfolio)  # Удаляем портфель, если активов нет
#                 coin_portfolio = None

#         await db.commit()  # Фиксируем изменения
#         await db.refresh(order)  # Обновляем объект из БД
#         if coin_portfolio:
#             await db.refresh(coin_portfolio)  # Обновляем объект из БД

#         return order
        
#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(status_code=500, detail=f"Order Update failed: {str(e)}")

# @router.post("/cancel_order/{id}", response_model=OrderResponse)
# async def cancel_order(order: OrderCancel, 
#                        user: User = Depends(verify_authorization), 
#                        db: AsyncSession = Depends(Server.get_db)):
#     try:
#         order: Transaction = await orm_get_transactions_by_id(db, order.id)

#         if not order:
#             raise HTTPException(status_code=404, detail="Order not found")
        
#         if order.status == "cancel":
#             raise HTTPException(status_code=400, detail="Order already cancelled")
#         elif order.status == "approve":
#             raise HTTPException(status_code=400, detail="Order already approved")
        
#         if user.role != "admin" and order.user_id != user.id:
#             raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
#         if order.type == OrderType.BUY:
#             total_cost = order.amount * order.price
#             user.balance += total_cost
#         else:  # OrderType.SELL
#             # Ищем актив в портфеле
#             portfolio_item = await orm_get_coin_portfolio(db, user.id, order.coin_id)

#             if not portfolio_item:
#                 raise HTTPException(status_code=400, detail="Insufficient assets")
            
#             portfolio_item.amount += order.amount

#         order.set_status(new_status="cancel")
#         await db.commit()  # Фиксируем изменения
#         await db.refresh(order)  # Обновляем объект из БД

#         return order
    
#     except Exception as e:
#         await db.rollback()  # Откатываем при ошибке
#         raise HTTPException(status_code=500, detail=f"Order cancellation failed: {str(e)}")
    
# @router.get("/get_orders/", response_model=list[OrderResponse])
# async def get_orders(user: User = Depends(verify_authorization), 
#                      db: AsyncSession = Depends(Server.get_db)):
#     try:
        
#         orders = await orm_get_user_transactions(db, user.id)
        
#         if not orders:
#             raise HTTPException(status_code=404, detail="No orders found for this user")
        
#         orders = sorted(orders, key=lambda x: x.created, reverse=True)

#         return orders 
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {str(e)}")
    
