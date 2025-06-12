
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.orm_query import (User, Transaction, Portfolio, 
                                     orm_get_transactions_by_id,
                                     orm_get_user_transactions,
                                     orm_get_user_coin_transactions,
                                     orm_get_coin_portfolio)

from backend.app.configuration import (Server, 
                                       UserResponse,
                                       OrderUpdateAmount,
                                       OrderResponse,
                                       OrderCreate,
                                       OrderCancel,
                                       OrderType,
                                       verify_authorization)

http_bearer = HTTPBearer(auto_error=False)

# Инициализация роутера
router = APIRouter(
    prefix="/api_db_order",
    dependencies=[Depends(Server.http_bearer), Depends(verify_authorization)],
    tags=["Api db order"]
)

@router.post("/create_order/", response_model=OrderResponse)
async def create_order(order_data: OrderCreate, 
                       user: User = Depends(verify_authorization), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:
        orders = await orm_get_user_coin_transactions(db, user.id, order_data.coin_id, 
                                                      status="!cancel",
                                                      type_order=order_data.type)
        if orders:
            raise HTTPException(status_code=400, detail="Order already exists")

        if order_data.type == OrderType.BUY:
            total_cost = order_data.amount * order_data.price

            if user.balance < total_cost:
                raise HTTPException(status_code=400, detail=f"Insufficient funds {user.balance}")
            user.balance -= total_cost

        else:
            portfolio_item = await orm_get_coin_portfolio(db, user.id, order_data.coin_id)

            if not portfolio_item or portfolio_item.amount < order_data.amount:
                raise HTTPException(status_code=400, detail="Insufficient assets")
        
            # Блокируем активы
            portfolio_item.amount -= order_data.amount

        new_order = Transaction(
            user_id=user.id,
            coin_id=order_data.coin_id,
            type=order_data.type,
            amount_orig=order_data.amount,
            amount=order_data.amount,
            price=order_data.price,
        )

        db.add(new_order)

        await db.commit()  # Фиксируем изменения
        await db.refresh(new_order)  # Обновляем объект из БД

        return new_order
    
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")

@router.post("/update_order_amount/", response_model=OrderResponse)
async def cancel_order(order_data: OrderUpdateAmount, 
                       user: User = Depends(verify_authorization), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:
        order: Transaction = await orm_get_transactions_by_id(db, order_data.id,
                                                              status="open")

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if user.role != "admin" and order.user_id != user.id:
            raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
        if order.amount < order_data.amount:
            raise HTTPException(status_code=400, detail="Insufficient amount in order")
        
        order.amount -= order_data.amount

        coin_portfolio = await orm_get_coin_portfolio(db, user.id, order.coin_id)

        if not coin_portfolio:

            coin_portfolio = Portfolio(
                user_id=user.id,
                coin_id=order.coin_id,
                amount=0,
                price_avg=order.price,  # Можно установить цену, если нужно 224
            )
            db.add(coin_portfolio)

        if order.type == OrderType.BUY:
            coin_portfolio.price_avg = (coin_portfolio.price_avg * coin_portfolio.amount +
                                    order.price * order_data.amount) / \
                                    (coin_portfolio.amount + order_data.amount)
            coin_portfolio.amount += order_data.amount

        else:  # OrderType.SELL
            if coin_portfolio.amount + order.amount != 0:
                coin_portfolio.price_avg = (coin_portfolio.price_avg * (coin_portfolio.amount + order.amount_orig) -
                                order.price * order_data.amount) / \
                                (coin_portfolio.amount + order.amount)
            
            user.balance += order.price * order_data.amount
            
        if order.amount == 0:
            order.set_status(new_status="approve")
            if coin_portfolio.amount == 0:
                await db.delete(coin_portfolio)  # Удаляем портфель, если активов нет
                coin_portfolio = None

        await db.commit()  # Фиксируем изменения
        await db.refresh(order)  # Обновляем объект из БД
        if coin_portfolio:
            await db.refresh(coin_portfolio)  # Обновляем объект из БД

        return order
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Order Update failed: {str(e)}")

@router.post("/cancel_order/{id}", response_model=OrderResponse)
async def cancel_order(order: OrderCancel, 
                       user: User = Depends(verify_authorization), 
                       db: AsyncSession = Depends(Server.get_db)):
    try:
        order: Transaction = await orm_get_transactions_by_id(db, order.id)

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if order.status == "cancel":
            raise HTTPException(status_code=400, detail="Order already cancelled")
        elif order.status == "approve":
            raise HTTPException(status_code=400, detail="Order already approved")
        
        if user.role != "admin" and order.user_id != user.id:
            raise HTTPException(status_code=403, detail="You do not have permission to cancel this order")
        
        if order.type == OrderType.BUY:
            total_cost = order.amount * order.price
            user.balance += total_cost
        else:  # OrderType.SELL
            # Ищем актив в портфеле
            portfolio_item = await orm_get_coin_portfolio(db, user.id, order.coin_id)

            if not portfolio_item:
                raise HTTPException(status_code=400, detail="Insufficient assets")
            
            portfolio_item.amount += order.amount

        order.set_status(new_status="cancel")
        await db.commit()  # Фиксируем изменения
        await db.refresh(order)  # Обновляем объект из БД

        return order
    
    except Exception as e:
        await db.rollback()  # Откатываем при ошибке
        raise HTTPException(status_code=500, detail=f"Order cancellation failed: {str(e)}")
    
@router.get("/get_orders/", response_model=list[OrderResponse])
async def get_orders(user: User = Depends(verify_authorization), 
                     db: AsyncSession = Depends(Server.get_db)):
    try:
        
        orders = await orm_get_user_transactions(db, user.id)
        
        if not orders:
            raise HTTPException(status_code=404, detail="No orders found for this user")
        
        orders = sorted(orders, key=lambda x: x.created, reverse=True)

        return orders 
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {str(e)}")
    
