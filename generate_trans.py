import asyncio
from core.database import db_helper
from core.database.orm_query import (orm_get_user_by_login,
                                     orm_add_transaction,
                                     orm_add_coin_portfolio,
                                     orm_update_transaction_amount,
                                     orm_get_user_coin_transactions,
                                     orm_update_transaction_status,
                                     orm_get_coins)

class UserLoginResponse:
    login = "CEO_AGENT"
    password = None

#120000000

async def test_create_transaction(session, user, coin):
    price = coin.price_now
    # amount 
    await orm_add_transaction(session, user.id, coin.id, amount, price)
    await orm_add_coin_portfolio(session, user.id, coin.id, 1)

async def main():
    async with db_helper.get_session() as session:
        user = await orm_get_user_by_login(session, UserLoginResponse())
        coins = await orm_get_coins(session)
        for i, coin in enumerate(coins):

            # await orm_add_coin_portfolio(session, user.user_id, coin.id, i + 1)
            # if i != 0:
            #     await orm_add_transaction(session, user.id, coin.id, i+1, coin.price_now - (coin.price_now * 0.05))
                
            new_transactions = await orm_get_user_coin_transactions(session, user.id, coin.id, "new")
            cancel_transactions = await orm_get_user_coin_transactions(session, user.id, coin.id, "cancel")
            approve_transactions = await orm_get_user_coin_transactions(session, user.id, coin.id, "approve")

            assert not new_transactions

            # if i % 2 == 0:
            #     for coin, data in transactions.items():
            #         await orm_update_transaction_amount(session, data["id"], 0)
            #         await orm_add_coin_portfolio(session, user.id, coin.id, data["amount"])

            # else:
            #      for coin, data in transactions.items():
            #         await orm_update_transaction_status(session, data["id"], "cancel")
                
            
            # transaction = transactions[-1]
            # await orm_update_transaction_status(session, transaction.id, "open")
            count = 0

            for coin, data in cancel_transactions.items():
                # print(coin, data)
                count += 1

            print(count)

            count = 0
            summa = 0
            for coin, data in approve_transactions.items():
                # print(coin, data)
                count += 1
                summa += data["amount"] * data["price"]

            print(count)
            print(summa)




if __name__ == "__main__":
    asyncio.run(main())