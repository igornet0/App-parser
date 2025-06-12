import asyncio

from backend.Dataset import DatasetTimeseries

from core.database import (db_helper, orm_get_coins,
                           orm_get_timeseries_by_coin, 
                           orm_get_data_timeseries,
                            orm_add_transaction,
                            orm_add_coin_portfolio
                           )

async def main():
    results = {}
    async with db_helper.get_session() as session:
        coins = await orm_get_coins(session)
        for coin in coins:
            timeseries = await orm_get_timeseries_by_coin(session, coin)
            # print(coin.name)
            for ts in timeseries:
                result = await orm_get_data_timeseries(session, ts.id)
                dt = DatasetTimeseries(result)
                results[coin.name] = dt
                # # print(len(result))
                # results[coin.name] = len(result)

    print(results)

if __name__ == "__main__":
    asyncio.run(main())