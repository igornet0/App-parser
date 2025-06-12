from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

from core.database import Coin, Agent, Transaction
from .Box import Box

# Функция расчёта спроса
def calculate_demand(buys, sells, base_price):
    """Вычисляет спрос как отношение покупок к продажам."""
    if sells == 0:
        return 1.2  # высокий спрос 
    
    return max(0.8, min(1.5, (buys / sells) * base_price / 100))

class Exhange(Box):

    def __init__(self, coins:Union[List[Coin], Coin]):

        self.coins: list[Coin] = []
        self.set_coins(coins)

        self.trade_log = {}

        self.orders: dict[Coin,list[Transaction]] = {}

        self.history_volume: dict[Coin, list[float]] = {}
    
    def set_coins(self, coins:Union[List[Coin], Coin]):
        if isinstance(coins, Coin):
            self.coins = [coins]
        else:
            self.coins = coins

    def get_coins(self):
        return self.coins
    
    def cancel_order(self, order_cancel: Transaction):
        if order_cancel.order_type == 'buy':
            if self.buy_orders.get(order_cancel.coin, 0):
                for i, order in enumerate(self.buy_orders[order_cancel.coin]):
                    if order == order_cancel:
                        self.buy_orders[order.coin].pop(i)
                        break
                if not self.buy_orders[order_cancel.coin]:
                    del self.buy_orders[order_cancel.coin]

        elif order_cancel.order_type == 'sell':
            if self.sell_orders.get(order_cancel.coin, 0):
                for i, order in enumerate(self.sell_orders[order_cancel.coin]):
                    if order == order_cancel:
                        self.sell_orders[order.coin].pop(i)
                        break

                if not self.sell_orders[order_cancel.coin]:
                    del self.sell_orders[order_cancel.coin]
    
    def add_order(self, order: Transaction):
        """Добавляет ордер на биржу."""
        if not isinstance(order, Transaction):
            raise TypeError('Transaction must be of type Transaction')
        
        # self.logger["WARNING EXCHANGE"](order)

        if order.order_type == 'buy':
            self.buy_orders.setdefault(order.coin, []).append(order)

        elif order.order_type == 'sell':
            self.sell_orders.setdefault(order.coin, []).append(order)

    def match_orders(self):
        """Исполняет ордера, если возможно."""
        # print(f"{self.buy_orders=}")
        # print(f"{self.sell_orders=}")
        # Сортируем ордера: покупка — по убыванию цены, продажа — по возрастанию
        for coin in self.coins:
            if coin not in self.buy_orders or coin not in self.sell_orders:
                continue

            self.buy_orders[coin] = sorted(self.buy_orders[coin], 
                                                    key=lambda o: o.price, reverse=True)
            self.sell_orders[coin] = sorted(self.sell_orders[coin], 
                                                     key=lambda o: o.price)
            
            
            
            # print("1", self.buy_orders[coin], self.sell_orders[coin])
            while self.buy_orders[coin] and self.sell_orders[coin]:
                highest_buy = self.buy_orders[coin][0]
                lowest_sell = self.sell_orders[coin][0]

                # Если лучшая цена на покупку >= лучшей цены на продажу, исполняем сделку
                if highest_buy.price >= lowest_sell.price:
                    # Количество актива, которое может быть исполнено
                    quantity_traded = min(highest_buy.quantity, lowest_sell.quantity)
                    trade_price = (highest_buy.price + lowest_sell.price) / 2  # Средняя цена

                    # Обновляем балансы агентов
                    highest_buy.user.execute_trade('buy', coin, trade_price, quantity_traded)
                    lowest_sell.user.execute_trade('sell', coin, trade_price, quantity_traded)

                    # Уменьшаем количество в ордерах
                    highest_buy.quantity -= quantity_traded
                    lowest_sell.quantity -= quantity_traded

                    # Удаляем полностью исполненные ордера
                    if highest_buy.quantity == 0:
                        self.buy_orders[coin].pop(0)
                    if lowest_sell.quantity == 0:
                        self.sell_orders[coin].pop(0)
                    
                    # self.order_complete(highest_buy, lowest_sell, quantity_traded, trade_price)
                    # Обновляем текущую цену
                    current_price = trade_price
                    # print("2", self.buy_orders[coin], self.sell_orders[coin])
                    # input(">>>")
                else:
                    break

                # Записываем историю цен
                
                coin.set_price(current_price)

    def decision(self, agent: Agent, coin: Coin, count, decision):
        if decision in ['buy', 'sell']:
            result = agent.create_order(coin, count, decision)
            if result[0]:
                order = result[1]
                self.add_order(order)
                self.match_orders()

            return result

        elif decision == 'hold':
            return True, f"Hold {agent.username}: {coin.name} - {coin.get_price()}$"
        else:
            return False, f"Invalid decision - {decision}"  

    def print_orders(self):
        for coin in self.coins:
            if self.buy_orders.get(coin):
                for order in self.buy_orders[coin]:
                    self.logger["INFO EXCHANGE"](order)

            if self.sell_orders.get(coin):
                for order in self.sell_orders[coin]:  
                    self.logger["INFO EXCHANGE"](order)
    
    def plot_history_coin(self, coin: Coin | None = None):
        if not coin:
            for coin in self.coins:
                coin.plot_history()
        else:
            coin.plot_history()

    def plot_history_volume_coin(self, coin: Coin | None = None):
        if not coin:
            for coin in self.coins:
                self.plot_history_volume_coin(coin)
        else:
            if not self.history_volume.get(coin, 0):
                return
            
            plt.plot(self.history_volume[coin])
            plt.title(f"{coin.name} volume")
            plt.show()

    def print_coins(self):
        for coin in self.coins:
            print(coin)