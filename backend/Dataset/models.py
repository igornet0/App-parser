class Coin:

    def __init__(self, name, time, open_price, close_price, max_price, min_price, value):
        self.name = name
        self.time = time

        self.open_price = open_price
        self.close_price = close_price
        
        self.max_price = max_price
        self.min_price = min_price
        
        self.value = value