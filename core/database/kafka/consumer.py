import threading
from kafka import KafkaConsumer
from clickhouse_driver import Client
from modules.data_processor import process_market_data

# Конфигурация
CLICKHOUSE_HOST = 'clickhouse-server'
KAFKA_BROKERS = ['kafka-broker:9092']
TOPIC = 'price_updates'

class KafkaConsumerPriceUpdates:
    def __init__(self):
        self.ch_client = Client(host=CLICKHOUSE_HOST)
        self.consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BROKERS,
            group_id='agent-manager-group'
        )
        self.process_pool = []
    
    def fetch_data(self, symbol):
        """Получение обновленных данных из ClickHouse"""
        query = f"""
        SELECT price, volume, timestamp 
        FROM crypto.prices
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 100
        """
        return self.ch_client.execute(query)
    
    def process_update(self, symbol):
        """Обработка обновления для конкретного символа"""
        data = self.fetch_data(symbol)
        process_market_data(symbol, data)
    
    def start(self):
        """Основной цикл обработки сообщений"""
        for message in self.consumer:
            symbol = message.key.decode()
            timestamp = float(message.value.decode())
            
            # Запуск обработки в отдельном потоке
            thread = threading.Thread(
                target=self.process_update,
                args=(symbol,)
            )
            thread.start()
            self.process_pool.append(thread)
            
            # Очистка завершенных потоков
            self.process_pool = [t for t in self.process_pool if t.is_alive()]

if __name__ == "__main__":
    manager = KafkaConsumerPriceUpdates()
    manager.start()