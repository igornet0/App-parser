import re
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

class CryptoImpactModel(nn.Module):
    def __init__(self, ner_model_name='Babelscape/wikineural-multilingual-ner', 
                 sentiment_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'):
        super().__init__()
        
        # Модуль распознавания криптовалют (NER)
        self.ner_tokenizer = BertTokenizer.from_pretrained(ner_model_name)
        self.ner_model = BertModel.from_pretrained(ner_model_name)
        self.ner_classifier = nn.Linear(self.ner_model.config.hidden_size, 1)
        
        # Модуль анализа тональности
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        
        # Список криптовалют (обновляется через API)
        self.crypto_list = self.get_crypto_list()
    
    def get_crypto_list(self):
        """Получение актуального списка криптовалют через CoinGecko API"""
        import requests
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url)
        coins = response.json()
        crypto_dict = {}
        for coin in coins:
            name = coin['id'].lower()
            symbol = coin['symbol'].lower()
            crypto_dict[name] = name
            crypto_dict[symbol] = name
        return crypto_dict

    def extract_cryptocurrencies(self, text):
        """Извлечение криптовалют из текста"""
        inputs = self.ner_tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            embeddings = outputs.last_hidden_state
            logits = self.ner_classifier(embeddings)
            predictions = torch.sigmoid(logits).squeeze(-1) > 0.5
        
        # Собираем упоминания криптовалют
        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        crypto_mentions = set()
        current_entity = []
        
        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if pred and token not in ['[CLS]', '[SEP]', '[PAD]']:
                # Обработка субтокенов
                clean_token = token.replace('##', '')
                if clean_token:
                    current_entity.append(clean_token)
            elif current_entity:
                entity = ''.join(current_entity).lower()
                if entity in self.crypto_list:
                    crypto_mentions.add(self.crypto_list[entity])
                current_entity = []
        
        return list(crypto_mentions)

    def analyze_sentiment(self, text, cryptocurrencies):
        """Анализ влияния на каждую криптовалюту"""
        results = {}
        sentences = sent_tokenize(text)
        
        for crypto in cryptocurrencies:
            crypto_scores = []
            pattern = re.compile(rf'\b({crypto}|{crypto[:4]})\b', re.IGNORECASE)
            
            for sentence in sentences:
                if pattern.search(sentence):
                    inputs = self.sentiment_tokenizer(
                        sentence, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = self.sentiment_model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)[0]
                    
                    # Конвертируем в оценку от -100 до 100
                    sentiment_score = probs[0].item() * 100 - probs[1].item() * 100
                    crypto_scores.append(sentiment_score)
            
            if crypto_scores:
                avg_score = sum(crypto_scores) / len(crypto_scores)
                results[crypto] = round(avg_score, 1)
        
        return results

    def forward(self, text):
        """Полный процесс обработки текста"""
        cryptos = self.extract_cryptocurrencies(text)
        impacts = self.analyze_sentiment(text, cryptos)
        return impacts

# Пример использования
if __name__ == "__main__":
    # Инициализация модели
    model = CryptoImpactModel()
    
    # Пример новости
    news = """
    Bitcoin surged 10% after Tesla announced renewed support for cryptocurrency payments. 
    Meanwhile, Ethereum developers confirmed delays in the Shanghai upgrade, causing concern among investors.
    Dogecoin also showed unusual activity after Elon Musk's latest tweet.
    """
    
    # Анализ влияния
    impact_scores = model(news)
    
    print("Влияние новости на криптовалюты:")
    for crypto, score in impact_scores.items():
        print(f"{crypto}: {score}")