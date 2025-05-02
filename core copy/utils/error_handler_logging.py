from logging.handlers import RotatingFileHandler

import logging

# Ваша функция, которая будет вызываться при ошибке
def error_handler_function():
    print("Произошла ошибка! Выполняю специальную функцию...")
    # Здесь можно добавить любую логику, например, отправку уведомления или перезапуск парсера

# Кастомный обработчик
class ErrorHandlerImg(logging.Handler):
    def save_image(self, image):
        # Сохранение изображения
        pass

    def emit(self, record):
        if record.levelno >= logging.ERROR:  # Проверяем, является ли сообщение ошибкой
            error_handler_function()  # Вызываем вашу функцию