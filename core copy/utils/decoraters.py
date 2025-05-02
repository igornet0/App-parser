from functools import wraps

def ui_method(description: str = None, exposed: bool = True, **kwargs):
    """
    Универсальный декоратор для методов API
    
    :param description: Описание метода для UI (если None, будет взят из docstring)
    :param exposed: Должен ли метод быть видимым в интерфейсе
    """
    def decorator(func):
        # Сохраняем метаданные
        func._is_ui_method = True
        func._is_exposed = exposed
        func._ui_description = description or func.__doc__
        func._file_params = kwargs.get('file_params', {})
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator