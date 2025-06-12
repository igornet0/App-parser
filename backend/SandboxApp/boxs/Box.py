from abc import ABC, abstractmethod

class Box(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass