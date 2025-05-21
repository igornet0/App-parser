from typing import Generator, List

class DataGenerator:
    def __init__(self, baths_loader: List[Generator], bath_size: int):
        self.baths_loader = baths_loader
        self.bath_size = bath_size

    def __iter__(self):
        for data in self.baths_loader:
            yield data

    def __len__(self):
        return self.bath_size