from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseFileLoader(ABC):
    @abstractmethod
    def can_handle(self, ext: str) -> bool:
        pass

    @abstractmethod
    def load(self, path: str) -> List[Document]:
        pass
