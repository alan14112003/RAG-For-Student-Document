from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class BaseConverter(ABC):
    @abstractmethod
    def convert(self, source: str, metadata: Dict[str, Any] = None) -> List[Document]:
        pass
