import os
from typing import List, Dict, Any, Type
from langchain_core.documents import Document
from .base import BaseConverter
from .loaders.base_loader import BaseFileLoader


class FileConverter(BaseConverter):
    _loaders: List[BaseFileLoader] = []

    @classmethod
    def register_loader(cls, loader_class: Type[BaseFileLoader]):
        cls._loaders.append(loader_class())

    def convert(self, source: str, metadata: Dict[str, Any] = None) -> List[Document]:
        if not os.path.exists(source):
            raise FileNotFoundError(f"File không tồn tại: {source}")

        ext = os.path.splitext(source)[1].lower()

        for loader in self._loaders:
            if loader.can_handle(ext):
                docs = loader.load(source)
                if metadata:
                    for doc in docs:
                        doc.metadata.update(metadata)
                return docs

        raise ValueError(f"Không có loader phù hợp cho định dạng: {ext}")
