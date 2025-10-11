from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from .base import BaseConverter

class WebConverter(BaseConverter):
    def convert(self, source: str, metadata: Dict[str, Any] = None) -> List[Document]:
        if not source.startswith("http"):
            raise ValueError("URL không hợp lệ")

        loader = WebBaseLoader(
            source,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            },
        )
        docs = loader.load()

        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)

        return docs
