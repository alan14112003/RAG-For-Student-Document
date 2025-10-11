import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from .base import BaseConverter


class APIConverter(BaseConverter):
    def convert(self, source: str, metadata: Dict[str, Any] = None) -> List[Document]:
        if not source.startswith("http"):
            raise ValueError("API URL không hợp lệ")

        resp = requests.get(source, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"})
        resp.raise_for_status()

        content = resp.text
        try:
            data = resp.json()
            content = str(data)
        except Exception:
            pass

        doc = Document(
            page_content=content,
            metadata=metadata or {"source": source, "type": "api"}
        )
        return [doc]
