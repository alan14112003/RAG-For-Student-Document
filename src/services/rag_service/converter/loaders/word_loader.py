from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from .base_loader import BaseFileLoader


class WordLoader(BaseFileLoader):
    def can_handle(self, ext: str) -> bool:
        return ext in [".doc", ".docx"]

    def load(self, path: str) -> list[Document]:
        return UnstructuredWordDocumentLoader(path).load()
