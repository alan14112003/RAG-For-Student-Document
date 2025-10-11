from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from .base_loader import BaseFileLoader


class TextFileLoader(BaseFileLoader):
    def can_handle(self, ext: str) -> bool:
        return ext in [".txt", ".md"]

    def load(self, path: str) -> list[Document]:
        return TextLoader(path).load()
