from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .base_loader import BaseFileLoader


class PDFLoader(BaseFileLoader):
    def can_handle(self, ext: str) -> bool:
        return ext == ".pdf"

    def load(self, path: str) -> list[Document]:
        return PyPDFLoader(path).load()
