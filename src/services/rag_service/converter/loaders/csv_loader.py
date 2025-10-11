import csv
from langchain_core.documents import Document
from .base_loader import BaseFileLoader


class CSVLoader(BaseFileLoader):
    def can_handle(self, ext: str) -> bool:
        return ext == ".csv"

    def load(self, path: str) -> list[Document]:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            content = "\n".join([", ".join(row) for row in reader])
        return [Document(page_content=content, metadata={"source": path})]
