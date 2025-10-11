from .file_converter import FileConverter
from .web_converter import WebConverter
from .api_converter import APIConverter
from .converter_factory import ConverterFactory

from .loaders.pdf_loader import PDFLoader
from .loaders.text_loader import TextFileLoader
from .loaders.word_loader import WordLoader
from .loaders.csv_loader import CSVLoader

FileConverter.register_loader(PDFLoader)
FileConverter.register_loader(TextFileLoader)
FileConverter.register_loader(WordLoader)
FileConverter.register_loader(CSVLoader)

ConverterFactory.register("file", FileConverter)
ConverterFactory.register("web", WebConverter)
ConverterFactory.register("api", APIConverter)
