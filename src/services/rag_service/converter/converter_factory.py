from typing import Type, Dict, Literal
from .base import BaseConverter

ConverterType = Literal["file", "web", "api"]

class ConverterFactory:
    _registry: Dict[str, Type[BaseConverter]] = {}

    @classmethod
    def register(cls, key: str, converter_class: Type[BaseConverter]):
        if not issubclass(converter_class, BaseConverter):
            raise TypeError(f"{converter_class.__name__} không kế thừa BaseConverter")
        cls._registry[key] = converter_class

    @classmethod
    def create(cls, type: ConverterType) -> BaseConverter:
        converter_class = cls._registry.get(type)
        if not converter_class:
            raise ValueError(f"Không tìm thấy converter cho loại '{type}'")

        return converter_class()

    @classmethod
    def get_available_converters(cls):
        return list(cls._registry.keys())
