from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.services.text_utils import ensure_markdown

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response từ LLM"""
    answer: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMService:
    """Service để tương tác với Gemini LLM models via OpenAI compatibility"""

    DEFAULT_SYSTEM_PROMPT = """
Ban la mot tro ly AI thong minh va huu ich. Nhiem vu cua ban la tra loi cau hoi dua tren ngu canh duoc cung cap.

Huong dan:
1. Chi su dung thong tin tu ngu canh duoc cung cap de tra loi
2. ***Neu cau tra loi khong co trong ngu canh, hay noi ro rang rang ban khong tim thay thong tin***
3. Trich dan nguon khi co the (vi du: "Theo tai lieu X...")
4. Tra loi bang tieng Viet mot cach ro rang va mach lac
5. Neu co nhieu nguon cung cap thong tin khac nhau, hay tong hop chung
6. ***Dung bo sung thong tin khong co trong ngu canh***
7. Dinh dang cau tra loi bang Markdown (tieu de, danh sach)
"""

    RAG_PROMPT_TEMPLATE = """Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên ngữ cảnh trên. Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ điều đó."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 120,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Khởi tạo LLM Service

        Args:
        model: Tên model Gemini (mặc định: gemini-2.0-flash)
        base_url: URL của Gemini OpenAI compatibility API
        api_key: Gemini API key
        temperature: Temperature cho generation (0.0 - 1.0)
        timeout: Timeout cho requests (giây)
            max_tokens: Default max tokens for responses
        """
        self.model = model
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided")

        # Khởi tạo LLM
        model_kwargs = {}
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        self._llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout=self.timeout,
            **model_kwargs,
        )

        # Khởi tạo prompt template
        self._rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.DEFAULT_SYSTEM_PROMPT),
            ("human", self.RAG_PROMPT_TEMPLATE),
        ])

        # Khởi tạo chain
        self._rag_chain = self._rag_prompt | self._llm | StrOutputParser()

        logger.info(
            f"LLMService initialized with model={self.model}, "
            f"base_url={self.base_url}, temperature={self.temperature}"
        )

    def answer_with_context(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Trả lời câu hỏi dựa trên context được cung cấp
        
        Args:
            question: Câu hỏi của người dùng
            context: Ngữ cảnh từ RAG
            temperature: Override temperature mặc định
            max_tokens: Giới hạn số tokens trong response
            
        Returns:
            LLMResponse với câu trả lời
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        if not context or not context.strip():
            logger.warning("Empty context provided, answering without context")
            context = "Không có thông tin liên quan được tìm thấy trong tài liệu."

        try:
            # Override temperature nếu được cung cấp
            if temperature is not None:
                original_temp = self._llm.temperature
                self._llm.temperature = temperature

            logger.info(f"Generating answer for question: '{question[:100]}...'")
            logger.debug(f"Context length: {len(context)} characters")

            # Invoke chain
            answer = self._rag_chain.invoke({
                "context": context,
                "question": question,
            })

            logger.info("Answer generated successfully")

            # Restore original temperature
            if temperature is not None:
                self._llm.temperature = original_temp

            return LLMResponse(
                answer=answer.strip(),
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}", exc_info=True)
            raise

    def answer_with_sources(
        self,
        question: str,
        sources: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Trả lời câu hỏi dựa trên danh sách sources
        
        Args:
            question: Câu hỏi của người dùng
            sources: List các chunks với metadata
            temperature: Override temperature
            max_tokens: Giới hạn tokens
            
        Returns:
            LLMResponse với câu trả lời
        """
        if not sources:
            context = "Không tìm thấy thông tin liên quan trong tài liệu."
        else:
            # Format context từ sources
            context_parts = []
            for idx, source in enumerate(sources, 1):
                file_name = source.get("file_name", "Unknown")
                content = source.get("content", "")
                score = source.get("score", 0.0)
                
                context_parts.append(
                    f"[Nguồn {idx} - {file_name} (độ liên quan: {score:.2f})]\n{content}"
                )
            
            context = "\n\n---\n\n".join(context_parts)

        return self.answer_with_context(
            question=question,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text từ prompt đơn giản (không dùng RAG)
        
        Args:
            prompt: Prompt text
            temperature: Override temperature
            max_tokens: Giới hạn tokens
            
        Returns:
            Generated text
        """
        try:
            if temperature is not None:
                original_temp = self._llm.temperature
                self._llm.temperature = temperature

            response = self._llm.invoke(prompt)
            
            if temperature is not None:
                self._llm.temperature = original_temp

            return response.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate text: {str(e)}", exc_info=True)
            raise

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Cập nhật system prompt
        
        Args:
            system_prompt: System prompt mới
        """
        self._rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", self.RAG_PROMPT_TEMPLATE),
        ])
        self._rag_chain = self._rag_prompt | self._llm | StrOutputParser()
        logger.info("System prompt updated")

    def get_available_models(self) -> List[str]:
        """
        Lấy danh sách models có sẵn từ Gemini (static list)

        Returns:
            List tên models
        """
        # Static list of available Gemini models via OpenAI compatibility
        return [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    def build_answer_payload(self, answer_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compose a JSON-friendly payload that pairs the raw answer with detailed source references.

        Args:
            answer_text: The natural-language answer returned by the LLM.
            sources: List of chunk metadata dictionaries returned from the retriever.

        Returns:
            Dictionary containing the answer content plus a `references` array.
        """
        references: List[Dict[str, Any]] = []

        for idx, source in enumerate(sources, start=1):
            source_id = source.get("source_id") or f"S{idx}"
            file_name = source.get("file_name", "unknown")
            document_id = source.get("document_id", "unknown")
            chunk_index = source.get("chunk_index", 0)
            score = float(source.get("score", 0.0))
            snippet = source.get("content", "")
            start_char = source.get("start_char", 0)
            end_char = source.get("end_char", 0)
            source_path = source.get("source_path")

            explanation = (
                f"Trich tu {file_name}, chunk #{chunk_index} "
                f"(ky tu {start_char}-{end_char})"
            )

            references.append(
                {
                    "source_id": source_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "chunk_index": chunk_index,
                    "score": score,
                    "snippet": snippet,
                    "start_char": start_char,
                    "end_char": end_char,
                    "source_path": source_path,
                    "explanation": explanation,
                    "content_format": "markdown",
                }
            )

        markdown_answer = ensure_markdown(answer_text, prefer_title=False)

        return {
            "content": markdown_answer,
            "references": references,
        }
