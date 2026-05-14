"""
NeoStats AI Chatbot — Configuration Module
All API keys and global settings are managed here.
Load secrets from environment variables for security.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Language model provider configurations."""

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_default_model: str = "gpt-4o-mini"
    openai_models: list = field(default_factory=lambda: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

    # Groq
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_default_model: str = "llama-3.3-70b-versatile"
    groq_models: list = field(default_factory=lambda: [
        "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
        "mixtral-8x7b-32768", "gemma2-9b-it"
    ])

    # Google Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_default_model: str = "gemini-1.5-flash"
    gemini_models: list = field(default_factory=lambda: [
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"
    ])


@dataclass
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) settings."""
    embedding_model: str = "all-MiniLM-L6-v2"          # SentenceTransformer model
    chunk_size: int = 512                                # Characters per chunk
    chunk_overlap: int = 64                              # Overlap between chunks
    top_k_results: int = 4                               # Number of chunks to retrieve
    similarity_threshold: float = 0.25                  # Min cosine similarity
    vector_store_path: str = "data/vector_store"        # FAISS index path
    knowledge_base_path: str = "data/knowledge_base"   # Uploaded docs path


@dataclass
class WebSearchConfig:
    """Web search integration settings."""
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    max_results: int = 5
    search_timeout: int = 10  # seconds


@dataclass
class AppConfig:
    """Application-level settings."""
    app_title: str = "MediBot Pro — Intelligent Healthcare Assistant"
    app_subtitle: str = "Powered by RAG + Live Web Search"
    app_icon: str = "🩺"
    version: str = "1.0.0"

    # Response modes
    concise_max_tokens: int = 300
    detailed_max_tokens: int = 1500
    default_temperature: float = 0.4

    # Chat history
    max_history_turns: int = 20

    # Supported file types for RAG
    supported_file_types: list = field(default_factory=lambda: [
        "pdf", "txt", "md", "docx", "csv", "xlsx", "xls"
    ])


# Singleton instances
llm_config = LLMConfig()
rag_config = RAGConfig()
web_search_config = WebSearchConfig()
app_config = AppConfig()
