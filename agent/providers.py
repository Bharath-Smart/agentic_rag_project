
from typing import Optional
from langchain_openai import ChatOpenAI
import openai
from .config import settings


def get_llm_model(model_choice: Optional[str] = None) -> ChatOpenAI:
    """
    Get LLM model configuration based on environment variables.
    
    Args:
        model_choice: Optional override for model choice
    
    Returns:
        Configured OpenAI-compatible model
    """
    llm_choice = model_choice or settings.llm_choice
    api_key = settings.openai_api_key

    model_kwargs = {"model": llm_choice}
    if api_key:
        model_kwargs["api_key"] = api_key
    return ChatOpenAI(**model_kwargs)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    
    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    return openai.AsyncOpenAI(
        api_key=settings.openai_api_key
    )


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return settings.embedding_model

