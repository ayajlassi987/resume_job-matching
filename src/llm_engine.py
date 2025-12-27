"""Unified LLM engine supporting both OpenAI and local LLMs.

This module provides a unified interface for LLM inference,
automatically falling back to local LLMs when OpenAI is unavailable.
"""
import os
from typing import Optional
from openai import OpenAI

try:
    client = OpenAI()
    OPENAI_AVAILABLE = True
except Exception:
    client = None
    OPENAI_AVAILABLE = False

from src.local_llm import ask_llm, LocalLLM


def ask(
    prompt: str,
    model: str = "gpt-4o-mini",
    use_openai: Optional[bool] = None,
    fallback_to_local: bool = True,
    local_model: str = "mistral",
    local_backend: str = "ollama",
    **kwargs
) -> str:
    """
    Unified LLM interface with automatic fallback.
    
    Args:
        prompt: Input prompt
        model: OpenAI model name (if using OpenAI)
        use_openai: Force OpenAI usage (None = auto-detect)
        fallback_to_local: Fallback to local LLM if OpenAI fails
        local_model: Local model name for fallback
        local_backend: Local backend type
        **kwargs: Additional parameters for local LLM
    
    Returns:
        Generated text
    """
    # Determine if we should use OpenAI
    if use_openai is None:
        use_openai = OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY"))
    
    if use_openai and OPENAI_AVAILABLE:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        except Exception as e:
            if fallback_to_local:
                print(f"OpenAI request failed, falling back to local LLM: {e}")
                return ask_llm(prompt, model=local_model, backend=local_backend, **kwargs)
            else:
                raise
    else:
        # Use local LLM
        return ask_llm(prompt, model=local_model, backend=local_backend, **kwargs)
