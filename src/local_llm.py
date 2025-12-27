"""Local LLM integration supporting multiple backends.

Supports:
- Ollama (default)
- Hugging Face Transformers
- llama.cpp via python bindings
- vLLM for faster inference
"""
import os
import subprocess
from typing import Optional, Dict, Any, Iterator
from enum import Enum


class LLMBackend(Enum):
    """Supported LLM backends."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"


class LocalLLM:
    """Unified interface for local LLM inference."""
    
    def __init__(
        self,
        backend: str = "ollama",
        model: str = "mistral",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize local LLM.
        
        Args:
            backend: Backend to use ("ollama", "huggingface", "llama_cpp", "vllm")
            model: Model name or path
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters
        """
        self.backend = LLMBackend(backend.lower())
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._model_instance = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == LLMBackend.OLLAMA:
            # Ollama doesn't need pre-initialization
            pass
        elif self.backend == LLMBackend.HUGGINGFACE:
            self._init_huggingface()
        elif self.backend == LLMBackend.LLAMA_CPP:
            self._init_llama_cpp()
        elif self.backend == LLMBackend.VLLM:
            self._init_vllm()
    
    def _init_huggingface(self):
        """Initialize Hugging Face Transformers backend."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            device = self.kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model_instance = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None
            )
            if device == "cpu":
                self._model_instance = self._model_instance.to(device)
        except ImportError:
            raise RuntimeError("transformers library not installed. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face model: {e}")
    
    def _init_llama_cpp(self):
        """Initialize llama.cpp backend."""
        try:
            from llama_cpp import Llama
            
            n_ctx = self.kwargs.get("n_ctx", 2048)
            n_threads = self.kwargs.get("n_threads", None)
            
            self._model_instance = Llama(
                model_path=self.model,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load llama.cpp model: {e}")
    
    def _init_vllm(self):
        """Initialize vLLM backend."""
        try:
            from vllm import LLM
            
            self._model_instance = LLM(
                model=self.model,
                tensor_parallel_size=self.kwargs.get("tensor_parallel_size", 1),
                max_model_len=self.kwargs.get("max_model_len", 2048)
            )
        except ImportError:
            raise RuntimeError("vllm not installed. Install with: pip install vllm")
        except Exception as e:
            raise RuntimeError(f"Failed to load vLLM model: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream responses (returns iterator)
        
        Returns:
            Generated text or iterator for streaming
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if stream:
            return self._generate_stream(prompt, temp, max_tok)
        
        if self.backend == LLMBackend.OLLAMA:
            return self._generate_ollama(prompt, temp, max_tok)
        elif self.backend == LLMBackend.HUGGINGFACE:
            return self._generate_huggingface(prompt, temp, max_tok)
        elif self.backend == LLMBackend.LLAMA_CPP:
            return self._generate_llama_cpp(prompt, temp, max_tok)
        elif self.backend == LLMBackend.VLLM:
            return self._generate_vllm(prompt, temp, max_tok)
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using Ollama."""
        try:
            # Use UTF-8 encoding explicitly to avoid Windows cp1252 encoding issues
            # Set environment variables to force UTF-8 in subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'  # Use UTF-8 on Windows
            
            # Use Popen with explicit encoding to avoid threading encoding issues
            process = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace',
                env=env,
                text=True
            )
            
            # Send prompt and get response
            stdout, stderr = process.communicate(input=prompt, timeout=600)
            
            if process.returncode != 0:
                error_msg = stderr if stderr else "Unknown error"
                raise RuntimeError(f"Ollama error: {error_msg}")
            
            return stdout.strip() if stdout else ""
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Please install Ollama from https://ollama.ai")
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("Ollama request timed out (10 minutes)")
        except UnicodeDecodeError as e:
            raise RuntimeError(f"Encoding error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def _generate_huggingface(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using Hugging Face Transformers."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model_instance.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model_instance.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text
    
    def _generate_llama_cpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using llama.cpp."""
        response = self._model_instance(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\n", "Human:", "Assistant:"]
        )
        return response["choices"][0]["text"].strip()
    
    def _generate_vllm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using vLLM."""
        outputs = self._model_instance.generate(
            [prompt],
            sampling_params={
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        return outputs[0].outputs[0].text.strip()
    
    def _generate_stream(self, prompt: str, temperature: float, max_tokens: int) -> Iterator[str]:
        """Generate streaming response (placeholder - implement per backend)."""
        # For now, return non-streaming and yield the result
        result = self.generate(prompt, temperature, max_tokens, stream=False)
        yield result


# Global instance for backward compatibility
_default_llm: Optional[LocalLLM] = None


def ask_llm(
    prompt: str,
    model: str = "mistral",
    backend: str = "ollama",
    **kwargs
) -> str:
    """
    Calls a local open-source LLM (backward compatible function).
    
    Args:
        prompt: Input prompt
        model: Model name
        backend: Backend to use ("ollama", "huggingface", "llama_cpp", "vllm")
        **kwargs: Additional parameters for LocalLLM
    
    Returns:
        Generated text
    """
    global _default_llm
    
    # Use environment variables for backend selection
    backend_env = os.environ.get("LLM_BACKEND", backend)
    model_env = os.environ.get("LLM_MODEL", model)
    
    # Create or reuse LLM instance
    if _default_llm is None or _default_llm.backend.value != backend_env or _default_llm.model != model_env:
        _default_llm = LocalLLM(backend=backend_env, model=model_env, **kwargs)
    
    return _default_llm.generate(prompt)
