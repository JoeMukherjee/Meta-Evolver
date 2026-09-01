"""Provider-agnostic model access and embeddings."""
from meta_evolver.llm.client import (
    BaseLLMClient,
    LiteLLMClient,
    LLMError,
    LLMResponse,
    ScriptedLLMClient,
    ToolCall,
    api_key_for,
    build_client,
    effective_model,
    sampling_params_deprecated,
)
from meta_evolver.llm.embeddings import Embedder, cosine, hashed_embedding

__all__ = [
    "BaseLLMClient",
    "Embedder",
    "LLMError",
    "LLMResponse",
    "LiteLLMClient",
    "ScriptedLLMClient",
    "ToolCall",
    "api_key_for",
    "build_client",
    "cosine",
    "effective_model",
    "hashed_embedding",
    "sampling_params_deprecated",
]
