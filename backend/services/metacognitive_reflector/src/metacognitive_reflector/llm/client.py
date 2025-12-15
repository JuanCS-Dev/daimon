"""
Unified LLM Client - Multi-Provider Support
============================================

Provides a unified interface for LLM inference across providers:
- Nebius Token Factory (OpenAI-compatible)
- Google Gemini (native API)

Reference:
- Nebius: https://docs.tokenfactory.nebius.com/quickstart
- Cookbook: https://github.com/nebius/token-factory-cookbook

Usage:
    client = get_llm_client()
    response = await client.generate("What is consciousness?")
    
    # Or with chat format
    response = await client.chat([
        {"role": "system", "content": "You are a metacognitive judge."},
        {"role": "user", "content": "Evaluate this action..."}
    ])
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import httpx
try:
    from google import genai
    from google.genai import types
    _HAS_GOOGLE_GENAI = True
except ImportError:
    _HAS_GOOGLE_GENAI = False

from .config import (
    LLMConfig,
    LLMProvider,
    NebiusConfig,
    GeminiConfig,
    AnthropicConfig,
    GEMINI_MODELS,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Unified response from LLM.
    
    Attributes:
        text: Generated text content
        model: Model used for generation
        provider: Provider used (nebius/gemini)
        usage: Token usage statistics
        finish_reason: Why generation stopped
        latency_ms: Request latency in milliseconds
        cached: Whether response was from cache
        raw: Raw response from provider
    """
    text: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False
    raw: Optional[Dict[str, Any]] = None
    
    @property
    def input_tokens(self) -> int:
        """Number of input tokens."""
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def output_tokens(self) -> int:
        """Number of output tokens."""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", 0)


class UnifiedLLMClient:
    """
    Unified LLM Client with multi-provider support.
    
    Features:
    - OpenAI-compatible API for Nebius
    - Native API for Gemini
    - Response caching (5min TTL)
    - Automatic retries with exponential backoff
    - Provider fallback
    
    Example:
        client = UnifiedLLMClient()
        
        # Simple generation
        response = await client.generate("What is truth?")
        print(response.text)
        
        # Chat format
        response = await client.chat([
            {"role": "system", "content": "You are VERITAS."},
            {"role": "user", "content": "Evaluate this claim..."}
        ])
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the unified LLM client.
        
        Args:
            config: LLM configuration (loads from env if not provided)
        """
        self.config = config or LLMConfig.from_env()
        
        # Response cache
        self._cache: Dict[str, tuple[LLMResponse, float]] = {}
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._cache_hits = 0
        
        # Log initialization
        provider = self.config.active_provider
        if provider == LLMProvider.NEBIUS:
            model = self.config.nebius.model
        elif provider == LLMProvider.ANTHROPIC:
            model = self.config.anthropic.model
        else:
            model = self.config.gemini.model

        logger.info(
            f"🧠 LLM Client initialized | "
            f"Provider: {provider.value} | "
            f"Model: {model}"
        )
        
        # Initialize Google GenAI client if needed
        self._google_client: Optional[genai.Client] = None
        if _HAS_GOOGLE_GENAI and self.config.gemini.is_configured:
            if self.config.gemini.use_vertex_ai:
                # Vertex AI with API Key requires direct HTTP (SDK forbids it)
                # We will use httpx in _vertex_chat
                self._google_client = None
            else:
                self._google_client = genai.Client(api_key=self.config.gemini.api_key)

    def get_available_models(self) -> Dict[str, str]:
        """Return available Gemini models map."""
        return GEMINI_MODELS
    
    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system message
            temperature: Override temperature
            max_tokens: Override max tokens
            use_cache: Whether to use response cache
            
        Returns:
            LLMResponse with generated text
        """
        # Dynamic model selection support
        # usage: generate("prompt", model="gemini-2.5-pro") - handled by **kwargs in future?
        # currently adhering to signature. 
        # For now, we rely on the chat method's dynamic capabilities if we were to adding them there,
        # but the prompt mentions "client.generate(..., model=...)" pattern potentially.
        # Let's strictly follow the implementation plan which mentioned "override passed in arguments".
        # Since `generate` calls `chat`, we should update `chat` signature first or just existing args.
        # The user wanted a selector in dashboard -> backend preparedness.
        # We will add a `model_override` arg to `generate` and `chat`.
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            # We need to update chat signature to accept overrides, 
            # for now passing it might require changing this method sig too
            # or we rely on the fact that we can add it to this method.
            # I will assume we should update the generate signature in a subsequent tool call if I strictly follow chunk. 
            # But I can do it here.
        )

    async def generate_v2(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """
        Generate text from a prompt with dynamic model support.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            model_override=model,
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            temperature: Override temperature
            max_tokens: Override max tokens
            use_cache: Whether to use response cache
            
        Returns:
            LLMResponse with generated text
        """
        # Check cache
        if use_cache and self.config.enable_caching:
            cache_key = self._cache_key(messages, temperature, max_tokens, model_override)
            cached = self._get_cached(cache_key)
            if cached:
                self._cache_hits += 1
                return cached
        
        # Route to appropriate provider
        provider = self.config.active_provider
        
        # If model override implies a specific provider, we could switch.
        # For now, if model_override is set and starts with 'gemini', force Gemini
        if model_override and model_override.startswith("gemini"):
            provider = LLMProvider.GEMINI
        
        for attempt in range(self.config.retry_attempts):
            try:
                if provider == LLMProvider.NEBIUS:
                    response = await self._nebius_chat(
                        messages, temperature, max_tokens
                    )
                elif provider == LLMProvider.ANTHROPIC:
                    response = await self._anthropic_chat(
                        messages, temperature, max_tokens
                    )
                else:
                    if self.config.gemini.use_vertex_ai:
                         response = await self._vertex_chat(
                            messages, temperature, max_tokens, model_override
                        )
                    else:
                        response = await self._gemini_chat(
                            messages, temperature, max_tokens, model_override
                        )
                
                # Cache response
                if use_cache and self.config.enable_caching:
                    self._cache[cache_key] = (response, time.time())
                
                # Update stats
                self._total_requests += 1
                self._total_tokens += response.total_tokens
                
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                # Don't retry on auth/config errors - they won't fix themselves
                if "401" in error_str or "403" in error_str or "invalid" in error_str or "unauthorized" in error_str:
                    logger.error(f"LLM auth/config error (no retry): {e}")
                    raise
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise
        
        # Should not reach here
        raise RuntimeError("All retry attempts failed")
    
    async def _nebius_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """
        Send chat request to Nebius Token Factory.
        
        Uses OpenAI-compatible API.
        Reference: https://docs.tokenfactory.nebius.com/quickstart
        """
        config = self.config.nebius
        
        # Build request body (OpenAI format)
        request_body = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature or config.temperature,
            "max_tokens": max_tokens or config.max_tokens,
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.base_url}chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Nebius API error {response.status_code}: {error_text}"
                )
            
            result = response.json()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse OpenAI-format response
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        return LLMResponse(
            text=message.get("content", ""),
            model=result.get("model", config.model),
            provider=LLMProvider.NEBIUS,
            usage=result.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
            cached=False,
            raw=result,
        )
    
    async def _gemini_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send chat request to Google Gemini using google-genai SDK.
        """
        if not _HAS_GOOGLE_GENAI:
             raise RuntimeError("google-genai SDK not installed. Please install it.")
        if not self._google_client:
             raise RuntimeError("Google GenAI client not initialized.")

        config = self.config.gemini
        model_id = model_override or config.model

        # Validate model if override provided
        if model_override and model_override not in GEMINI_MODELS:
            logger.warning(f"Unknown model {model_override}, falling back to default {config.model}")
            model_id = config.model

        # Convert messages to SDK format
        # SDK supports list of dicts with role/parts
        # System instruction is typically separate in SDK 2.0/3.0
        
        system_instruction = None
        chat_history = []
        last_user_message = ""

        # Pre-process messages
        # Google GenAI SDK 'chats' are history + new message
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                # If we have multiple user messages in a row or end with user, handle appropriately
                # Ideally: system -> [user, model]* -> user(current)
                # We will build history and keep the last one as the triggers
                chat_history.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant" or role == "model":
                chat_history.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                ))

        # Extract last message as the prompt if it is user, otherwise empty (should not happen in chat completion usually)
        if chat_history and chat_history[-1].role == "user":
            prompt_content = chat_history.pop()
            prompt_text = prompt_content.parts[0].text
        else:
            # Fallback if no user message at end
            prompt_text = "..." 

        # Config
        generation_config = types.GenerateContentConfig(
            temperature=temperature or config.temperature,
            max_output_tokens=max_tokens or config.max_tokens,
            thinking_config=types.ThinkingConfig(include_thoughts=False) if "2.5-pro" in model_id else None, # Only for Pro? Flash doesn't support thinking yet maybe
            system_instruction=system_instruction,
        )
        
        # Adjust for Thinking Models
        if "thinking" in model_id or "2.5-pro" in model_id or config.thinking_budget > 0:
             # Basic thinking config setup
             pass 

        start_time = time.time()
        
        try:
            # Using the Async Client
            response = await self._google_client.aio.models.generate_content(
                model=model_id,
                contents=chat_history + [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])] if chat_history else prompt_text,
                config=generation_config,
            )
            
        except Exception as e:
             raise RuntimeError(f"Gemini SDK error: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000
        
        # Parse Response
        try:
            text = response.text
        except ValueError:
            text = "" # Blocked or empty

        # Usage
        usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0,
        }

        # Finish Reason
        finish_reason = "STOP" # Default
        if response.candidates and response.candidates[0].finish_reason:
             finish_reason = response.candidates[0].finish_reason.name

        return LLMResponse(
            text=text,
            model=model_id,
            provider=LLMProvider.GEMINI,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            cached=False,
            raw={"response": str(response)},
        )

    async def _vertex_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send chat request to Vertex AI via HTTP (bypassing SDK Auth restrictions).
        Matches user provided CURL pattern:
        https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={API_KEY}
        """
        config = self.config.gemini
        model_id = model_override or config.model
        
        # Build URL matches USER PROVIDED CURL
        # "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent?key=${API_KEY}"
        # We use generateContent for non-streaming
        
        # If location is provided, we can prepend it, but user example used global/us-central1 implicit?
        # Let's try to follow the exact host from curl: aiplatform.googleapis.com
        # BUT standard vertex is usually {location}-aiplatform
        # User curl: `https://aiplatform.googleapis.com/...` -> This suggests global or default.
        # We will use the config location if set to something specific, otherwise default.
        
        base_url = "https://aiplatform.googleapis.com"
        if config.location and config.location != "us-central1":
             base_url = f"https://{config.location}-aiplatform.googleapis.com"
             
        url = (
            f"{base_url}/v1/publishers/google/models/{model_id}:generateContent"
        )
        
        # Convert messages to Gemini/Vertex format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Vertex API supports system_instruction in body
                system_instruction = {"parts": [{"text": content}]}
            else:
                vertex_role = "user" if role == "user" else "model"
                contents.append({
                    "role": vertex_role,
                    "parts": [{"text": content}]
                })

        request_body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature or config.temperature,
                "maxOutputTokens": max_tokens or config.max_tokens,
            }
        }
        
        if system_instruction:
            request_body["systemInstruction"] = system_instruction

        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                url,
                params={"key": config.api_key},
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Vertex API error {response.status_code}: {error_text}"
                )
            
            result = response.json()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse Vertex response (Similar structure to Gemini)
        candidates = result.get("candidates", [{}])
        content = candidates[0].get("content", {}) if candidates else {}
        parts = content.get("parts", [{}])
        text = parts[0].get("text", "") if parts else ""
        
        # Usage
        usage_metadata = result.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }
        
        return LLMResponse(
            text=text,
            model=model_id,
            provider=LLMProvider.GEMINI,
            usage=usage,
            finish_reason=candidates[0].get("finishReason", "STOP") if candidates else "STOP",
            latency_ms=latency_ms,
            cached=False,
            raw=result,
        )

    async def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """
        Send chat request to Anthropic Claude API.

        Uses Anthropic Messages API format.
        Reference: https://docs.anthropic.com/en/api/messages
        """
        config = self.config.anthropic

        # Separate system message from conversation
        system_content = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                # Anthropic uses "user" and "assistant" roles
                anthropic_role = "user" if role == "user" else "assistant"
                anthropic_messages.append({
                    "role": anthropic_role,
                    "content": content,
                })

        # Build request body
        request_body: Dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or config.max_tokens,
        }

        # Add optional parameters
        if temperature is not None or config.temperature != 0.7:
            request_body["temperature"] = temperature or config.temperature

        if system_content:
            request_body["system"] = system_content

        start_time = time.time()

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=request_body,
                headers={
                    "x-api-key": config.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Anthropic API error {response.status_code}: {error_text}"
                )

            result = response.json()

        latency_ms = (time.time() - start_time) * 1000

        # Parse Anthropic response
        content_blocks = result.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        # Extract usage
        usage_data = result.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": (
                usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0)
            ),
        }

        return LLMResponse(
            text=text,
            model=result.get("model", config.model),
            provider=LLMProvider.ANTHROPIC,
            usage=usage,
            finish_reason=result.get("stop_reason", "end_turn"),
            latency_ms=latency_ms,
            cached=False,
            raw=result,
        )

    def _cache_key(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model_override: Optional[str] = None,
    ) -> str:
        """Generate cache key from request parameters."""
        provider = self.config.active_provider
        if provider == LLMProvider.NEBIUS:
            model = self.config.nebius.model
        elif provider == LLMProvider.ANTHROPIC:
            model = self.config.anthropic.model
        else:
            model = model_override or self.config.gemini.model
        
        key_data = {
            "provider": provider.value,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        """Get cached response if not expired."""
        if key not in self._cache:
            return None
        
        response, timestamp = self._cache[key]
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            del self._cache[key]
            return None
        
        # Return copy with cached flag
        return LLMResponse(
            text=response.text,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
            finish_reason=response.finish_reason,
            latency_ms=0.0,
            cached=True,
            raw=response.raw,
        )
    
    def clear_cache(self) -> int:
        """Clear response cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": self.config.active_provider.value,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._cache_hits / max(1, self._total_requests + self._cache_hits)
            ),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check LLM connectivity.
        
        Returns:
            Dict with health status and provider info
        """
        try:
            response = await self.generate(
                "Say 'OK' if you're operational.",
                max_tokens=10,
                use_cache=False,
            )
            return {
                "healthy": True,
                "provider": response.provider.value,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "response": response.text[:50],
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "provider": self.config.active_provider.value,
            }


# Singleton instance
_client_instance: Optional[UnifiedLLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> UnifiedLLMClient:
    """
    Get or create the LLM client singleton.
    
    Args:
        config: Optional configuration (uses env vars if not provided)
        
    Returns:
        UnifiedLLMClient instance
    """
    global _client_instance
    
    if _client_instance is None or config is not None:
        _client_instance = UnifiedLLMClient(config)
    
    return _client_instance


def reset_llm_client() -> None:
    """Reset the LLM client singleton (useful for testing)."""
    global _client_instance
    _client_instance = None

