"""Gemini Client - Google Gemini Integration for Maximus AI
========================================================

Cliente para Google Gemini (versão 3.0) com suporte a:
- Text generation com Thinking Config (Chain-of-Thought nativo)
- Tool calling (function calling)
- Embeddings
- Temporal Grounding (Contexto de Data/Hora)
- JSON Schema Output
- **Vertex AI Support (via google-genai SDK)**

Model: gemini-3.0-pro-001
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime

import httpx

try:
    from google import genai
    from google.genai import types
    _HAS_GOOGLE_GENAI = True
except ImportError:
    _HAS_GOOGLE_GENAI = False

from maximus_core_service.config import get_settings

logger = logging.getLogger(__name__)


class GenerateParams(TypedDict, total=False):
    """Parâmetros opcionais para geração de texto."""
    system_instruction: Optional[str]
    tools: Optional[List[Dict[str, Any]]]
    temperature: Optional[float]
    max_tokens: Optional[int]
    response_schema: Optional[Dict[str, Any]]


@dataclass
class GeminiConfig:
    """
    Configuração do Gemini.

    Attributes:
        api_key: Chave de API do Google Gemini.
        model: Identificador do modelo (ex: gemini-3.0-pro-001).
        temperature: Temperatura para amostragem (0.0 a 1.0).
        max_tokens: Máximo de tokens na saída.
        timeout: Timeout da requisição em segundos.
        thinking_level: Nível de raciocínio (HIGH/LOW).
        use_vertex: Habilitar Vertex AI backend.
        vertex_project_id: Google Cloud Project ID.
        vertex_location: Google Cloud Location (ex: us-central1).
    """
    api_key: str
    model: str = "gemini-3.0-pro-001"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: int = 60
    thinking_level: str = "HIGH"
    use_vertex: bool = False
    vertex_project_id: Optional[str] = None
    vertex_location: str = "us-central1"


class GeminiError(Exception):
    """Exceção base para erros do Gemini."""


class GeminiClient:
    """
    Cliente para Google Gemini (Vertex AI & AI Studio).
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: Optional[GeminiConfig] = None) -> None:
        """
        Inicializa o cliente Gemini.

        Args:
            config: Configuração opcional.
        """
        if config is None:
            settings = get_settings().llm
            vertex_pid = str(settings.vertex_project_id) if settings.vertex_project_id else None
            # pylint: disable=no-member
            self.config = GeminiConfig(
                api_key=str(settings.api_key),
                model=str(settings.model),
                temperature=float(settings.temperature),
                max_tokens=int(settings.max_tokens),
                timeout=int(settings.timeout),
                thinking_level=str(settings.thinking_level),
                use_vertex=bool(settings.use_vertex),
                vertex_project_id=vertex_pid,
                vertex_location=str(settings.vertex_location)
            )
        else:
            self.config = config

        self.api_key = self.config.api_key
        self.model = self.config.model

        # Initialize Vertex Client if enabled
        self.vertex_client: Optional[Any] = None
        if self.config.use_vertex:
            if not _HAS_GOOGLE_GENAI:
                logger.error("❌ Vertex AI enabled but 'google-genai' SDK not found.")
            else:
                try:
                    self.vertex_client = genai.Client(
                        vertexai=True,
                        project=self.config.vertex_project_id,
                        location=self.config.vertex_location
                    )
                    logger.info("🟢 Vertex AI Client Initialized (%s)",
                                self.config.vertex_location)
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    logger.error("❌ Failed to initialize Vertex AI Client: %s", e)

        self._log_boot_status()

    def _log_boot_status(self) -> None:
        """Exibe status de inicialização no log."""
        backend = "Vertex AI" if self.vertex_client else "AI Studio (Legacy)"
        logger.info(
            "🟢 DAIMON LINK ESTABLISHED | Model: %s | Backend: %s | Thinking: %s",
            self.model,
            backend,
            self.config.thinking_level
        )

    def _get_temporal_context(self) -> str:
        """Gera o contexto temporal atual."""
        current_time = datetime.now().strftime("%A, %d %B %Y, %H:%M")
        return (
            f"SYSTEM OVERRIDE: Current Operational Date is {current_time}. "
            f"You are running on Gemini 3.0 Pro High hardware."
        )

    def _build_generation_config(
        self,
        params: GenerateParams
    ) -> Dict[str, Any]:
        """Constrói a configuração de geração (Legacy HTTP)."""
        temp = params.get("temperature")
        max_tok = params.get("max_tokens")
        
        config: Dict[str, Any] = {
            "temperature": temp if temp is not None else self.config.temperature,
            "maxOutputTokens": max_tok if max_tok is not None else self.config.max_tokens,
        }

        if self.config.thinking_level:
            config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingLevel": self.config.thinking_level
            }

        if params.get("response_schema"):
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = params.get("response_schema")

        return config

    async def generate_text(
        self,
        prompt: str,
        **kwargs: Any  # Accepts optional args mapped to GenerateParams keys
    ) -> Dict[str, Any]:
        """
        Gera texto usando Gemini 3.0 via Vertex AI (Preferred) ou AI Studio (Fallback).
        Args:
            prompt: O texto do prompt.
            **kwargs: system_instruction, tools, temperature, max_tokens, response_schema.
        """
        params: GenerateParams = {
            "system_instruction": kwargs.get("system_instruction"),
            "tools": kwargs.get("tools"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "response_schema": kwargs.get("response_schema"),
        }

        temporal_context = self._get_temporal_context()
        sys_instr = params.get("system_instruction")
        final_system = (
            f"{temporal_context}\n\n{sys_instr}" if sys_instr else temporal_context
        )

        # Route to Vertex API if enabled and initialized
        if self.vertex_client:
            return await self._generate_vertex(prompt, final_system, params)

        # Fallback to Legacy HTTP (AI Studio)
        return await self._generate_legacy(prompt, final_system, params)

    async def _generate_vertex(
        self,
        prompt: str,
        system_instruction: str,
        params: GenerateParams
    ) -> Dict[str, Any]:
        """Execução via Vertex AI SDK."""
        try:
            # Prepare config values
            temp = params.get("temperature")
            max_tok = params.get("max_tokens")
            
            temperature = temp if temp is not None else self.config.temperature
            max_tokens = max_tok if max_tok is not None else self.config.max_tokens
            response_schema = params.get("response_schema")

            # Thinking Config
            thinking_config = None
            if self.config.thinking_level:
                thinking_config = types.ThinkingConfig(include_thoughts=True)

            # Tools
            gemini_tools = None
            if params.get("tools"):
                converted = self._convert_tools(params.get("tools") or [])
                gemini_tools = [types.Tool(function_declarations=converted)]

            # Execution
            response = self.vertex_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=gemini_tools,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=thinking_config,
                    response_mime_type="application/json" if response_schema else None,
                    response_schema=response_schema
                )
            )

            # Parse SDK object to Dict
            return self._parse_sdk_response(response)

        except Exception as e:
            logger.error("❌ Vertex AI Generation Failed: %s", e)
            raise GeminiError(f"Vertex Error: {e}") from e

    async def _generate_legacy(
        self,
        prompt: str,
        system_instruction: str,
        params: GenerateParams
    ) -> Dict[str, Any]:
        """Execução Legacy (httpx + AI Studio)."""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        generation_config = self._build_generation_config(params)

        request_body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
            "systemInstruction": {"parts": [{"text": system_instruction}]}
        }

        tools = params.get("tools")
        if tools:
            request_body["tools"] = [{
                "functionDeclarations": self._convert_tools(tools)
            }]

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=request_body,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                error_msg = f"Gemini Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise GeminiError(error_msg)

            return self._parse_gemini_response(response.json())

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converte tools para formato Gemini (compatível SDK e REST)."""
        gemini_tools = []
        for tool in tools:
            gemini_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
            }
            if "input_schema" in tool:
                gemini_tool["parameters"] = tool["input_schema"]
            elif "parameters" in tool:
                gemini_tool["parameters"] = tool["parameters"]
            gemini_tools.append(gemini_tool)
        return gemini_tools

    def _parse_sdk_response(self, response: Any) -> Dict[str, Any]:
        """Normaliza resposta do SDK."""
        text = response.text or ""
        tool_calls = []

        # Check candidates/parts for function calls
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "name": part.function_call.name,
                        "arguments": part.function_call.args
                    })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": "STOP",  # Simplified
            "raw": str(response)
        }

    def _parse_gemini_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza resposta REST JSON."""
        candidates = result.get("candidates", [])
        if not candidates:
            return {"text": "", "finish_reason": "error", "raw": result}

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text = ""
        tool_calls = []

        for part in parts:
            if "text" in part:
                text += part["text"]
            elif "functionCall" in part:
                func = part["functionCall"]
                tool_calls.append({
                    "name": func.get("name"),
                    "arguments": func.get("args", {})
                })

        return {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": candidate.get("finishReason", "STOP"),
            "raw": result,
        }

    async def generate_embeddings(self, text: str) -> List[float]:
        """Gera embeddings (Compatível Vertex e Legacy)."""
        if self.vertex_client:
            try:
                # Vertex Embeddings
                model = "text-embedding-004"
                resp = self.vertex_client.models.embed_content(
                    model=model,
                    contents=text
                )
                return resp.embeddings[0].values
            except Exception as e:
                logger.error("Vertex Embeddings failed: %s", e)
                raise GeminiError(f"Vertex Embeddings Error: {e}") from e

        # Legacy HTTP Embeddings
        url = f"{self.BASE_URL}/models/text-embedding-004:embedContent"
        request_body = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=request_body,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise GeminiError(f"Embeddings Error: {response.status_code}")

            return response.json().get("embedding", {}).get("values", [])
