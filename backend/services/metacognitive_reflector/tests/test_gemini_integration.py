
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Adjust path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from metacognitive_reflector.llm.client import UnifiedLLMClient, LLMProvider, LLMConfig
from metacognitive_reflector.llm.config import GeminiConfig

@pytest.fixture
def mock_google_genai():
    with patch("metacognitive_reflector.llm.client.genai") as mock_genai:
        # Setup the client mock
        mock_client_instance = AsyncMock()
        mock_genai.Client.return_value = mock_client_instance
        
        # Setup aio.models.generate_content return value
        mock_response = MagicMock()
        mock_response.text = "Mocked Gemini Response"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"))]
        
        mock_client_instance.aio.models.generate_content.return_value = mock_response
        yield mock_client_instance

@pytest.fixture
def gemini_client(mock_google_genai):
    config = LLMConfig(
        provider=LLMProvider.GEMINI,
        gemini=GeminiConfig(
            api_key="fake_key_long_enough_to_pass_validation", 
            model="gemini-2.5-flash",
            use_vertex_ai=False
        )
    )
    # We patch _HAS_GOOGLE_GENAI to True for tests
    with patch("metacognitive_reflector.llm.client._HAS_GOOGLE_GENAI", True):
        client = UnifiedLLMClient(config)
        return client

@pytest.mark.asyncio
async def test_default_model_usage(gemini_client, mock_google_genai):
    """Test that default model from config is used."""
    await gemini_client.generate("Hello")
    
    # Check calling args
    args, kwargs = mock_google_genai.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"

@pytest.mark.asyncio
async def test_dynamic_model_override(gemini_client, mock_google_genai):
    """Test that model_override updates the API call."""
    await gemini_client.chat(
        [{"role": "user", "content": "Hi"}],
        model_override="gemini-2.5-pro"
    )
    
    args, kwargs = mock_google_genai.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-pro"

@pytest.mark.asyncio
async def test_unknown_model_fallback(gemini_client, mock_google_genai):
    """Test fallback to default if unknown model requested."""
    # We need to mock GEMINI_MODELS check indirectly or rely on client logic
    # client code: if model_override and model_override not in GEMINI_MODELS: fallback
    
    await gemini_client.chat(
        [{"role": "user", "content": "Hi"}],
        model_override="gemini-fake-9000"
    )
    
    args, kwargs = mock_google_genai.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"  # Should fallback

@pytest.mark.asyncio
async def test_error_handling(gemini_client, mock_google_genai):
    """Test graceful handling of SDK errors."""
    mock_google_genai.aio.models.generate_content.side_effect = Exception("API Error")
    
    with pytest.raises(RuntimeError) as excinfo:
        await gemini_client.generate("Trigger error", use_cache=False)
    
    assert "Gemini SDK error" in str(excinfo.value)

@pytest.mark.asyncio
async def test_token_accounting(gemini_client):
    """Test that token usage is correctly mapped."""
    response = await gemini_client.generate("Count tokens")
    
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["completion_tokens"] == 20
    assert response.usage["total_tokens"] == 30

@pytest.mark.asyncio
async def test_system_instruction(gemini_client, mock_google_genai):
    """Test parsing of system instruction."""
    await gemini_client.chat([
        {"role": "system", "content": "Be polite"},
        {"role": "user", "content": "Hello"}
    ])
    
    # Check messages passed to SDK
    args, kwargs = mock_google_genai.aio.models.generate_content.call_args
    
    # Verify system_instruction in config
    config_arg = kwargs.get("config")
    assert config_arg is not None
    assert config_arg.system_instruction == "Be polite" 

@pytest.mark.asyncio
async def test_vertex_initialization_logic():
    """Test that Vertex mode initializes client correctly."""
    config = LLMConfig(
        provider=LLMProvider.GEMINI,
        gemini=GeminiConfig(
            api_key="fake_key_longer_than_needed",
            model="gemini-2.5-flash",
            use_vertex_ai=True,
            project_id="my-project",
            location="us-central1"
        )
    )
    
    with patch("metacognitive_reflector.llm.client.genai") as mock_genai:
        with patch("metacognitive_reflector.llm.client._HAS_GOOGLE_GENAI", True):
            # Patch genai.Client to return a mock
            mock_client_instance = AsyncMock()
            mock_genai.Client.return_value = mock_client_instance
            
            # Setup mock system to verify init call, response not needed for init test
            client = UnifiedLLMClient(config)
            
            # Validate call args
            mock_genai.Client.assert_called_with(
                vertexai=True,
                project="my-project",
                location="us-central1",
                api_key="fake_key_longer_than_needed"
            ) 
