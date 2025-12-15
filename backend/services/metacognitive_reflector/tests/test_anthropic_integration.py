"""
Anthropic Claude Integration Tests
==================================

Tests for the unified LLM client with Anthropic Claude support.

Run:
    pytest tests/test_anthropic_integration.py -v

Or standalone:
    python tests/test_anthropic_integration.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest


class TestAnthropicConfig:
    """Test Anthropic configuration."""

    def test_config_from_env(self):
        """Test configuration loads from environment."""
        from metacognitive_reflector.llm import LLMConfig

        config = LLMConfig.from_env()

        # Should have valid structure
        assert config.anthropic is not None
        assert config.retry_attempts >= 1

    def test_anthropic_config_defaults(self):
        """Test Anthropic default configuration."""
        from metacognitive_reflector.llm import AnthropicConfig

        config = AnthropicConfig()

        assert config.model == "claude-3-5-haiku-20241022"
        assert config.max_tokens == 4096
        assert config.timeout == 60
        assert config.temperature == 0.7

    def test_anthropic_config_is_configured(self):
        """Test is_configured property."""
        from metacognitive_reflector.llm import AnthropicConfig

        # Without API key
        config_without_key = AnthropicConfig()
        # Will be False if ANTHROPIC_API_KEY not in env
        if not os.getenv("ANTHROPIC_API_KEY"):
            assert not config_without_key.is_configured

    def test_provider_auto_selection_with_anthropic(self):
        """Test automatic provider selection includes Anthropic."""
        from metacognitive_reflector.llm import LLMConfig, LLMProvider

        config = LLMConfig.from_env()

        # If ANTHROPIC_API_KEY is set and Nebius not, should select Anthropic
        if not os.getenv("NEBIUS_API_KEY") and os.getenv("ANTHROPIC_API_KEY"):
            assert config.active_provider == LLMProvider.ANTHROPIC

    def test_explicit_anthropic_selection(self):
        """Test explicit Anthropic provider selection."""
        from metacognitive_reflector.llm import LLMConfig, LLMProvider

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        assert config.active_provider == LLMProvider.ANTHROPIC


class TestUnifiedLLMClientAnthropic:
    """Test the unified LLM client with Anthropic."""

    @pytest.mark.asyncio
    async def test_anthropic_generation(self):
        """Test text generation with Anthropic."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        response = await client.generate(
            "Say 'Hello from Claude' and nothing else.",
            max_tokens=50,
            use_cache=False,
        )

        assert response.text
        assert response.provider == LLMProvider.ANTHROPIC
        assert response.latency_ms > 0
        assert response.total_tokens > 0
        print(f"✓ Anthropic response: {response.text[:100]}")
        print(f"✓ Latency: {response.latency_ms:.0f}ms")
        print(f"✓ Tokens: {response.total_tokens}")

    @pytest.mark.asyncio
    async def test_anthropic_chat(self):
        """Test chat format with Anthropic."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        response = await client.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your purpose? Answer in one sentence."},
        ], max_tokens=100)

        assert response.text
        assert len(response.text) > 10
        print(f"✓ Chat response: {response.text[:200]}")

    @pytest.mark.asyncio
    async def test_anthropic_response_caching(self):
        """Test response caching works with Anthropic."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        prompt = "What is 2 + 2? Answer with just the number."

        # First call - not cached
        response1 = await client.generate(prompt, max_tokens=10)
        assert not response1.cached

        # Second call - should be cached
        response2 = await client.generate(prompt, max_tokens=10)
        assert response2.cached
        assert response2.latency_ms == 0.0
        assert response2.text == response1.text

        print(f"✓ Cache working: hit_rate={client.stats['cache_hit_rate']:.1%}")

    @pytest.mark.asyncio
    async def test_anthropic_health_check(self):
        """Test health check endpoint with Anthropic."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        health = await client.health_check()

        assert "healthy" in health
        print(f"✓ Health check: {health}")


class TestAnthropicDaimonIntegration:
    """Test integration with DAIMON learner tasks."""

    @pytest.mark.asyncio
    async def test_classification_task(self):
        """Test classification capability for preference learning."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        # Simulate a classification task from PreferenceLearner
        response = await client.chat([
            {
                "role": "system",
                "content": (
                    "You classify user responses to AI suggestions. "
                    "Output JSON with: {\"category\": \"approval\"|\"rejection\"|\"neutral\", "
                    "\"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Classify this user response:\n"
                    "\"No, that's not what I meant. Please try again.\""
                )
            }
        ], max_tokens=150, temperature=0.3)

        assert response.text
        assert "rejection" in response.text.lower() or "category" in response.text.lower()
        print(f"✓ Classification response:\n{response.text[:300]}")

    @pytest.mark.asyncio
    async def test_insight_extraction_task(self):
        """Test insight extraction for reflection engine."""
        from metacognitive_reflector.llm import (
            get_llm_client,
            LLMConfig,
            LLMProvider,
            reset_llm_client,
        )

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        reset_llm_client()
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        client = get_llm_client(config)

        # Simulate insight extraction task
        response = await client.chat([
            {
                "role": "system",
                "content": (
                    "You analyze behavioral data and extract actionable insights. "
                    "Be concise and specific."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze this data and extract insights:\n"
                    "- Rejections: 5 (category: code_style)\n"
                    "- Approvals: 2 (category: documentation)\n"
                    "- Time period: last 24 hours\n\n"
                    "What patterns do you see?"
                )
            }
        ], max_tokens=200, temperature=0.5)

        assert response.text
        assert len(response.text) > 50
        print(f"✓ Insight extraction:\n{response.text[:400]}")


async def run_quick_test():
    """Run a quick integration test."""
    from metacognitive_reflector.llm import (
        get_llm_client,
        LLMConfig,
        LLMProvider,
        reset_llm_client,
    )

    print("\n" + "=" * 60)
    print("🤖 ANTHROPIC CLAUDE - QUICK TEST")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set in environment")
        print("   Set it in .env or export ANTHROPIC_API_KEY=...")
        return False

    reset_llm_client()
    config = LLMConfig(provider=LLMProvider.ANTHROPIC)
    client = get_llm_client(config)

    print(f"\n📡 Provider: {client.config.active_provider.value}")
    print(f"🤖 Model: {client.config.anthropic.model}")

    print("\n⏳ Testing connection...")

    try:
        response = await client.generate(
            "Say 'DAIMON is learning!' and nothing else.",
            max_tokens=20,
            use_cache=False,
        )

        print(f"\n✅ SUCCESS!")
        print(f"   Response: {response.text}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print(f"   Tokens: {response.total_tokens}")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    # Load .env if exists
    env_path = Path(__file__).parent.parent.parent.parent.parent.parent / ".env"
    if env_path.exists():
        print(f"Loading .env from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

    # Run quick test
    success = asyncio.run(run_quick_test())
    sys.exit(0 if success else 1)
