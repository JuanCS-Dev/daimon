
import asyncio
import os
import sys

# Add src to path
# We are in backend/services/metacognitive_reflector
# we need to add src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from metacognitive_reflector.llm.client import get_llm_client
from metacognitive_reflector.llm.config import GEMINI_MODELS

async def main():
    print("--- Verifying Gemini Integration ---")
    
    # 1. Check Config
    client = get_llm_client()
    print(f"Provider: {client.config.active_provider}")
    print(f"Default Model: {client.config.gemini.model}")
    
    if client.config.active_provider != "gemini":
        print("WARNING: Active provider is not GEMINI")

    # 2. Test Default Generation (Flash)
    print("\n--- Testing Default (Flash) ---")
    try:
        response = await client.generate("Hello, just say 'Flash is ready'.", max_tokens=20)
        print(f"Response: {response.text}")
        print(f"Model Used: {response.model}")
        print(f"Latency: {response.latency_ms:.2f}ms")
    except Exception as e:
        print(f"ERROR: {e}")

    # 3. Test Dynamic Override (Pro)
    print("\n--- Testing Override (Pro) ---")
    try:
        # Override with Pro model
        # Note: We added model_override to chat/generate (via kwargs or explicit args depending on how tool executed)
        # client.py update added model to generate_v2 and chat. Let's try to use chat directly or generate if args passed
        # My refactor included model in generate_v2 but I missed updating generate in one chunk maybe? 
        # I updated chat to accept model_override.
        # Let's test client.chat directly with override.
        response = await client.chat(
            [{"role": "user", "content": "Explain quantum entanglement briefly."}],
            model_override="gemini-2.5-pro" 
        )
        print(f"Response: {response.text[:100]}...")
        print(f"Model Used: {response.model}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv("/media/juan/DATA/projetos/Noesis/Daimon/.env")
    asyncio.run(main())
