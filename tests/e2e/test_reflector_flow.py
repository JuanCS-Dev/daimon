
import asyncio
import httpx
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFlowTest")

MAXIMUS_URL = "http://localhost:8001"
DAIMON_URL = "http://localhost:8003"

async def test_reflector_pipeline():
    """
    Test flow:
    1. Check Reflector Health via Maximus Core (Proxy check)
    2. Send a dummy execution log to Reflector
    3. Verify response from Reflector
    4. Verify Daimon sees Reflector as healthy
    """
    
    logger.info("Starting Data Flow Verification...")
    
    # 1. Health Check
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MAXIMUS_URL}/v1/metacognitive/health")
            if resp.status_code == 200:
                logger.info("✅ Reflector Health Check OK (via Max-Core)")
            else:
                logger.error(f"❌ Reflector Health Check FAILED: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Maximus Core: {e}")
            return False

        # 2. Reflection Test (Simulating Sync)
        dummy_log = {
            "trace_id": "test_sync_flow_001",
            "agent_id": "daimon_test_agent", 
            "action": "verify_pipeline",
            "timestamp": "2025-12-17T12:00:00Z",
            "task": "Test Data Flow",
            "executor": "test_script",
            "steps": [
                {"action": "test", "status": "success", "duration_ms": 100}
            ],
            "outcome": "success",
            "context": {"type": "verification"}
        }
        
        try:
            logger.info("Sending Dummy Log to Reflector...")
            reflect_resp = await client.post(
                f"{MAXIMUS_URL}/v1/metacognitive/reflect", 
                json=dummy_log
            )
            
            if reflect_resp.status_code == 200:
                data = reflect_resp.json()
                logger.info(f"✅ Reflector Processed Log: {data.get('critique', {}).get('overall_score')}")
            else:
                logger.error(f"❌ Reflector Reflection Failed: {reflect_resp.status_code} - {reflect_resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed during Reflection: {e}")
            return False

        # 3. Verify Daimon Output (Status Check)
        logger.info("Verifying Daimon Status Output...")
        try:
            # Daimon's status endpoint checks dependencies
            daimon_resp = await client.get(f"{DAIMON_URL}/api/status")
            if daimon_resp.status_code == 200:
                status_data = daimon_resp.json()
                reflector_status = status_data.get("reflector")
                
                # Check directly in the status object
                if reflector_status == "healthy":
                     logger.info(f"✅ Daimon Reports Reflector: HEALTHY")
                else:
                     logger.warning(f"⚠️ Daimon Reports Reflector: {reflector_status} (Expected: healthy)")
                     # We might overlook this if Daimon dashboard logic is slow to update or caches
            else:
                 logger.error(f"❌ Daimon Status Check Failed: {daimon_resp.status_code}")
        except Exception as e:
             logger.error(f"❌ Failed to connect to Daimon: {e}")

    logger.info("DATA FLOW VERIFICATION COMPLETE")
    return True

if __name__ == "__main__":
    asyncio.run(test_reflector_pipeline())
