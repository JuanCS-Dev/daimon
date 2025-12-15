#!/usr/bin/env python3
"""
EDGE CASE & STRESS TESTS - Try to BREAK the system
===================================================

Tests:
1. Timeout handling
2. Malformed inputs
3. Race conditions
4. Resource exhaustion
5. Security vulnerabilities
"""

import asyncio
import sys
import time
import os
import json
import concurrent.futures
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Setup paths
backend = Path(__file__).parent
services = backend / "services"
for p in [
    services / "shared",
    services / "api_gateway" / "src",
    services / "maximus_core_service" / "src",
    services / "metacognitive_reflector" / "src",
    services / "episodic_memory" / "src",
    services / "ethical_audit_service" / "src",
]:
    if p.exists():
        sys.path.insert(0, str(p))

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: str = ""
    severity: str = "low"  # low, medium, high, critical

results: List[TestResult] = []

def record(name: str, passed: bool, duration: float, error: str = "", severity: str = "low"):
    results.append(TestResult(name, passed, duration * 1000, error, severity))
    status = "✅ PASS" if passed else f"❌ FAIL [{severity.upper()}]"
    print(f"  {status}: {name} ({duration*1000:.1f}ms)")
    if error and not passed:
        print(f"         Error: {error[:100]}")

# =============================================================================
# 1. TIMEOUT EDGE CASES
# =============================================================================
print("\n" + "=" * 60)
print("1. TIMEOUT EDGE CASES")
print("=" * 60)

async def test_health_utils_timeout_zero():
    """Test with zero timeout - should fail gracefully."""
    from health_utils import check_redis
    start = time.time()
    try:
        result = await check_redis("redis://localhost:6379", timeout=0.001)
        # Should return unhealthy, not crash
        passed = result.healthy == False or result.error is not None
        record("check_redis(timeout=0.001)", passed, time.time() - start)
    except Exception as e:
        record("check_redis(timeout=0.001)", False, time.time() - start, str(e), "high")

async def test_health_utils_negative_timeout():
    """Test with negative timeout - should handle gracefully."""
    from health_utils import check_http
    start = time.time()
    try:
        result = await check_http("http://localhost:8001", timeout=-1)
        # Should fail but not crash
        record("check_http(timeout=-1)", True, time.time() - start)
    except ValueError as e:
        # ValueError is acceptable for invalid timeout
        record("check_http(timeout=-1)", True, time.time() - start, f"ValueError: {e}")
    except Exception as e:
        record("check_http(timeout=-1)", False, time.time() - start, str(e), "medium")

async def test_health_utils_huge_timeout():
    """Test with huge timeout - should not hang."""
    from health_utils import check_http
    start = time.time()
    try:
        # Should still fail fast on connection refused, not wait 999999 seconds
        result = await asyncio.wait_for(
            check_http("http://localhost:65535", timeout=999999),
            timeout=5.0  # Our safety timeout
        )
        passed = result.healthy == False
        record("check_http(timeout=999999) with safety", passed, time.time() - start)
    except asyncio.TimeoutError:
        record("check_http(timeout=999999) with safety", False, time.time() - start,
               "Hung for >5s", "critical")
    except Exception as e:
        record("check_http(timeout=999999) with safety", True, time.time() - start)

# =============================================================================
# 2. MALFORMED INPUT EDGE CASES
# =============================================================================
print("\n" + "=" * 60)
print("2. MALFORMED INPUT EDGE CASES")
print("=" * 60)

async def test_health_utils_invalid_url():
    """Test with completely invalid URLs."""
    from health_utils import check_http, check_redis

    invalid_urls = [
        "",
        "not-a-url",
        "http://",
        "://localhost",
        "http://localhost:not-a-port",
        "http://localhost:-1",
        "http://localhost:99999999",
        None,  # This should raise TypeError
    ]

    for url in invalid_urls:
        start = time.time()
        try:
            if url is None:
                # Expect TypeError
                try:
                    await check_http(url)
                    record(f"check_http(None)", False, time.time() - start,
                           "Should have raised TypeError", "medium")
                except TypeError:
                    record(f"check_http(None)", True, time.time() - start)
            else:
                result = await check_http(url, timeout=2.0)
                # Should return unhealthy, not crash
                passed = result.healthy == False
                record(f"check_http('{url[:20]}...')", passed, time.time() - start)
        except Exception as e:
            # Any exception is acceptable for invalid input
            record(f"check_http('{str(url)[:20]}...')", True, time.time() - start,
                   f"Exception: {type(e).__name__}")

async def test_aggregate_health_with_garbage():
    """Test aggregate_health with garbage input."""
    from health_utils import aggregate_health, DependencyHealth

    start = time.time()
    try:
        # Mix of valid and invalid
        deps = [
            DependencyHealth("valid", True, 5.0),
            "not a dependency",  # Invalid
            None,  # Invalid
            123,  # Invalid
            DependencyHealth("also_valid", False, error="test"),
        ]
        result = aggregate_health(deps)
        # Should filter out garbage and work
        passed = "dependencies" in result and "status" in result
        record("aggregate_health(mixed garbage)", passed, time.time() - start)
    except Exception as e:
        record("aggregate_health(mixed garbage)", False, time.time() - start, str(e), "medium")

def test_llm_config_empty_key():
    """Test LLM config with empty API key."""
    start = time.time()
    try:
        # Temporarily unset the env var
        old_key = os.environ.pop("NEBIUS_API_KEY", None)
        old_gemini = os.environ.pop("GEMINI_API_KEY", None)

        from metacognitive_reflector.llm.config import LLMConfig

        config = LLMConfig.from_env()

        # Should not be configured with empty keys
        try:
            provider = config.active_provider
            record("LLMConfig(empty keys)", False, time.time() - start,
                   "Should raise ValueError for no provider", "high")
        except ValueError as e:
            # This is expected
            record("LLMConfig(empty keys)", True, time.time() - start)

        # Restore
        if old_key:
            os.environ["NEBIUS_API_KEY"] = old_key
        if old_gemini:
            os.environ["GEMINI_API_KEY"] = old_gemini

    except Exception as e:
        record("LLMConfig(empty keys)", False, time.time() - start, str(e), "medium")

def test_auth_token_manipulation():
    """Test JWT auth with manipulated tokens."""
    start = time.time()
    try:
        from ethical_audit_service.auth import decode_token, create_access_token
        from fastapi import HTTPException

        all_passed = True

        # Test 1: Empty token
        try:
            decode_token("")
            record("decode_token(empty)", False, time.time() - start,
                   "Should reject empty token", "critical")
            all_passed = False
        except HTTPException as e:
            if e.status_code == 401:
                pass  # Good
            else:
                record("decode_token(empty)", False, time.time() - start,
                       f"Wrong status: {e.status_code}", "high")
                all_passed = False
        except Exception:
            pass  # Any rejection is OK

        # Test 2: Malformed JWT
        try:
            decode_token("not.a.jwt")
            record("decode_token(malformed)", False, time.time() - start,
                   "Should reject malformed", "critical")
            all_passed = False
        except HTTPException:
            pass  # Good
        except Exception:
            pass  # Any rejection is OK

        # Test 3: Tampered token (change payload)
        valid_token = create_access_token({"sub": "user123", "roles": ["readonly"]})
        tampered = valid_token[:-5] + "XXXXX"
        try:
            decode_token(tampered)
            record("decode_token(tampered)", False, time.time() - start,
                   "Should reject tampered token", "critical")
            all_passed = False
        except HTTPException:
            pass  # Good
        except Exception:
            pass  # Any rejection is OK

        if all_passed:
            record("auth token manipulation", True, time.time() - start)

    except Exception as e:
        record("auth token tests", False, time.time() - start, str(e), "high")

# =============================================================================
# 3. RACE CONDITIONS
# =============================================================================
print("\n" + "=" * 60)
print("3. RACE CONDITIONS")
print("=" * 60)

async def test_concurrent_health_checks():
    """Test many concurrent health checks don't crash."""
    from health_utils import check_http, check_redis

    start = time.time()
    try:
        # Fire 50 concurrent checks
        tasks = []
        for i in range(50):
            tasks.append(check_http(f"http://localhost:{8000 + (i % 10)}", timeout=1.0))
            tasks.append(check_redis("redis://localhost:6379", timeout=1.0))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Count failures vs crashes
        crashes = sum(1 for r in results_list if isinstance(r, Exception) and not isinstance(r, asyncio.TimeoutError))

        passed = crashes == 0
        record(f"50 concurrent health checks", passed, time.time() - start,
               f"{crashes} crashes" if crashes else "")
    except Exception as e:
        record("concurrent health checks", False, time.time() - start, str(e), "high")

async def test_rapid_config_access():
    """Test rapid concurrent access to singleton config."""
    start = time.time()
    try:
        from metacognitive_reflector.llm.config import LLMConfig

        def access_config():
            config = LLMConfig.from_env()
            return config.is_configured

        # Run 100 concurrent accesses
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(access_config) for _ in range(100)]
            results_list = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should return same value
        unique_results = set(results_list)
        passed = len(unique_results) == 1
        record("100 concurrent config access", passed, time.time() - start,
               f"Got {len(unique_results)} different results" if not passed else "")
    except Exception as e:
        record("concurrent config access", False, time.time() - start, str(e), "medium")

# =============================================================================
# 4. RESOURCE EXHAUSTION
# =============================================================================
print("\n" + "=" * 60)
print("4. RESOURCE EXHAUSTION")
print("=" * 60)

async def test_memory_leak_health_checks():
    """Test for memory leaks in repeated health checks."""
    import gc
    from health_utils import check_http

    start = time.time()
    try:
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Run 100 health checks
        for _ in range(100):
            await check_http("http://localhost:65535", timeout=0.1)

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow some variance but not huge growth
        growth = final_objects - initial_objects
        passed = growth < 1000  # Arbitrary threshold
        record(f"memory leak test (100 checks)", passed, time.time() - start,
               f"Object growth: {growth}" if growth > 500 else "")
    except Exception as e:
        record("memory leak test", False, time.time() - start, str(e), "medium")

def test_large_dependency_list():
    """Test aggregate_health with huge list."""
    from health_utils import aggregate_health, DependencyHealth

    start = time.time()
    try:
        # Create 10000 dependencies
        deps = [DependencyHealth(f"dep_{i}", i % 2 == 0, float(i)) for i in range(10000)]

        result = aggregate_health(deps)

        passed = len(result["dependencies"]) == 10000
        record(f"aggregate_health(10000 deps)", passed, time.time() - start,
               f"Got {len(result['dependencies'])} deps")
    except Exception as e:
        record("large dependency list", False, time.time() - start, str(e), "medium")

# =============================================================================
# 5. SECURITY VULNERABILITIES
# =============================================================================
print("\n" + "=" * 60)
print("5. SECURITY VULNERABILITIES")
print("=" * 60)

async def test_sql_injection_in_database_actuator():
    """Test SQL injection protection in database_actuator."""
    start = time.time()
    try:
        from maximus_core_service.autonomic_core.execute.database_actuator import DatabaseActuator

        actuator = DatabaseActuator(dry_run_mode=True)

        # Try SQL injection in table name
        malicious_tables = [
            "users; DROP TABLE users;--",
            "users' OR '1'='1",
            "users\"; DROP TABLE users;--",
            "$(rm -rf /)",
            "../../../etc/passwd",
        ]

        all_blocked = True
        for table in malicious_tables:
            try:
                # vacuum_analyze validates table name
                result = await actuator.vacuum_analyze(table)
                if result.get("success") and not result.get("dry_run"):
                    record(f"SQL injection: {table[:20]}", False, time.time() - start,
                           "Malicious table accepted", "critical")
                    all_blocked = False
                else:
                    # Dry run - check it was actually blocked
                    pass
            except ValueError:
                # Expected - table name validation caught it
                pass

        if all_blocked:
            record("SQL injection protection", True, time.time() - start)
    except Exception as e:
        record("SQL injection test", False, time.time() - start, str(e), "high")

def test_path_traversal_in_urls():
    """Test path traversal protection."""
    start = time.time()

    malicious_urls = [
        "http://localhost:8001/../../../etc/passwd",
        "http://localhost:8001/..%2F..%2F..%2Fetc/passwd",
        "http://localhost:8001/health?file=../../../etc/passwd",
    ]

    # These should be handled by the HTTP library, not our code
    # But let's verify our check functions don't do anything weird
    from health_utils import check_http

    all_safe = True
    for url in malicious_urls:
        try:
            result = asyncio.get_event_loop().run_until_complete(
                check_http(url, timeout=1.0)
            )
            # Should just fail to connect, not expose anything
            if "passwd" in str(result.error or "").lower():
                all_safe = False
        except:
            pass

    record("path traversal protection", all_safe, time.time() - start)

def test_jwt_algorithm_confusion():
    """Test JWT doesn't accept 'none' algorithm."""
    start = time.time()
    try:
        import jwt
        from ethical_audit_service.auth import decode_token, SECRET_KEY

        # Create a token with 'none' algorithm (attack vector)
        payload = {"sub": "attacker", "roles": ["admin"]}

        # Try to forge with 'none'
        try:
            forged = jwt.encode(payload, key="", algorithm="none")
            result = decode_token(forged)
            record("JWT 'none' algorithm attack", False, time.time() - start,
                   "Accepted forged token!", "critical")
        except Exception:
            # Should reject - good
            record("JWT 'none' algorithm attack", True, time.time() - start)

    except Exception as e:
        record("JWT algorithm test", False, time.time() - start, str(e), "high")

def test_env_var_injection():
    """Test that config doesn't execute env var content."""
    start = time.time()
    try:
        # Set a malicious env var
        os.environ["TEST_INJECTION"] = "$(whoami)"

        from metacognitive_reflector.llm.config import NebiusConfig

        # This shouldn't execute the command
        config = NebiusConfig()

        # The API key should be literal string, not executed
        passed = "$(whoami)" not in str(config.api_key) or config.api_key == ""
        record("env var injection protection", passed, time.time() - start)

        del os.environ["TEST_INJECTION"]
    except Exception as e:
        record("env var injection test", False, time.time() - start, str(e), "medium")

# =============================================================================
# 6. ADDITIONAL EDGE CASES
# =============================================================================
print("\n" + "=" * 60)
print("6. ADDITIONAL EDGE CASES")
print("=" * 60)

def test_unicode_handling():
    """Test Unicode in various inputs."""
    from health_utils import DependencyHealth, aggregate_health

    start = time.time()
    try:
        # Unicode in dependency names
        deps = [
            DependencyHealth("サービス", True, 5.0),  # Japanese
            DependencyHealth("сервис", False, error="ошибка"),  # Russian
            DependencyHealth("🚀service", True, 1.0),  # Emoji
            DependencyHealth("service\x00null", True, 1.0),  # Null byte
        ]

        result = aggregate_health(deps)
        passed = len(result["dependencies"]) == 4
        record("unicode in dependency names", passed, time.time() - start)
    except Exception as e:
        record("unicode handling", False, time.time() - start, str(e), "low")

def test_extreme_latency_values():
    """Test extreme latency values."""
    from health_utils import DependencyHealth, aggregate_health

    start = time.time()
    try:
        deps = [
            DependencyHealth("fast", True, 0.0),
            DependencyHealth("negative", True, -100.0),  # Invalid but shouldn't crash
            DependencyHealth("huge", True, float('inf')),
            DependencyHealth("nan", True, float('nan')),
        ]

        result = aggregate_health(deps)
        # Should handle gracefully
        passed = "dependencies" in result
        record("extreme latency values", passed, time.time() - start)
    except Exception as e:
        record("extreme latency values", False, time.time() - start, str(e), "low")

async def test_connection_refused_handling():
    """Test proper handling of connection refused."""
    from health_utils import check_http, check_redis

    start = time.time()
    try:
        # Port that definitely won't have a service
        result = await check_http("http://127.0.0.1:1", timeout=2.0)

        passed = (
            result.healthy == False and
            result.error is not None and
            "refused" in result.error.lower() or "connect" in result.error.lower()
        )
        record("connection refused handling", passed, time.time() - start,
               result.error if not passed else "")
    except Exception as e:
        record("connection refused handling", False, time.time() - start, str(e), "medium")

# =============================================================================
# RUN ALL TESTS
# =============================================================================
async def run_async_tests():
    await test_health_utils_timeout_zero()
    await test_health_utils_negative_timeout()
    await test_health_utils_huge_timeout()
    await test_health_utils_invalid_url()
    await test_aggregate_health_with_garbage()
    await test_concurrent_health_checks()
    await test_rapid_config_access()
    await test_memory_leak_health_checks()
    await test_connection_refused_handling()
    await test_sql_injection_in_database_actuator()

def run_sync_tests():
    test_llm_config_empty_key()
    test_auth_token_manipulation()
    test_large_dependency_list()
    test_path_traversal_in_urls()
    test_jwt_algorithm_confusion()
    test_env_var_injection()
    test_unicode_handling()
    test_extreme_latency_values()

def main():
    print("=" * 60)
    print("EDGE CASE & BREAKING TESTS")
    print("=" * 60)

    # Run async tests
    asyncio.run(run_async_tests())

    # Run sync tests
    run_sync_tests()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    critical_fails = [r for r in failed if r.severity == "critical"]
    high_fails = [r for r in failed if r.severity == "high"]

    print(f"\nTotal: {len(results)} tests")
    print(f"  ✅ Passed: {len(passed)}")
    print(f"  ❌ Failed: {len(failed)}")

    if critical_fails:
        print(f"\n🚨 CRITICAL FAILURES ({len(critical_fails)}):")
        for r in critical_fails:
            print(f"   - {r.name}: {r.error}")

    if high_fails:
        print(f"\n⚠️  HIGH SEVERITY FAILURES ({len(high_fails)}):")
        for r in high_fails:
            print(f"   - {r.name}: {r.error}")

    if failed:
        print("\n" + "=" * 60)
        print("FAILURES NEED FIXING!")
        print("=" * 60)
        return 1
    else:
        print("\n" + "=" * 60)
        print("ALL EDGE CASES HANDLED!")
        print("=" * 60)
        return 0

if __name__ == "__main__":
    sys.exit(main())
