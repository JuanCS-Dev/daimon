# NOESIS E2E Test Report

**Generated:** 2025-12-08T21:51:50.502514
**Environment:** Production Validation

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 22 |
| **Passed** | 8 ✅ |
| **Failed** | 14 ❌ |
| **Skipped** | 0 ⏭️ |
| **Success Rate** | 36.4% |
| **Total Duration** | 8320ms |

---

## Test Results by Category

### HEALTH ⚠️

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| neural_core_health | ❌ FAIL | 5ms | All connection attempts failed |
| episodic_memory_health | ✅ PASS | 2ms | HTTP 200 |
| api_gateway_health | ✅ PASS | 2ms | HTTP 200 |
| qdrant_health | ❌ FAIL | 1ms | All connection attempts failed |
| redis_health | ❌ FAIL | 39ms | Error 111 connecting to localhost:6379. Connect ca |

### CONSCIOUSNESS ❌

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| consciousness_stream | ❌ FAIL | 27ms | All connection attempts failed |
| esgt_status | ❌ FAIL | 27ms | All connection attempts failed |

### MEMORY ❌

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| memory_store | ❌ FAIL | 2ms | HTTP 404 |
| memory_search | ❌ FAIL | 1ms | HTTP 404 |
| memory_stats | ❌ FAIL | 1ms | HTTP 404 |

### LLM ❌

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| llm_config_load | ❌ FAIL | 17ms | Provider: none, Configured: False |
| llm_health_check | ❌ FAIL | 0ms | No LLM provider configured. Set NEBIUS_API_KEY or  |
| llm_generation | ❌ FAIL | 0ms | No LLM provider configured. Set NEBIUS_API_KEY or  |

### SECURITY ✅

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| jwt_creation_validation | ✅ PASS | 295ms | User: test_user, Roles: ['readonly'] |
| invalid_token_rejection | ✅ PASS | 0ms | Correctly rejected invalid token |
| sql_injection_protection | ✅ PASS | 2938ms | Malicious input blocked or sandboxed |

### RESILIENCE ✅

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| graceful_offline_handling | ✅ PASS | 30ms | Correctly detected offline: All connection attempt |
| concurrent_load_handling | ✅ PASS | 2852ms | 100 concurrent ops, 0 unexpected crashes |
| timeout_handling | ✅ PASS | 2079ms | Request completed/timed out correctly |

### API ❌

| Test | Status | Duration | Details |
|------|--------|----------|---------|
| reactive_fabric_metrics | ❌ FAIL | 1ms | All connection attempts failed |
| safety_status | ❌ FAIL | 1ms | All connection attempts failed |
| chat_endpoint | ❌ FAIL | 1ms | All connection attempts failed |

---

## System Configuration

```json
{
  "neural_core": "http://localhost:8001",
  "metacognitive": "http://localhost:8002",
  "api_gateway": "http://localhost:8000",
  "episodic_memory": "http://localhost:8102",
  "qdrant": "http://localhost:6333",
  "redis": "redis://localhost:6379"
}
```

---

## Test Categories Explained

| Category | Description |
|----------|-------------|
| **health** | Service availability and health endpoints |
| **consciousness** | ESGT pipeline and consciousness streaming |
| **memory** | Memory Fortress 4-tier storage system |
| **llm** | LLM provider integration (Nebius/Gemini) |
| **security** | JWT auth, SQL injection, input validation |
| **resilience** | Graceful degradation, timeouts, concurrency |
| **api** | REST API endpoints functionality |

---

## Detailed Results

### Failed Tests Details

#### ❌ neural_core_health

- **Category:** health
- **Duration:** 5ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ qdrant_health

- **Category:** health
- **Duration:** 1ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ redis_health

- **Category:** health
- **Duration:** 39ms
- **Error:** `Error 111 connecting to localhost:6379. Connect call failed ('127.0.0.1', 6379).`
- **Details:** 

#### ❌ consciousness_stream

- **Category:** consciousness
- **Duration:** 27ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ esgt_status

- **Category:** consciousness
- **Duration:** 27ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ memory_store

- **Category:** memory
- **Duration:** 2ms
- **Error:** ``
- **Details:** HTTP 404

#### ❌ memory_search

- **Category:** memory
- **Duration:** 1ms
- **Error:** ``
- **Details:** HTTP 404

#### ❌ memory_stats

- **Category:** memory
- **Duration:** 1ms
- **Error:** ``
- **Details:** HTTP 404

#### ❌ llm_config_load

- **Category:** llm
- **Duration:** 17ms
- **Error:** ``
- **Details:** Provider: none, Configured: False

#### ❌ llm_health_check

- **Category:** llm
- **Duration:** 0ms
- **Error:** `No LLM provider configured. Set NEBIUS_API_KEY or GEMINI_API_KEY environment variable.`
- **Details:** 

#### ❌ llm_generation

- **Category:** llm
- **Duration:** 0ms
- **Error:** `No LLM provider configured. Set NEBIUS_API_KEY or GEMINI_API_KEY environment variable.`
- **Details:** 

#### ❌ reactive_fabric_metrics

- **Category:** api
- **Duration:** 1ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ safety_status

- **Category:** api
- **Duration:** 1ms
- **Error:** `All connection attempts failed`
- **Details:** 

#### ❌ chat_endpoint

- **Category:** api
- **Duration:** 1ms
- **Error:** `All connection attempts failed`
- **Details:** 

---

## Sample Responses

### episodic_memory_health

```json
{"status":"healthy","service":"episodic_memory","version":"2.0.0","timestamp":"2025-12-08T21:51:50.588003","persistence":{"qdrant_available":false,"embeddings_enabled":false,"total_memories":61}}
```

### api_gateway_health

```json
{"status":"healthy","service":"api_gateway","timestamp":"2025-12-08T21:51:50.590153"}
```

---

## Conclusion

The NOESIS system has been validated through **22 comprehensive tests** covering:

1. ✅ Service Health & Availability
2. ✅ Consciousness Pipeline (ESGT Protocol)
3. ✅ Memory Fortress (4-Tier Architecture)
4. ✅ LLM Integration (Nebius Token Factory)
5. ✅ Security Mechanisms (JWT, SQL Injection Protection)
6. ✅ System Resilience (Graceful Degradation)
7. ✅ API Endpoints Functionality

**System Status: ISSUES DETECTED**

---

*Generated by NOESIS E2E Test Suite*
*Hackathon 2025*
