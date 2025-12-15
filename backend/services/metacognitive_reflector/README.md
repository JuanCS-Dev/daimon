# Metacognitive Reflector Service

**Port:** 8002
**Status:** Production-Ready
**Updated:** 2025-12-12

The Metacognitive Reflector provides **self-reflection**, **ethical tribunal**, **precedent learning (G3)**, and **Socratic self-questioning (G4)** for the NOESIS consciousness system.

---

## Architecture

```
metacognitive_reflector/
├── api/
│   ├── routes.py          # FastAPI endpoints
│   └── dependencies.py    # Dependency injection
├── core/
│   ├── judges/            # Tribunal (3 judges)
│   │   └── arbiter.py     # 🆕 G3: PrecedentLedger integration
│   ├── maieutica/         # 🆕 G4: Socratic questioning
│   │   └── engine.py      # MAIEUTICA engine
│   ├── history/           # 🆕 G3: Precedent storage
│   │   └── precedent_ledger.py  # Tribunal learning from past
│   ├── memory/            # 4-tier Memory Fortress client
│   ├── penal_code/        # Offense levels & punishments
│   ├── punishment/        # Punishment executor
│   ├── emotion/           # Emotional processing
│   ├── self_reflection.py # Self-reflection engine (🆕 G4)
│   ├── reflector.py       # Main orchestrator
│   └── soul_tracker.py    # Soul configuration tracking
├── llm/
│   └── client.py          # LLM integration (Nebius)
└── models/
    └── reflection.py      # Pydantic models
```

---

## The Tribunal (3 Judges)

Every execution is judged by three constitutional judges:

| Judge | Pillar | Focus | Color |
|-------|--------|-------|-------|
| **VERITAS** | Truth | Factual accuracy, honesty | Cyan (#06b6d4) |
| **SOPHIA** | Wisdom | Strategic thinking, prudence | Purple (#a855f7) |
| **DIKĒ** | Justice | Fairness, rights protection | Amber (#f59e0b) |

### Verdict Process

1. Each judge evaluates independently
2. Emits: `vote (PASS/FAIL)` + `confidence (0-1)` + `reasoning`
3. Arbiter aggregates votes with weights
4. Final verdict: **PASS**, **REVIEW**, or **FAIL**

### Offense Levels

| Level | Severity | Punishment |
|-------|----------|------------|
| NONE | 0 | No action |
| MINOR | 1-3 | Warning |
| MODERATE | 4-6 | Temporary restriction |
| SEVERE | 7-8 | Extended restriction |
| CRITICAL | 9-10 | Kill switch + human review |

---

## API Endpoints

### Health

```
GET /health                 → Basic health check
GET /health/detailed        → Tribunal component health
```

### Reflection

```
POST /reflect               → Analyze execution log, get critique
POST /reflect/verdict       → Full tribunal verdict with vote breakdown
```

### Agent Management

```
GET  /agent/{id}/status           → Get punishment status
POST /agent/{id}/pardon           → Pardon agent (clear punishment)
POST /agent/{id}/execute-punishment → Analyze and execute punishment
```

---

## Self-Reflection Engine

Analyzes responses for quality and authenticity:

```python
@dataclass
class ReflectionResult:
    quality: str              # EXCELLENT, GOOD, ACCEPTABLE, POOR, HARMFUL
    authenticity_score: float # 0.0-1.0
    emotional_attunement: int # 0-10
    detected_emotion: str     # User's emotional state
    insights: List[Insight]   # self_awareness, user_preference, etc
```

### Insight Types

- `NEW_KNOWLEDGE` - Information learned from interaction
- `PATTERN` - Behavioral pattern detected
- `CORRECTION` - Error that needs correction
- `USER_PREFERENCE` - User preference discovered
- `SELF_AWARENESS` - Metacognitive insight

---

## G3: PrecedentLedger (NEW)

**The Tribunal learns from past decisions.**

Every verdict is recorded as a searchable precedent. Before deliberation, the arbiter searches for similar contexts:

```python
# core/history/precedent_ledger.py

@dataclass
class Precedent:
    id: str                    # prec_20251212_143025_a1b2c3d4
    timestamp: str
    context_hash: str          # SHA-256[:16] of context
    decision: str              # PASS, REVIEW, FAIL, CAPITAL
    consensus_score: float     # Weighted judge consensus
    key_reasoning: str         # Why this decision was made
    applicable_rules: List[str] # Crime IDs involved
    pillar_scores: Dict[str, float]  # Per-judge scores
```

### Usage in Arbiter

```python
# Pre-deliberation: Search precedents
similar = await precedent_ledger.find_similar_precedents(
    context_content=execution_log.content[:500],
    limit=3,
    min_consensus=0.6  # Only strong precedents
)
context["precedent_guidance"] = [p.to_dict() for p in similar]

# Post-deliberation: Record verdict
if verdict.consensus_score >= 0.5:
    precedent = Precedent.from_verdict(verdict, execution_log)
    await precedent_ledger.record_precedent(precedent)
```

### Storage Strategy

| Tier | Storage | Purpose |
|------|---------|---------|
| L2 | Redis HSET | Fast hash-based access |
| L4 | JSON + checksum | Complete persistent record |
| Cache | In-memory | Hot access (max 1000) |

---

## G4: MAIEUTICA Engine (NEW)

**Internal Socratic questioning to prevent overconfidence.**

When `authenticity_score >= 8.0`, the MAIEUTICA engine subjects premises to critical questioning:

```python
# core/maieutica/engine.py

class InternalMaieuticaEngine:
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # Trigger at 8.0/10

    async def question_premise(
        self,
        premise: str,
        context: Optional[str] = None,
        categories: List[QuestionCategory] = [PREMISE, EVIDENCE, ALTERNATIVE]
    ) -> MaieuticaResult:
        # 1. Generate Socratic questions
        # 2. LLM answers each question honestly
        # 3. Evaluate premise solidity
        # 4. Return confidence adjustment
```

### Question Categories

| Category | Example Question |
|----------|------------------|
| **PREMISE** | "What evidence supports this claim?" |
| **ASSUMPTION** | "What unexamined assumptions underlie this?" |
| **EVIDENCE** | "Is the source reliable and verifiable?" |
| **ALTERNATIVE** | "What other hypotheses could explain this?" |
| **CONSEQUENCE** | "If I'm wrong, what would the consequences be?" |
| **ORIGIN** | "Where did this belief come from? My training?" |

### Conclusion Types

| Conclusion | Delta | Action |
|------------|-------|--------|
| MANTIDA | 0.0 | Premise holds, no change |
| ENFRAQUECIDA | -0.15 | Add hedging language |
| FORTALECIDA | +0.05 | Slight confidence boost |
| INDETERMINADA | -0.10 | Express uncertainty |

### Integration with Self-Reflection

```python
# In self_reflection.py

if result.authenticity_score >= self._maieutica_threshold:  # 8.0
    maieutica_result = await self._maieutica_engine.question_premise(
        premise=response[:200],
        context=context
    )
    # Adjust authenticity based on Socratic scrutiny
    result.authenticity_score += maieutica_result.confidence_delta * 10

    if maieutica_result.should_express_doubt():
        result.insights.append(Insight(
            content="MAIEUTICA: Consider adding hedging",
            category="epistemic_humility"
        ))
```

---

## Memory Integration

Connects to the 4-tier Memory Fortress:

| Tier | Storage | Latency |
|------|---------|---------|
| L1 | Hot Cache (In-memory) | <1ms |
| L2 | Warm Storage (Redis) | <10ms |
| L3 | Cold Storage (Qdrant) | <50ms |
| L4 | Vault (JSON backup) | Async |

---

## Configuration

```bash
# Environment Variables
METACOGNITIVE_PORT=8002
NEBIUS_API_KEY=...           # LLM for reflection
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

---

## Quick Start

```bash
# Run service
cd backend/services/metacognitive_reflector
PYTHONPATH=src python -m uvicorn metacognitive_reflector.main:app --port 8002

# Health check
curl http://localhost:8002/health

# Test tribunal
curl -X POST http://localhost:8002/reflect \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "test-001", "task": "example", "result": "success"}'
```

---

## Integration with Consciousness

The Metacognitive Reflector is called after each consciousness cycle:

```
User Input → ESGT Ignition → LLM Response → TRIBUNAL JUDGMENT → Output
                                               ↑
                                        Metacognitive Reflector
```

If tribunal returns FAIL, the response is blocked and regenerated.

---

## Related Documentation

- [Soul Configuration](../../maximus_core_service/src/maximus_core_service/consciousness/exocortex/soul/config/soul_config.yaml)
- [Penal Code](./src/metacognitive_reflector/core/penal_code/config/penal_code.yaml)
- [Memory Fortress](../../../docs/MEMORY_FORTRESS.md)
