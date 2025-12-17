/**
 * API Configuration - Uses env vars with localhost fallback
 *
 * Environment variables (set in .env.local):
 * - NEXT_PUBLIC_CONSCIOUSNESS_URL: Neural core service URL
 * - NEXT_PUBLIC_TRIBUNAL_URL: Metacognitive reflector URL
 * - NEXT_PUBLIC_WS_URL: WebSocket URL for real-time updates
 * - NEXT_PUBLIC_API_URL: API Gateway URL
 */

export const API_CONFIG = {
  // Neural Core (Reactive Fabric)
  CONSCIOUSNESS_URL:
    process.env.NEXT_PUBLIC_CONSCIOUSNESS_URL || "http://localhost:8001",

  // Metacognitive Reflector (Tribunal)
  TRIBUNAL_URL:
    process.env.NEXT_PUBLIC_TRIBUNAL_URL || "http://localhost:8002",

  // WebSocket for real-time consciousness state
  WS_URL:
    process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8001",

  // API Gateway
  API_URL:
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",

  // Memory Service
  MEMORY_URL:
    process.env.NEXT_PUBLIC_MEMORY_URL || "http://localhost:8102",
} as const;

// Derived URLs for common endpoints
export const ENDPOINTS = {
  // Consciousness streaming
  CONSCIOUSNESS_STREAM: `${API_CONFIG.CONSCIOUSNESS_URL}/api/consciousness/stream/process`,

  // WebSocket consciousness
  WS_CONSCIOUSNESS: `${API_CONFIG.WS_URL}/api/consciousness/ws`,

  // Metrics
  REACTIVE_FABRIC_METRICS: `${API_CONFIG.CONSCIOUSNESS_URL}/api/consciousness/reactive-fabric/metrics`,
  SAFETY_STATUS: `${API_CONFIG.CONSCIOUSNESS_URL}/api/consciousness/safety/status`,

  // Tribunal
  TRIBUNAL_CHAT: `${API_CONFIG.TRIBUNAL_URL}/api/tribunal/chat`,
} as const;

export default API_CONFIG;
