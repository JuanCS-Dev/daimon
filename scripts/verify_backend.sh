#!/bin/bash
# =============================================================================
# NOESIS BACKEND VERIFICATION
# =============================================================================
# Verifica saúde de todos os serviços do backend
# Usage: ./scripts/verify_backend.sh
# =============================================================================

set -e

echo "=== NOESIS BACKEND VERIFICATION ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_service() {
    local name=$1
    local url=$2
    local endpoint=${3:-"/health"}

    printf "%-25s" "$name:"

    if response=$(curl -s --connect-timeout 3 --max-time 5 "$url$endpoint" 2>/dev/null); then
        if echo "$response" | grep -qi "healthy\|ok\|status"; then
            echo -e " ${GREEN}ONLINE${NC}"
            return 0
        else
            echo -e " ${YELLOW}DEGRADED${NC} (response: ${response:0:50}...)"
            return 1
        fi
    else
        echo -e " ${RED}OFFLINE${NC}"
        return 1
    fi
}

check_redis() {
    printf "%-25s" "Redis (6379):"
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e " ${GREEN}ONLINE${NC}"
        return 0
    else
        echo -e " ${RED}OFFLINE${NC}"
        return 1
    fi
}

check_qdrant() {
    printf "%-25s" "Qdrant (6333):"
    if response=$(curl -s --connect-timeout 3 --max-time 5 "http://localhost:6333/healthz" 2>/dev/null); then
        echo -e " ${GREEN}ONLINE${NC}"
        return 0
    else
        echo -e " ${RED}OFFLINE${NC}"
        return 1
    fi
}

echo "--- Infrastructure ---"
check_redis || true
check_qdrant || true

echo ""
echo "--- Core Services ---"
check_service "Neural Core (8001)" "http://localhost:8001" "/v1/health" || true
check_service "Episodic Memory (8102)" "http://localhost:8102" "/health" || true
check_service "Metacognitive (8002)" "http://localhost:8002" "/health" || true
check_service "API Gateway (8000)" "http://localhost:8000" "/health" || true

echo ""
echo "--- Optional Services ---"
check_service "PFC Service (8005)" "http://localhost:8005" "/health" || true
check_service "Ethical Audit (8006)" "http://localhost:8006" "/health" || true

echo ""
echo "=== VERIFICATION COMPLETE ==="
echo ""
echo "To start services:"
echo "  ./noesis start      # Start core services"
echo "  ./noesis status     # Check detailed status"
