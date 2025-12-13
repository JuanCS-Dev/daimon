"""
DAIMON User Model Service
=========================

Service for managing user model persistence.

Provides:
- CRUD operations for UserModel
- NOESIS persistence with local fallback
- CLAUDE.md synchronization

Architecture:
    ┌─────────────────┐
    │   UserModel     │
    │  ┌───────────┐  │
    │  │Preferences│  │   Primary: NOESIS /v1/memory/user_model
    │  ├───────────┤  │   Fallback: ~/.daimon/user_model.json
    │  │Cognitive  │  │
    │  ├───────────┤  │   Sync: CLAUDE.md ↔ UserModel
    │  │Patterns   │  │
    │  └───────────┘  │
    └─────────────────┘

Usage:
    from memory.user_model import get_user_model_service
    
    service = get_user_model_service()
    model = service.load("default")
    model = service.update_preferences("default", {"code_style": "documented"})

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .user_models import (
    UserPreferences,
    CognitiveProfile,
    UserModel,
    MAX_PATTERNS,
)

logger = logging.getLogger("daimon.user_model")

# Default storage paths
DAIMON_DIR = Path.home() / ".daimon"
DEFAULT_USER_MODEL_PATH = DAIMON_DIR / "user_model.json"


class UserModelService:
    """
    Service for managing UserModel persistence.
    
    Provides CRUD operations with:
    - Primary storage: NOESIS /v1/memory/user_model
    - Fallback storage: Local JSON file
    - Sync with CLAUDE.md
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        noesis_url: Optional[str] = None,
    ):
        """
        Initialize service.
        
        Args:
            storage_path: Local storage path (default: ~/.daimon/user_model.json)
            noesis_url: NOESIS base URL (default: from config)
        """
        self.storage_path = storage_path or DEFAULT_USER_MODEL_PATH
        self.noesis_url = noesis_url
        self._cache: Dict[str, UserModel] = {}
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self, user_id: str = "default") -> UserModel:
        """
        Load user model from storage.
        
        Tries NOESIS first, falls back to local storage.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserModel (creates default if not found)
        """
        if user_id in self._cache:
            return self._cache[user_id]
        
        model = self._load_from_noesis(user_id)
        if model:
            self._cache[user_id] = model
            return model
        
        model = self._load_from_local(user_id)
        if model:
            self._cache[user_id] = model
            return model
        
        model = UserModel(user_id=user_id)
        self._cache[user_id] = model
        return model
    
    def save(self, model: UserModel) -> bool:
        """
        Save user model to storage.
        
        Saves to both NOESIS and local for redundancy.
        """
        model.version += 1
        model.last_updated = datetime.now()
        self._cache[model.user_id] = model
        
        noesis_ok = self._save_to_noesis(model)
        local_ok = self._save_to_local(model)
        
        return noesis_ok or local_ok
    
    def update_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any],
    ) -> UserModel:
        """Update user preferences."""
        model = self.load(user_id)
        
        for key, value in updates.items():
            if hasattr(model.preferences, key):
                setattr(model.preferences, key, value)
        
        self.save(model)
        return model
    
    def update_cognitive(
        self,
        user_id: str,
        updates: Dict[str, Any],
    ) -> UserModel:
        """Update cognitive profile."""
        model = self.load(user_id)
        
        for key, value in updates.items():
            if hasattr(model.cognitive, key):
                setattr(model.cognitive, key, value)
        
        self.save(model)
        return model
    
    def add_pattern(
        self,
        user_id: str,
        pattern: Dict[str, Any],
    ) -> UserModel:
        """
        Add a learned pattern to user model.
        
        Maintains top MAX_PATTERNS patterns sorted by confidence.
        """
        model = self.load(user_id)
        model.patterns.append(pattern)
        model.patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        model.patterns = model.patterns[:MAX_PATTERNS]
        self.save(model)
        return model
    
    def sync_with_claude_md(self, model: UserModel, claude_md_path: Path) -> None:
        """Sync UserModel with CLAUDE.md file."""
        if not claude_md_path.exists():
            logger.debug("CLAUDE.md not found: %s", claude_md_path)
            return
        
        try:
            content = claude_md_path.read_text(encoding="utf-8")
            prefs_section = self._generate_claude_md_section(model)
            
            marker_start = "<!-- DAIMON:START -->"
            marker_end = "<!-- DAIMON:END -->"
            
            if marker_start in content:
                start_idx = content.index(marker_start)
                end_idx = content.index(marker_end) + len(marker_end)
                content = content[:start_idx] + prefs_section + content[end_idx:]
            else:
                content += f"\n\n{prefs_section}"
            
            claude_md_path.write_text(content, encoding="utf-8")
            logger.info("Synced UserModel to CLAUDE.md")
            
        except Exception as e:
            logger.warning("Failed to sync with CLAUDE.md: %s", e)
    
    def _generate_claude_md_section(self, model: UserModel) -> str:
        """Generate CLAUDE.md section from UserModel."""
        lines = [
            "<!-- DAIMON:START -->",
            "## Learned Preferences (DAIMON)",
            "",
            f"- **Communication:** {model.preferences.communication_style}",
            f"- **Code Style:** {model.preferences.code_style}",
            f"- **Language:** {model.preferences.language}",
            "",
            f"*Last updated: {model.last_updated.strftime('%Y-%m-%d %H:%M')}*",
            "<!-- DAIMON:END -->",
        ]
        return "\n".join(lines)
    
    def _load_from_noesis(self, user_id: str) -> Optional[UserModel]:
        """Load from NOESIS memory service."""
        if not self.noesis_url:
            try:
                from integrations.mcp_tools.config import NOESIS_MEMORY_URL
                self.noesis_url = NOESIS_MEMORY_URL
            except ImportError:
                return None
        
        try:
            from integrations.mcp_tools.http_utils import http_get
            import asyncio
            
            url = f"{self.noesis_url}/v1/memory/user_model/{user_id}"
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None
            
            response = loop.run_until_complete(http_get(url))
            
            if response and response.get("status") == "ok":
                return UserModel.from_dict(response.get("data", {}))
            
        except Exception as e:
            logger.debug("NOESIS load failed: %s", e)
        
        return None
    
    def _save_to_noesis(self, model: UserModel) -> bool:
        """Save to NOESIS memory service."""
        if not self.noesis_url:
            try:
                from integrations.mcp_tools.config import NOESIS_MEMORY_URL
                self.noesis_url = NOESIS_MEMORY_URL
            except ImportError:
                return False
        
        try:
            from integrations.mcp_tools.http_utils import http_post
            import asyncio
            
            url = f"{self.noesis_url}/v1/memory/user_model"
            payload = model.to_dict()
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return False
            
            response = loop.run_until_complete(http_post(url, payload))
            return response.get("status") == "ok"
            
        except Exception as e:
            logger.debug("NOESIS save failed: %s", e)
            return False
    
    def _load_from_local(self, user_id: str) -> Optional[UserModel]:
        """Load from local JSON file."""
        if not self.storage_path.exists():
            return None
        
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            models = data.get("models", {})
            
            if user_id in models:
                return UserModel.from_dict(models[user_id])
                
        except Exception as e:
            logger.warning("Local load failed: %s", e)
        
        return None
    
    def _save_to_local(self, model: UserModel) -> bool:
        """Save to local JSON file."""
        try:
            if self.storage_path.exists():
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            else:
                data = {"models": {}}
            
            data["models"][model.user_id] = model.to_dict()
            
            self.storage_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            return True
            
        except Exception as e:
            logger.warning("Local save failed: %s", e)
            return False
    
    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "cached_users": len(self._cache),
            "storage_path": str(self.storage_path),
            "noesis_url": self.noesis_url or "not configured",
        }


# Singleton instance
_service: Optional[UserModelService] = None


def get_user_model_service() -> UserModelService:
    """Get singleton UserModelService instance."""
    global _service
    if _service is None:
        _service = UserModelService()
    return _service


def reset_user_model_service() -> None:
    """Reset singleton service."""
    global _service
    if _service:
        _service.clear_cache()
    _service = None


# Re-export models for backward compatibility
__all__ = [
    "UserPreferences",
    "CognitiveProfile",
    "UserModel",
    "UserModelService",
    "get_user_model_service",
    "reset_user_model_service",
    "MAX_PATTERNS",
]
