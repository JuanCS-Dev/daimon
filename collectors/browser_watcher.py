#!/usr/bin/env python3
"""
DAIMON Browser Watcher - Web Activity Tracking
===============================================

Receives web activity data from browser extension.
Privacy-preserving: captures DOMAIN only, not full URLs or content.

Browser extension sends:
- Domain of active tab
- Time spent on domain
- Tab changes

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import BaseWatcher, Heartbeat
from .registry import register_collector
from memory.activity_store import get_activity_store

logger = logging.getLogger("daimon.browser")

# Configuration
AGGREGATION_INTERVAL = 60.0  # Aggregate domain time every 60s
MAX_DOMAINS_TRACKED = 50  # Limit domains in memory


@dataclass
class DomainActivity:
    """
    Activity on a single domain.

    Attributes:
        domain: The domain name (e.g., "github.com")
        time_spent: Total seconds spent on this domain
        visit_count: Number of times visited
        last_visit: When last visited
    """
    domain: str
    time_spent: float = 0.0
    visit_count: int = 0
    last_visit: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "domain": self.domain,
            "time_spent": round(self.time_spent, 1),
            "visit_count": self.visit_count,
            "last_visit": self.last_visit.isoformat() if self.last_visit else None,
        }


@dataclass
class BrowserSession:
    """
    Current browser session state.

    Tracks active domain and pending activity data from extension.
    """
    active_domain: Optional[str] = None
    domain_start: Optional[datetime] = None
    activities: Dict[str, DomainActivity] = field(default_factory=dict)

    def switch_domain(self, new_domain: Optional[str]) -> Optional[float]:
        """
        Switch to a new domain, returning time spent on previous.

        Args:
            new_domain: New active domain (or None if browser unfocused).

        Returns:
            Seconds spent on previous domain, or None if no previous.
        """
        now = datetime.now()
        time_spent = None

        # Close out previous domain
        if self.active_domain and self.domain_start:
            time_spent = (now - self.domain_start).total_seconds()

            if self.active_domain not in self.activities:
                self.activities[self.active_domain] = DomainActivity(
                    domain=self.active_domain
                )

            activity = self.activities[self.active_domain]
            activity.time_spent += time_spent
            activity.last_visit = now

        # Start new domain
        if new_domain:
            self.active_domain = new_domain
            self.domain_start = now

            if new_domain not in self.activities:
                self.activities[new_domain] = DomainActivity(domain=new_domain)
            self.activities[new_domain].visit_count += 1
        else:
            self.active_domain = None
            self.domain_start = None

        return time_spent

    def get_and_reset(self) -> List[DomainActivity]:
        """
        Get all activities and reset counters.

        Returns:
            List of domain activities since last reset.
        """
        # Update current domain time before reset
        if self.active_domain and self.domain_start:
            now = datetime.now()
            time_spent = (now - self.domain_start).total_seconds()

            if self.active_domain in self.activities:
                self.activities[self.active_domain].time_spent += time_spent

            # Restart timer for current domain
            self.domain_start = now

        result = list(self.activities.values())
        self.activities.clear()

        # Re-add current domain with zero time (timer restarted)
        if self.active_domain:
            self.activities[self.active_domain] = DomainActivity(
                domain=self.active_domain,
                visit_count=1,
                last_visit=datetime.now(),
            )

        return result


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL, removing subdomains like 'www'.

    Args:
        url: Full URL.

    Returns:
        Domain or None if invalid.
    """
    if not url:
        return None

    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if not domain:
            return None

        # Remove port if present
        domain = domain.split(":")[0]

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Basic validation - domain pattern
        domain_pattern = r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?"
        domain_pattern += r"(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$"
        if not re.match(domain_pattern, domain):
            return None

        return domain

    except (ValueError, AttributeError):
        return None


@register_collector
class BrowserWatcher(BaseWatcher):
    """
    Watcher for browser activity via extension.

    Receives domain/time data from browser extension.
    Aggregates and stores domain-level statistics only.

    Attributes:
        name: "browser_watcher"
        version: "1.0.0"
    """

    name = "browser_watcher"
    version = "1.0.0"

    def __init__(self, batch_interval: float = 60.0):
        """
        Initialize browser watcher.

        Args:
            batch_interval: Seconds between aggregation.
        """
        super().__init__(batch_interval=batch_interval)
        self._session = BrowserSession()
        self._pending_events: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def receive_event(self, event: Dict[str, Any]) -> bool:
        """
        Receive event from browser extension.

        Called by HTTP endpoint when extension sends data.

        Args:
            event: Event data with 'type' and payload.

        Returns:
            True if event was valid and processed.
        """
        async with self._lock:
            event_type = event.get("type")

            if event_type == "tab_change":
                url = event.get("url", "")
                domain = extract_domain(url)
                self._session.switch_domain(domain)
                logger.debug("Tab change: %s", domain or "(unfocused)")
                return True

            if event_type == "tab_close":
                self._session.switch_domain(None)
                logger.debug("Tab closed")
                return True

            if event_type == "heartbeat":
                # Extension sends periodic heartbeats
                url = event.get("url", "")
                domain = extract_domain(url)
                if domain and domain != self._session.active_domain:
                    self._session.switch_domain(domain)
                return True

            logger.warning("Unknown browser event type: %s", event_type)
            return False

    async def collect(self) -> Optional[Heartbeat]:
        """
        Collect aggregated browser activity.

        Returns:
            Heartbeat with domain statistics, or None if no activity.
        """
        async with self._lock:
            activities = self._session.get_and_reset()

        if not activities:
            return None

        # Sort by time spent
        activities.sort(key=lambda a: a.time_spent, reverse=True)

        # Limit domains
        if len(activities) > MAX_DOMAINS_TRACKED:
            activities = activities[:MAX_DOMAINS_TRACKED]

        return Heartbeat(
            timestamp=datetime.now(),
            watcher_type=self.name,
            data={
                "domains": [a.to_dict() for a in activities],
                "total_time": sum(a.time_spent for a in activities),
                "unique_domains": len(activities),
            },
        )

    def _should_merge(self, heartbeat: Heartbeat) -> bool:
        """Never merge browser heartbeats - each aggregation is unique."""
        return False

    def get_collection_interval(self) -> float:
        """Return aggregation interval."""
        return self.batch_interval

    def get_config(self) -> Dict[str, Any]:
        """Return watcher configuration."""
        return {
            "batch_interval": self.batch_interval,
            "active_domain": self._session.active_domain,
            "tracked_domains": len(self._session.activities),
        }

    def get_current_domain(self) -> Optional[str]:
        """Get currently active domain."""
        return self._session.active_domain

    async def flush(self) -> list:
        """Flush heartbeats to ActivityStore."""
        flushed = await super().flush()
        if not flushed:
            return flushed

        # Store in ActivityStore
        try:
            store = get_activity_store()
            for hb in flushed:
                store.add(watcher_type="browser", timestamp=hb.timestamp, data=hb.data)
        except Exception as e:
            logger.warning("Failed to store browser data: %s", e)

        return flushed
