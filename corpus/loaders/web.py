"""
DAIMON Web Loader - Fetch and extract text from URLs.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlparse

from .base import BaseLoader, Document

logger = logging.getLogger("daimon.corpus.web")


class WebLoader(BaseLoader):
    """
    Loader for web URLs.

    Fetches HTML content and extracts readable text.
    Requires httpx and beautifulsoup4.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "DAIMON/1.0 (Personal Exocortex)",
    ):
        """
        Initialize web loader.

        Args:
            timeout: Request timeout in seconds.
            user_agent: User agent string for requests.
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self._deps_available: Optional[bool] = None

    def _check_deps(self) -> bool:
        """Check if required dependencies are available."""
        if self._deps_available is None:
            try:
                import httpx  # noqa: F401
                from bs4 import BeautifulSoup  # noqa: F401
                self._deps_available = True
            except ImportError as e:
                self._deps_available = False
                logger.warning(
                    "Web loader dependencies not available: %s. "
                    "Install with: pip install httpx beautifulsoup4",
                    e
                )
        return self._deps_available

    def load(self, path: str) -> Document:
        """
        Load and extract text from URL.

        Args:
            path: URL to fetch.

        Returns:
            Document with extracted text.

        Raises:
            ValueError: If dependencies not available or URL invalid.
        """
        if not self._check_deps():
            raise ValueError(
                "Web loading requires httpx and beautifulsoup4. "
                "Install with: pip install httpx beautifulsoup4"
            )

        # Validate URL
        parsed = urlparse(path)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        import httpx  # pylint: disable=import-outside-toplevel
        from bs4 import BeautifulSoup  # pylint: disable=import-outside-toplevel

        # Fetch content
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    path,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True,
                )
                response.raise_for_status()
                html = response.text
        except httpx.HTTPError as e:
            raise ValueError(f"Failed to fetch URL: {e}") from e

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Try to find main content
        main_content = self._find_main_content(soup)
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fallback to body text
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

        text = self._clean_text(text)

        # Extract metadata
        metadata = {
            "format": "web",
            "url": path,
            "domain": parsed.netloc,
        }

        # Extract meta tags
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]

        meta_author = soup.find("meta", attrs={"name": "author"})
        if meta_author and meta_author.get("content"):
            metadata["author"] = meta_author["content"]

        # og:title fallback
        if not title:
            og_title = soup.find("meta", attrs={"property": "og:title"})
            if og_title and og_title.get("content"):
                title = og_title["content"]

        if not title:
            title = parsed.netloc

        return Document(
            text=text,
            source=path,
            title=title,
            metadata=metadata,
        )

    def _find_main_content(self, soup) -> Optional["BeautifulSoup"]:
        """
        Find main content area of page.

        Tries common content containers in order.
        """
        # Common main content selectors
        selectors = [
            ("article", {}),
            ("main", {}),
            ("div", {"class": re.compile(r"(content|article|post|entry)")}),
            ("div", {"id": re.compile(r"(content|article|post|main)")}),
            ("div", {"role": "main"}),
        ]

        for tag, attrs in selectors:
            element = soup.find(tag, attrs)
            if element:
                # Verify it has meaningful content
                text = element.get_text(strip=True)
                if len(text) > 200:  # Minimum content threshold
                    return element

        return None
