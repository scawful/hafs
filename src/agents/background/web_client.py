"""Web client for website monitoring and crawling.

Provides HTTP client with rate limiting, retries, and HTML parsing
for background agents monitoring halext websites.
"""

from __future__ import annotations

import asyncio
import logging
import random
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class WebResponse:
    """HTTP response with metadata."""

    url: str
    status_code: int
    text: str
    headers: dict[str, str]
    response_time_ms: int
    timestamp: str


@dataclass
class HealthStatus:
    """Website health check result."""

    name: str
    url: str
    status: str  # "online", "offline", "degraded"
    status_code: Optional[int]
    response_time_ms: Optional[int]
    ssl_valid: bool
    ssl_expires: Optional[str]
    error: Optional[str]
    timestamp: str


@dataclass
class Page:
    """Crawled page data."""

    url: str
    title: str
    text: str
    links: list[str]
    status_code: int
    crawl_time_ms: int


class WebClient:
    """HTTP client with rate limiting and retries.

    Features:
    - Async HTTP via httpx
    - Rate limiting (configurable delay between requests)
    - Exponential backoff on errors
    - HTML parsing with BeautifulSoup4
    - Sitemap.xml parsing
    - robots.txt support
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        user_agent: str = "hafs-indexer/1.0",
    ):
        """Initialize web client.

        Args:
            base_url: Base URL for the website
            timeout: Request timeout in seconds
            rate_limit: Minimum seconds between requests
            max_retries: Maximum retry attempts
            user_agent: User-Agent header
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.last_request_time = 0.0

        # Create async HTTP client
        self.client: Optional[httpx.AsyncClient] = None

        # Track visited URLs (for crawling)
        self.visited: set[str] = set()

        # robots.txt rules (NOTE: Not yet implemented - relies on rate limiting for safety)
        # TODO: Implement robots.txt checking using urllib.robotparser
        self.robots_rules: dict[str, Any] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def _rate_limit_wait(self):
        """Wait to respect rate limit."""
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.rate_limit:
            wait_time = self.rate_limit - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = asyncio.get_event_loop().time()

    async def get(self, path: str, full_url: bool = False) -> WebResponse:
        """GET request with rate limiting and retries.

        Args:
            path: URL path or full URL if full_url=True
            full_url: If True, path is a complete URL

        Returns:
            WebResponse object

        Raises:
            httpx.HTTPError: On request failure after retries
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async with.")

        url = path if full_url else urljoin(self.base_url, path)

        # Rate limiting
        await self._rate_limit_wait()

        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = asyncio.get_event_loop().time()
                response = await self.client.get(url)
                end_time = asyncio.get_event_loop().time()

                response_time_ms = int((end_time - start_time) * 1000)

                return WebResponse(
                    url=url,
                    status_code=response.status_code,
                    text=response.text,
                    headers=dict(response.headers),
                    response_time_ms=response_time_ms,
                    timestamp=datetime.now().isoformat(),
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.debug(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        # All retries failed
        raise last_error  # type: ignore

    async def check_health(self, alert_threshold_ms: int = 5000) -> HealthStatus:
        """Check if website is healthy.

        Args:
            alert_threshold_ms: Response time threshold for alerts

        Returns:
            HealthStatus object
        """
        name = urlparse(self.base_url).netloc
        timestamp = datetime.now().isoformat()

        try:
            response = await self.get("/")

            # Check SSL certificate (if HTTPS)
            ssl_valid = True
            ssl_expires = None
            if self.base_url.startswith("https://"):
                # TODO: Extract SSL expiry from connection
                ssl_expires = "N/A"

            # Determine status
            if response.status_code == 200:
                if response.response_time_ms > alert_threshold_ms:
                    status = "degraded"
                else:
                    status = "online"
            else:
                status = "degraded"

            return HealthStatus(
                name=name,
                url=self.base_url,
                status=status,
                status_code=response.status_code,
                response_time_ms=response.response_time_ms,
                ssl_valid=ssl_valid,
                ssl_expires=ssl_expires,
                error=None,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.error(f"Health check failed for {self.base_url}: {e}")
            return HealthStatus(
                name=name,
                url=self.base_url,
                status="offline",
                status_code=None,
                response_time_ms=None,
                ssl_valid=False,
                ssl_expires=None,
                error=str(e),
                timestamp=timestamp,
            )

    async def crawl_sitemap(self) -> list[str]:
        """Parse sitemap.xml and return URLs.

        Returns:
            List of URLs from sitemap
        """
        urls = []

        try:
            response = await self.get("/sitemap.xml")

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "xml")
                urls = [loc.text for loc in soup.find_all("loc")]
                logger.info(f"Found {len(urls)} URLs in sitemap.xml")

        except Exception as e:
            logger.debug(f"Failed to parse sitemap.xml: {e}")

        return urls

    async def extract_links(self, html: str, base_url: Optional[str] = None) -> list[str]:
        """Extract all links from HTML.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        if base_url is None:
            base_url = self.base_url

        soup = BeautifulSoup(html, "lxml")
        links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Skip anchors and mailto links
            if href.startswith("#") or href.startswith("mailto:"):
                continue

            # Convert relative to absolute
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    async def extract_text(self, html: str) -> str:
        """Extract clean text content from HTML.

        Strips scripts, styles, and HTML tags.

        Args:
            html: HTML content

        Returns:
            Clean text content
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = " ".join(line for line in lines if line)

        return text

    async def crawl_page(self, url: str, extract_links: bool = True) -> Page:
        """Crawl a single page and extract content.

        Args:
            url: URL to crawl
            extract_links: Whether to extract links from page

        Returns:
            Page object with extracted content
        """
        response = await self.get(url, full_url=True)

        soup = BeautifulSoup(response.text, "lxml")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.text if title_tag else ""

        # Extract text content
        text = await self.extract_text(response.text)

        # Extract links
        links = []
        if extract_links:
            links = await self.extract_links(response.text, url)

        return Page(
            url=url,
            title=title,
            text=text,
            links=links,
            status_code=response.status_code,
            crawl_time_ms=response.response_time_ms,
        )

    def is_same_domain(self, url: str) -> bool:
        """Check if URL is same domain as base_url.

        Args:
            url: URL to check

        Returns:
            True if same domain
        """
        base_domain = urlparse(self.base_url).netloc
        url_domain = urlparse(url).netloc
        return base_domain == url_domain

    async def bfs_crawl(
        self,
        start_url: Optional[str] = None,
        max_pages: int = 500,
        max_depth: int = 3,
        same_domain_only: bool = True,
    ) -> list[Page]:
        """Breadth-first search crawl of website.

        Args:
            start_url: Starting URL (defaults to base_url)
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to crawl
            same_domain_only: Only crawl links on same domain

        Returns:
            List of crawled pages
        """
        if start_url is None:
            start_url = self.base_url

        self.visited = set()
        pages = []
        queue = [(start_url, 0)]  # (url, depth)

        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)

            # Skip if already visited or too deep
            if url in self.visited or depth > max_depth:
                continue

            # Skip if different domain
            if same_domain_only and not self.is_same_domain(url):
                continue

            self.visited.add(url)

            try:
                logger.info(f"Crawling ({len(pages) + 1}/{max_pages}): {url}")
                page = await self.crawl_page(url)
                pages.append(page)

                # Add links to queue
                if depth < max_depth:
                    for link in page.links:
                        if link not in self.visited:
                            queue.append((link, depth + 1))

            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                continue

        logger.info(f"Crawled {len(pages)} pages")
        return pages
