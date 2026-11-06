"""Web search tool: gated EXTERNAL access to the open web.

Design notes:

- The search itself is a *pluggable fetcher*: any callable with the
  signature ``fetcher(query: str, max_results: int) -> list[SearchResult]``
  can be injected (a test stub, a paid API client, a local index). The
  built-in default is a zero-dependency HTML endpoint fetcher over
  ``urllib``; its endpoint is configurable, so the owner is never locked
  to one search provider.
- The tool is registered with risk ``EXTERNAL`` (see ``tools.py``), so
  the deliberation gate denies it unless the owner explicitly approves
  the run. Denial happens before any network call.
- Results are stored as *episodic* memory, never as semantic facts:
  a web result is an external claim, not a verified fact. Its source is
  the result URL itself, so every stored item carries its provenance,
  and the later consolidation job can only promote it with owner
  approval (its confidence is below the auto-promote threshold).
- Only ``http``/``https`` result URLs are accepted, and duplicate URLs
  within one search are collapsed, so a fetcher cannot smuggle
  non-web schemes or flood memory with repeats.
"""
from __future__ import annotations

import html as _html
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

DEFAULT_ENDPOINT = "https://html.duckduckgo.com/html/"
DEFAULT_TIMEOUT = 10
MAX_RESULTS_LIMIT = 10
WEB_RESULT_CONFIDENCE = 0.8


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str = ""

    def as_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}


def _is_web_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class _ResultPageParser(HTMLParser):
    """Extract result links and snippets from a search-result HTML page.

    The parser only recognises anchors explicitly marked as results
    (``class="result__a"``) and snippets (``class="result__snippet"``);
    navigation and ad links are ignored. Result hrefs that wrap the real
    URL in a redirect parameter (``uddg``) are unwrapped.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._capture: str | None = None  # "title" | "snippet" | None
        self._buffer: list[str] = []
        self._current: dict = {}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "a":
            return
        attrs_d = dict(attrs)
        classes = attrs_d.get("class", "").split()
        if "result__a" in classes:
            self._flush_current()
            self._current = {"url": self._unwrap(attrs_d.get("href", ""))}
            self._capture = "title"
            self._buffer = []
        elif "result__snippet" in classes:
            self._capture = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._capture is None:
            return
        text = _html.unescape(" ".join("".join(self._buffer).split()))
        if self._capture == "title":
            self._current["title"] = text
        else:
            self._current["snippet"] = text
        self._capture = None
        self._buffer = []

    def _flush_current(self) -> None:
        if self._current.get("url") and self._current.get("title"):
            self.results.append(SearchResult(
                url=self._current["url"],
                title=self._current["title"],
                snippet=self._current.get("snippet", "")))
        self._current = {}

    def close(self) -> None:
        super().close()
        self._flush_current()

    @staticmethod
    def _unwrap(href: str) -> str:
        parsed = urllib.parse.urlparse(href)
        query = urllib.parse.parse_qs(parsed.query)
        wrapped = query.get("uddg")
        if wrapped:
            return wrapped[0]
        if href.startswith("//"):
            return "https:" + href
        return href


class HtmlEndpointFetcher:
    """Fetch search results from an HTML search endpoint (stdlib only)."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT,
                 timeout: int = DEFAULT_TIMEOUT) -> None:
        if not _is_web_url(endpoint):
            raise ValueError(f"search endpoint must be an http(s) URL: {endpoint!r}")
        self.endpoint = endpoint
        self.timeout = timeout

    def __call__(self, query: str, max_results: int) -> list[SearchResult]:
        data = urllib.parse.urlencode({"q": query}).encode()
        request = urllib.request.Request(
            self.endpoint, data=data,
            headers={"User-Agent": "Steelbound/0.1 (+local-first AI core)"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
        parser = _ResultPageParser()
        parser.feed(page)
        parser.close()
        return parser.results[:max_results]

    @staticmethod
    def parse_html(page: str, max_results: int = MAX_RESULTS_LIMIT) -> list[SearchResult]:
        """Parse a result page without touching the network (for tests)."""
        parser = _ResultPageParser()
        parser.feed(page)
        parser.close()
        return parser.results[:max_results]


class WebSearch:
    """Search the web through a pluggable fetcher, with memory storage."""

    def __init__(self, fetcher=None) -> None:
        self.fetcher = fetcher or HtmlEndpointFetcher()

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        q = (query or "").strip()
        if not q:
            raise ValueError("search query must not be empty")
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            raise ValueError("max_results must be an integer")
        n = max(1, min(MAX_RESULTS_LIMIT, n))
        raw = self.fetcher(q, n)
        seen: set[str] = set()
        results: list[SearchResult] = []
        for item in raw or []:
            url = (item.url or "").strip()
            title = (item.title or "").strip()
            if not url or not title or not _is_web_url(url):
                continue
            if url in seen:
                continue
            seen.add(url)
            results.append(SearchResult(url=url, title=title,
                                        snippet=(item.snippet or "").strip()))
        return results[:n]

    def search_as_dicts(self, query: str, max_results: int = 5) -> list[dict]:
        return [r.as_dict() for r in self.search(query, max_results)]

    def store_results(self, memory, ledger, query: str,
                      results: list[SearchResult]) -> list[int]:
        """Store results as episodic memory, source URL as provenance."""
        stored: list[int] = []
        for result in results:
            content = result.title
            if result.snippet:
                content = f"{result.title} — {result.snippet}"
            item_id = memory.episode(
                content, source=result.url,
                confidence=WEB_RESULT_CONFIDENCE,
                tags=["web_search", f"query:{query}"])
            stored.append(item_id)
            if ledger:
                ledger.append("websearch", "WEB_SEARCH_STORED", {
                    "item_id": item_id, "url": result.url, "query": query})
        return stored
