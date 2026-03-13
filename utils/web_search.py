"""
NeoStats AI Chatbot — Web Search Integration
Supports Serper.dev, Tavily, and Brave Search APIs with fallback chain.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ─── Result Model ──────────────────────────────────────────────────────────────

class SearchResult:
    """Represents a single search result."""

    def __init__(self, title: str, url: str, snippet: str, source: str = "web"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }

    def __repr__(self):
        return f"SearchResult(title={self.title!r}, url={self.url!r})"


# ─── Serper.dev ────────────────────────────────────────────────────────────────

def search_serper(query: str, api_key: str, max_results: int = 5,
                  timeout: int = 10) -> list[SearchResult]:
    """
    Search using Serper.dev (Google Search API).

    Args:
        query: Search query string
        api_key: Serper API key
        max_results: Max results to return
        timeout: Request timeout in seconds

    Returns:
        List of SearchResult objects
    """
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": max_results}

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="serper",
            ))

        # Include answer box if present
        if "answerBox" in data:
            ab = data["answerBox"]
            results.insert(0, SearchResult(
                title=ab.get("title", "Answer"),
                url=ab.get("link", ""),
                snippet=ab.get("answer", ab.get("snippet", "")),
                source="serper_answer_box",
            ))

        return results
    except requests.exceptions.Timeout:
        logger.warning("Serper search timed out")
        return []
    except Exception as e:
        logger.error(f"Serper search error: {e}")
        return []


# ─── Tavily ────────────────────────────────────────────────────────────────────

def search_tavily(query: str, api_key: str, max_results: int = 5,
                  timeout: int = 10) -> list[SearchResult]:
    """
    Search using Tavily AI Search API.

    Args:
        query: Search query string
        api_key: Tavily API key
        max_results: Max results to return
        timeout: Request timeout in seconds

    Returns:
        List of SearchResult objects
    """
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        }

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        # Include direct answer if present
        if data.get("answer"):
            results.append(SearchResult(
                title="Direct Answer",
                url="",
                snippet=data["answer"],
                source="tavily_answer",
            ))

        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:300],
                source="tavily",
            ))

        return results
    except requests.exceptions.Timeout:
        logger.warning("Tavily search timed out")
        return []
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


# ─── Brave Search ──────────────────────────────────────────────────────────────

def search_brave(query: str, api_key: str, max_results: int = 5,
                 timeout: int = 10) -> list[SearchResult]:
    """
    Search using Brave Search API.

    Args:
        query: Search query string
        api_key: Brave API key
        max_results: Max results to return
        timeout: Request timeout in seconds

    Returns:
        List of SearchResult objects
    """
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": max_results}

        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source="brave",
            ))
        return results
    except requests.exceptions.Timeout:
        logger.warning("Brave search timed out")
        return []
    except Exception as e:
        logger.error(f"Brave search error: {e}")
        return []


# ─── Orchestrator ──────────────────────────────────────────────────────────────

def web_search(
    query: str,
    serper_key: str = "",
    tavily_key: str = "",
    brave_key: str = "",
    max_results: int = 5,
    timeout: int = 10,
) -> list[SearchResult]:
    """
    Try available search providers in order: Serper → Tavily → Brave.

    Returns results from the first provider that succeeds.
    """
    if serper_key:
        results = search_serper(query, serper_key, max_results, timeout)
        if results:
            return results

    if tavily_key:
        results = search_tavily(query, tavily_key, max_results, timeout)
        if results:
            return results

    if brave_key:
        results = search_brave(query, brave_key, max_results, timeout)
        if results:
            return results

    logger.warning("All web search providers failed or unconfigured.")
    return []


def format_search_results(results: list[SearchResult]) -> str:
    """
    Format search results into a context string for LLM injection.

    Args:
        results: List of SearchResult objects

    Returns:
        Formatted string for use in system/user prompt
    """
    if not results:
        return ""

    parts = ["🌐 **Live Web Search Results:**\n"]
    for i, r in enumerate(results, 1):
        entry = f"\n[{i}] **{r.title}**"
        if r.url:
            entry += f"\n🔗 {r.url}"
        entry += f"\n{r.snippet}\n"
        parts.append(entry)
    parts.append("\n---\n")
    return "".join(parts)


def should_search_web(query: str, llm_response: str = "") -> bool:
    """
    Heuristic to decide whether web search is needed.
    Triggers on: recent events, statistics, prices, news, "latest", "current", etc.
    """
    trigger_keywords = [
        "latest", "recent", "current", "today", "now", "2024", "2025", "2026",
        "news", "update", "price", "statistics", "study", "research",
        "when did", "what happened", "who is", "breaking",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in trigger_keywords)
