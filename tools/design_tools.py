from __future__ import annotations
import logging
import os
import json
logger = logging.getLogger(__name__)
from typing import Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── Firecrawl scrape ───────────────────────────────────────────────────────────

class ScrapeInput(BaseModel):
    url: str = Field(description="URL of the page to scrape")
    extract_prompt: Optional[str] = Field(
        default=None,
        description="Prompt for LLM-based structured data extraction"
    )


class FirecrawlScrapeTool(BaseTool):
    name: str = "firecrawl_scrape"
    description: str = (
        "Scrapes a single page and returns Markdown content. "
        "Use for artlebedev.ru, awwwards.com, dribbble.com and other design sites. "
        "Pass extract_prompt to extract structured data via LLM."
    )
    args_schema: type[BaseModel] = ScrapeInput

    def _run(self, url: str, extract_prompt: Optional[str] = None) -> str:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

            if extract_prompt:
                result = app.scrape_url(url, params={
                    "formats": ["markdown", "extract"],
                    "extract": {"prompt": extract_prompt},
                })
                extracted = result.get("extract", {})
                markdown = result.get("markdown", "")[:2000]
                return (
                    f"EXTRACTED:\n{json.dumps(extracted, ensure_ascii=False, indent=2)}"
                    f"\n\nMARKDOWN (preview):\n{markdown}"
                )
            else:
                result = app.scrape_url(url, params={"formats": ["markdown"]})
                return result.get("markdown", "")[:4000]

        except Exception as e:
            logger.error("firecrawl_scrape failed url=%s: %s", url, e)
            return f"[firecrawl_scrape error] {url}: {e}"


# ── Firecrawl crawl (multiple pages) ──────────────────────────────────────────

class CrawlInput(BaseModel):
    url: str = Field(description="Starting URL for crawling")
    limit: int = Field(default=6, description="Max number of pages (no more than 10)")


class FirecrawlCrawlTool(BaseTool):
    name: str = "firecrawl_crawl"
    description: str = (
        "Crawls a site and returns content from multiple pages. "
        "Ideal for artlebedev.ru/everything/ — to collect several projects at once."
    )
    args_schema: type[BaseModel] = CrawlInput

    def _run(self, url: str, limit: int = 6) -> str:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

            result = app.crawl_url(url, params={
                "limit": min(limit, 10),
                "formats": ["markdown"],
            })
            pages = result.get("data", [])

            summaries = []
            for page in pages:
                meta = page.get("metadata", {})
                content = page.get("markdown", "")[:600]
                title = meta.get("title", "No title")
                src = meta.get("sourceURL", "")
                summaries.append(f"### {title}\nURL: {src}\n{content}")

            return "\n\n---\n\n".join(summaries) if summaries else "No pages found."

        except Exception as e:
            logger.error("firecrawl_crawl failed url=%s: %s", url, e)
            return f"[firecrawl_crawl error] {url}: {e}"


# ── Web search (DuckDuckGo, без API) ──────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Search query in English")
    max_results: int = Field(default=5)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Searches the web by query. "
        "Use to find references on Awwwards, Dribbble, Behance. "
        "Queries should be formulated in English."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        f"**{r['title']}**\n{r['href']}\n{r['body'][:250]}"
                    )
            return "\n\n".join(results) if results else "No results found."
        except Exception as e:
            logger.error("web_search failed query=%r: %s", query, e)
            return f"[web_search error] {e}"


# ── File writer ────────────────────────────────────────────────────────────────

class WriteInput(BaseModel):
    filename: str = Field(description="Filename with extension, e.g. landing.html")
    content: str = Field(description="Full file contents")


class FileWriterTool(BaseTool):
    name: str = "file_writer"
    description: str = (
        "Saves a finished HTML, SVG or Markdown file to the output/ directory. "
        "Always call at the end of generation."
    )
    args_schema: type[BaseModel] = WriteInput

    def _run(self, filename: str, content: str) -> str:
        out_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"saved:{path}"