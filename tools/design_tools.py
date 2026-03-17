from __future__ import annotations
import os
import json
from typing import Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── Firecrawl scrape ───────────────────────────────────────────────────────────

class ScrapeInput(BaseModel):
    url: str = Field(description="URL страницы для скрапинга")
    extract_prompt: Optional[str] = Field(
        default=None,
        description="Промпт для LLM-извлечения структурированных данных"
    )


class FirecrawlScrapeTool(BaseTool):
    name: str = "firecrawl_scrape"
    description: str = (
        "Скрапит одну страницу и возвращает Markdown контент. "
        "Используй для artlebedev.ru, awwwards.com, dribbble.com и других дизайн-сайтов. "
        "Передай extract_prompt чтобы достать структурированные данные через LLM."
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
            return f"[firecrawl_scrape error] {url}: {e}"


# ── Firecrawl crawl (несколько страниц) ───────────────────────────────────────

class CrawlInput(BaseModel):
    url: str = Field(description="Стартовый URL для краулинга")
    limit: int = Field(default=6, description="Макс. кол-во страниц (не более 10)")


class FirecrawlCrawlTool(BaseTool):
    name: str = "firecrawl_crawl"
    description: str = (
        "Краулит сайт и возвращает контент нескольких страниц. "
        "Идеально для artlebedev.ru/everything/ — собрать сразу несколько проектов."
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

            return "\n\n---\n\n".join(summaries) if summaries else "Страницы не найдены."

        except Exception as e:
            return f"[firecrawl_crawl error] {url}: {e}"


# ── Web search (DuckDuckGo, без API) ──────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Поисковый запрос на английском")
    max_results: int = Field(default=5)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Ищет в вебе по запросу. "
        "Используй для поиска референсов на Awwwards, Dribbble, Behance. "
        "Запросы лучше формулировать на английском."
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
            return "\n\n".join(results) if results else "Ничего не найдено."
        except Exception as e:
            return f"[web_search error] {e}"


# ── File writer ────────────────────────────────────────────────────────────────

class WriteInput(BaseModel):
    filename: str = Field(description="Имя файла с расширением, напр. landing.html")
    content: str = Field(description="Полное содержимое файла")


class FileWriterTool(BaseTool):
    name: str = "file_writer"
    description: str = (
        "Сохраняет готовый HTML, SVG или Markdown файл в папку output/. "
        "Всегда вызывай в конце генерации."
    )
    args_schema: type[BaseModel] = WriteInput

    def _run(self, filename: str, content: str) -> str:
        out_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"saved:{path}"