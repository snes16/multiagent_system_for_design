from __future__ import annotations
import os
from crewai import Agent, Task, LLM

from tools.design_tools import (
    FirecrawlScrapeTool,
    FirecrawlCrawlTool,
    WebSearchTool,
    FileWriterTool,
    FileReaderTool,
    WorkspaceWriterTool,
)

REFS_FILE     = "output/.workspace/references.md"
BRIEF_FILE    = "output/.workspace/brief.json"
CRITIQUE_FILE = "output/.workspace/critique.json"


def _llm(model: str = "claude-sonnet-4-5", temperature: float = 0.7) -> LLM:
    return LLM(
        model=f"anthropic/{model}",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature,
        max_tokens=8192,
    )


def research_agent() -> Agent:
    return Agent(
        role="Design Research Specialist",
        goal=(
            "Собрать 4-6 визуальных референсов и сохранить их в workspace. "
            "Парсить artlebedev.ru/everything/, искать на Awwwards и Dribbble. "
            f"Результат сохранить в {REFS_FILE} через workspace_writer."
        ),
        backstory=(
            "Опытный арт-директор, который умеет быстро находить лучшие дизайн-решения. "
            "Хорошо знает каталог Артлебедева — /everything/ содержит все проекты. "
            "Всегда сохраняет результаты в файл — не держит данные в памяти."
        ),
        tools=[FirecrawlCrawlTool(), FirecrawlScrapeTool(), WebSearchTool(), WorkspaceWriterTool()],
        llm=_llm("claude-sonnet-4-5", temperature=0.3),
        verbose=True,
        allow_delegation=False,
        max_iter=6,
    )


def style_analyst_agent() -> Agent:
    return Agent(
        role="Senior Art Director & Style Analyst",
        goal=(
            f"Прочитать {REFS_FILE}, синтезировать DesignBrief. "
            f"Сохранить JSON бриф в {BRIEF_FILE} через workspace_writer."
        ),
        backstory=(
            "Арт-директор с опытом в Pentagram и Артлебедеве. "
            "Всегда читает данные из файлов, а не из описания задачи — "
            "это позволяет работать с большими объёмами референсов без потери качества."
        ),
        tools=[FileReaderTool(), WebSearchTool(), WorkspaceWriterTool()],
        llm=_llm("claude-opus-4-5", temperature=0.8),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def generator_agent() -> Agent:
    return Agent(
        role="Senior Frontend Designer & Developer",
        goal=(
            f"Прочитать {BRIEF_FILE}, создать production-ready дизайн. "
            "Сохранить финальный файл через file_writer в output/."
        ),
        backstory=(
            "Senior frontend, который делает визуально выдающиеся интерфейсы. "
            "Работает только с актуальными данными из файлов."
        ),
        tools=[FileReaderTool(), FileWriterTool()],
        llm=_llm("claude-opus-4-5", temperature=0.9),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def critic_agent() -> Agent:
    return Agent(
        role="Design Critic & Quality Reviewer",
        goal=(
            f"Прочитать {BRIEF_FILE}, оценить дизайн по 5 критериям. "
            f"Сохранить CritiqueResult JSON в {CRITIQUE_FILE} через workspace_writer."
        ),
        backstory=(
            "Жёсткий но справедливый критик. "
            "Сверяет результат с брифом читая оба из файлов."
        ),
        tools=[FileReaderTool(), WorkspaceWriterTool()],
        llm=_llm("claude-sonnet-4-5", temperature=0.2),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


# ── Таски — только пути к файлам, никаких данных в description ────────────────

def research_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=f"""
Исследуй дизайн-референсы для задачи: **{prompt}**

1. Краулинг Артлебедева:
   firecrawl_crawl(url="https://www.artlebedev.ru/everything/", limit=8)
   Для 2-3 проектов — firecrawl_scrape с extract_prompt:
   "Extract: project name, visual style, colors used, typography, key design patterns"

2. Поиск дополнительных референсов:
   web_search: "{prompt} website design awwwards 2024"
   web_search: "{prompt} brand identity dribbble behance"

3. Сформируй отчёт по 4-6 лучшим референсам:
   SOURCE / PROJECT / URL / KEY_PATTERNS / COLOR_NOTES / TYPOGRAPHY_NOTES
   + блок SYNTHESIS (3 предложения об общих трендах)

4. Сохрани ВЕСЬ результат:
   workspace_writer(filename="references.md", content=<полный отчёт>)
""",
        expected_output=f"Подтверждение сохранения: {REFS_FILE}",
        agent=agent,
    )


def style_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=f"""
Сформируй DesignBrief для задачи: **{prompt}**

1. Прочитай референсы:
   file_reader(filepath="{REFS_FILE}")

2. Синтезируй бриф:
   - Шрифты из Google Fonts с характером (не Inter, не Roboto):
     Заголовки: Cormorant Garamond / Syne / DM Serif Display / Fraunces / Playfair Display
     Текст: DM Sans / Plus Jakarta Sans / Outfit / Manrope / Epilogue
   - 5 конкретных HEX цветов
   - style_direction: минимум 3 предложения

3. Сохрани JSON (без markdown обёртки):
   workspace_writer(filename="brief.json", content=<JSON>)

JSON формат:
{{
  "color_primary": "#...", "color_secondary": "#...", "color_accent": "#...",
  "color_background": "#...", "color_text": "#...",
  "font_heading": "...", "font_body": "...",
  "style_direction": "...", "mood": "...",
  "layout_pattern": "...", "visual_metaphor": "...",
  "key_elements": ["...", "...", "..."],
  "target_audience": "...", "brand_personality": "..."
}}
""",
        expected_output=f"Подтверждение сохранения: {BRIEF_FILE}",
        agent=agent,
    )


_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "html": (
        "Создай полноценный одностраничный HTML лэндинг.\n"
        "- <!DOCTYPE html>, Google Fonts через @import\n"
        "- Минимум 5 секций: hero, about, services, showcase, contacts\n"
        "- CSS переменные для цветов и шрифтов из брифа\n"
        "- Hover-анимации, адаптивность (media queries)\n"
        "- Реальный контент, не Lorem ipsum; декор через CSS/inline SVG\n"
        "Сохрани: file_writer(filename=\"landing.html\", content=<полный HTML>)"
    ),
    "svg": (
        "Создай SVG макет 1440x900px.\n"
        "- viewBox=\"0 0 1440 900\", шрифты через <defs><style>@import\n"
        "- Цвета из брифа, структура: header, hero, контент, footer\n"
        "Сохрани: file_writer(filename=\"layout.svg\", content=<полный SVG>)"
    ),
    "moodboard": (
        "Создай HTML мудборд.\n"
        "- CSS Grid сетка, цветовые чипы с HEX, типографические образцы\n"
        "- Карточки референсов, блок визуальных ключей\n"
        "Сохрани: file_writer(filename=\"moodboard.html\", content=<полный HTML>)"
    ),
    "brandbook": (
        "Создай HTML брендбук.\n"
        "Разделы: миссия, логотип (SVG inline), цветовая система,\n"
        "типографика, визуальный язык, примеры применения, do/don't.\n"
        "Сохрани: file_writer(filename=\"brandbook.html\", content=<полный HTML>)"
    ),
}


def generation_task(agent: Agent, prompt: str, output_format: str,
                    revision_notes: str = "") -> Task:
    instructions = _FORMAT_INSTRUCTIONS.get(output_format, _FORMAT_INSTRUCTIONS["html"])
    revision_block = f"\n[ПРАВКИ ОТ КРИТИКА]\n{revision_notes}\n" if revision_notes else ""

    return Task(
        description=f"""
Создай дизайн для задачи: **{prompt}**
Формат: **{output_format}**
{revision_block}
1. Прочитай бриф:
   file_reader(filepath="{BRIEF_FILE}")

2. {instructions}

Строго следуй цветам и шрифтам из брифа.
""",
        expected_output="Подтверждение сохранения финального файла в output/.",
        agent=agent,
    )


def critique_task(agent: Agent, prompt: str, output_path: str) -> Task:
    return Task(
        description=f"""
Оцени дизайн для задачи: **{prompt}**
Файл для оценки: {output_path}

1. Прочитай бриф для сверки:
   file_reader(filepath="{BRIEF_FILE}")

2. Оцени дизайн (каждый критерий 0.0–10.0):
   - visual_hierarchy_score, typography_score, color_harmony_score
   - layout_score, brief_alignment_score
   overall_score = среднее; should_iterate=true если < 7.0

3. Сохрани результат:
   workspace_writer(filename="critique.json", content=<JSON без markdown>)

JSON формат:
{{
  "overall_score": 0.0,
  "visual_hierarchy_score": 0.0, "typography_score": 0.0,
  "color_harmony_score": 0.0, "layout_score": 0.0, "brief_alignment_score": 0.0,
  "strengths": ["...", "..."], "improvements": ["...", "..."],
  "verdict": "...", "should_iterate": false, "revised_brief_notes": "..."
}}
""",
        expected_output=f"Подтверждение сохранения: {CRITIQUE_FILE}",
        agent=agent,
    )