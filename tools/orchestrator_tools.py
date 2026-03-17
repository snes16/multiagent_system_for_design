from __future__ import annotations
import logging
import os
import time
from crewai import Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from agents.builders import (
    BRIEF_FILE,
    CRITIQUE_FILE,
    REFS_FILE,
    critic_agent,
    critique_task,
    generation_task,
    generator_agent,
    research_agent,
    research_task,
    style_analyst_agent,
    style_task,
)
from tools.workspace import FileReaderTool, workspace_path

logger = logging.getLogger(__name__)


def _run_crew_with_retry(crew, max_retries: int = 3, wait: int = 5) -> str:
    """Run a Crew with simple retry on exception."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as exc:
            last_exc = exc
            logger.warning("Crew attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(wait * attempt)  # exponential-ish back-off
    logger.error("All %d attempts failed. Last error: %s", max_retries, last_exc)
    raise RuntimeError(f"Crew failed after {max_retries} attempts: {last_exc}") from last_exc


# ── run_research ──────────────────────────────────────────────────────────────

class RunResearchInput(BaseModel):
    prompt: str = Field(description="Дизайн-задача для исследования референсов")


class RunResearchTool(BaseTool):
    name: str = "run_research"
    description: str = (
        "Запускает Research Agent: парсит artlebedev.ru, ищет референсы на Awwwards и Dribbble. "
        f"Результат сохраняется в {REFS_FILE}. "
        "Используй первым — до анализа стиля и генерации."
    )
    args_schema: type[BaseModel] = RunResearchInput

    def _run(self, prompt: str) -> str:
        logger.info("Running %s", self.name)
        try:
            agent = research_agent()
            task = research_task(agent, prompt)
            _run_crew_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False),
                max_retries=3, wait=5,
            )
            return f"Research complete → {REFS_FILE}"
        except RuntimeError as e:
            logger.error("run_research error: %s", e)
            return f"[run_research failed] {e}"


# ── run_style_analysis ────────────────────────────────────────────────────────

class RunStyleInput(BaseModel):
    prompt: str = Field(description="Дизайн-задача для формирования брифа")


class RunStyleAnalysisTool(BaseTool):
    name: str = "run_style_analysis"
    description: str = (
        f"Запускает Style Analyst: читает {REFS_FILE}, формирует DesignBrief. "
        f"Результат сохраняется в {BRIEF_FILE}. "
        f"Используй после run_research — файл {REFS_FILE} должен существовать."
    )
    args_schema: type[BaseModel] = RunStyleInput

    def _run(self, prompt: str) -> str:
        logger.info("Running %s", self.name)
        try:
            if not workspace_path("references.md").exists():
                return f"Ошибка: {REFS_FILE} не найден. Сначала запусти run_research."
            agent = style_analyst_agent()
            task = style_task(agent, prompt)
            _run_crew_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False),
                max_retries=3, wait=5,
            )
            return f"Style analysis завершён → {BRIEF_FILE}"
        except RuntimeError as e:
            logger.error("run_style_analysis error: %s", e)
            return f"[run_style_analysis failed] {e}"


# ── run_generation ────────────────────────────────────────────────────────────

class RunGenerationInput(BaseModel):
    prompt: str = Field(description="Дизайн-задача")
    output_format: str = Field(
        default="html",
        description="Формат вывода: html, svg, moodboard, brandbook"
    )
    revision_notes: str = Field(
        default="",
        description="Правки от критика для улучшения (опционально)"
    )


class RunGenerationTool(BaseTool):
    name: str = "run_generation"
    description: str = (
        f"Запускает Design Generator: читает {BRIEF_FILE}, создаёт дизайн в output/. "
        f"Используй после run_style_analysis — файл {BRIEF_FILE} должен существовать. "
        "При повторном запуске передай revision_notes с замечаниями от критика."
    )
    args_schema: type[BaseModel] = RunGenerationInput

    def _run(self, prompt: str, output_format: str = "html",
             revision_notes: str = "") -> str:
        logger.info("Running %s", self.name)
        try:
            if not workspace_path("brief.json").exists():
                return f"Ошибка: {BRIEF_FILE} не найден. Сначала запусти run_style_analysis."
            agent = generator_agent()
            task = generation_task(agent, prompt, output_format, revision_notes=revision_notes)
            _run_crew_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False),
                max_retries=3, wait=5,
            )
            return f"Генерация завершена → output/{_filename_for(output_format)}"
        except RuntimeError as e:
            logger.error("run_generation error: %s", e)
            return f"[run_generation failed] {e}"


# ── run_critique ──────────────────────────────────────────────────────────────

class RunCritiqueInput(BaseModel):
    prompt: str = Field(description="Оригинальная дизайн-задача")
    output_path: str = Field(description="Путь к сгенерированному файлу для оценки")


class RunCritiqueTool(BaseTool):
    name: str = "run_critique"
    description: str = (
        "Запускает Critic Agent: читает brief.json и сгенерированный файл, оценивает по 5 критериям. "
        f"Результат (JSON с оценками) сохраняется в {CRITIQUE_FILE}. "
        "Используй после run_generation. Возвращает overall_score и список улучшений."
    )
    args_schema: type[BaseModel] = RunCritiqueInput

    def _run(self, prompt: str, output_path: str) -> str:
        logger.info("Running %s", self.name)
        try:
            if not workspace_path("brief.json").exists():
                return f"Ошибка: {BRIEF_FILE} не найден."
            agent = critic_agent()
            task = critique_task(agent, prompt, output_path)
            _run_crew_with_retry(
                Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False),
                max_retries=3, wait=5,
            )
            reader = FileReaderTool()
            raw = reader._run(filepath=str(workspace_path("critique.json")))
            return f"Critique завершён → {CRITIQUE_FILE}\n\nРезультат:\n{raw}"
        except RuntimeError as e:
            logger.error("run_critique error: %s", e)
            return f"[run_critique failed] {e}"


# ── read_workspace_file ───────────────────────────────────────────────────────

class ReadWorkspaceInput(BaseModel):
    filename: str = Field(
        description="Имя файла в workspace: references.md, brief.json, critique.json"
    )


class ReadWorkspaceFileTool(BaseTool):
    name: str = "read_workspace_file"
    description: str = (
        "Читает промежуточный файл из output/.workspace/. "
        "Используй чтобы проверить результат агента перед следующим шагом. "
        "Доступные файлы: references.md, brief.json, critique.json"
    )
    args_schema: type[BaseModel] = ReadWorkspaceInput

    def _run(self, filename: str) -> str:
        logger.info("Running %s", self.name)
        path = workspace_path(filename)
        if not path.exists():
            return f"File not found: {path}"
        content = path.read_text(encoding="utf-8")
        preview = content[:2000]
        truncated = len(content) > 2000
        return preview + ("\n\n[...truncated to save context]" if truncated else "")


# ── helpers ───────────────────────────────────────────────────────────────────

def _filename_for(output_format: str) -> str:
    return {
        "html": "landing.html",
        "svg": "layout.svg",
        "moodboard": "moodboard.html",
        "brandbook": "brandbook.html",
    }.get(output_format, "landing.html")
