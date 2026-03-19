from __future__ import annotations
from pathlib import Path
from crewai import Agent, Task, LLM
from config.settings import settings


def _load_prompt(name: str) -> str:
    return (Path(__file__).parent.parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")

from tools.orchestrator_tools import (
    RunResearchTool,
    RunStyleAnalysisTool,
    RunStyleRevisionTool,
    RunGenerationTool,
    RunCritiqueTool,
    ReadWorkspaceFileTool,
    PromoteOutputTool,
)

def _llm() -> LLM:
    return LLM(
        model=f"anthropic/{settings.orchestrator_model}",
        api_key=settings.anthropic_api_key,
        temperature=settings.orchestrator_temperature,
        max_tokens=4096,
    )


def orchestrator_agent() -> Agent:
    return Agent(
        role="Design Project Orchestrator",
        goal=(
            "Координировать работу дизайн-агентов для создания качественного результата. "
            "Самостоятельно решать: какие агенты запустить, в каком порядке, "
            "нужна ли повторная генерация после критики. "
            f"Добиться оценки не ниже {settings.min_quality_score}/10 от Critic Agent."
        ),
        backstory=(
            "Опытный проектный менеджер дизайн-студии. "
            "Знает когда нужно глубокое исследование, а когда можно пропустить этап. "
            "Читает результаты каждого агента перед следующим шагом — "
            "не запускает генерацию если бриф слабый. "
            "Умеет интерпретировать оценки критика и формулировать точные правки. "
            "Не останавливается пока качество не достигнет цели."
        ),
        tools=[
            RunResearchTool(),
            RunStyleAnalysisTool(),
            RunStyleRevisionTool(),
            RunGenerationTool(),
            RunCritiqueTool(),
            ReadWorkspaceFileTool(),
            PromoteOutputTool(),
        ],
        llm=_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=settings.max_orchestrator_iter,
    )


def orchestrator_task(prompt: str, output_format: str) -> Task:
    return Task(
        description=_load_prompt("orchestrator_task").format(
            prompt=prompt,
            output_format=output_format,
            min_score=settings.min_quality_score,
        ),
        expected_output=(
            "Финальный отчёт: путь к созданному файлу, итоговая оценка, "
            "краткое описание принятых решений (почему такой стиль, какие итерации были)."
        ),
        agent=orchestrator_agent(),
    )