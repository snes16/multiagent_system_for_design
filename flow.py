from __future__ import annotations
import json
import os
import re
from pathlib import Path

from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, router, start
from rich.console import Console
from rich.panel import Panel

from agents.builders import (
    CRITIQUE_FILE,
    critic_agent,
    critique_task,
    generation_task,
    generator_agent,
    research_agent,
    research_task,
    style_analyst_agent,
    style_task,
)
from models.state import CritiqueResult, DesignBrief, DesignerState, OutputFormat
from tools.workspace import workspace_path

console = Console()
MIN_SCORE = float(os.getenv("MIN_QUALITY_SCORE", "7.0"))


class DesignerFlow(Flow[DesignerState]):
    """
    Пайплайн:
      research → analyze_style → generate → critique
                                    ↑____________|  (если should_iterate)

    Данные между агентами передаются через файлы в output/.workspace/,
    а не через self.state или task.description — контекст не засоряется.
    """

    # ── 1. Research ───────────────────────────────────────────────────────────

    @start()
    def research(self) -> str:
        console.print(Panel(
            f"[bold]Задача:[/bold] {self.state.prompt}\n"
            f"[bold]Формат:[/bold] {self.state.output_format.value}",
            title="[cyan]Step 1 / Research Agent[/cyan]",
            border_style="cyan",
        ))

        agent = research_agent()
        task = research_task(agent, self.state.prompt)
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Не читаем результат в state — он лежит в output/.workspace/references.md
        # Style Analyst прочитает сам через FileReaderTool
        console.print("[green]Research done[/green] → output/.workspace/references.md")
        return "analyze"

    # ── 2. Style Analysis ─────────────────────────────────────────────────────

    @listen("analyze")
    def analyze_style(self) -> str:
        console.print(Panel(
            "Читаю references.md → синтезирую бриф → сохраняю brief.json",
            title="[cyan]Step 2 / Style Analyst[/cyan]",
            border_style="cyan",
        ))

        agent = style_analyst_agent()
        task = style_task(agent, self.state.prompt)
        # Никаких данных в description — агент сам читает файл
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Парсим бриф из файла для state (нужен только для финального лога)
        brief = _read_workspace_model("brief.json", DesignBrief)
        if brief:
            self.state.brief = brief
            console.print(
                f"[green]Brief ready[/green] → {brief.font_heading} / {brief.color_primary}"
            )
        else:
            console.print("[yellow]Brief saved but parse failed for state[/yellow]")

        return "generate"

    # ── 3. Generate ───────────────────────────────────────────────────────────

    @listen("generate")
    def generate(self) -> str:
        console.print(Panel(
            f"Читаю brief.json → генерирую дизайн (итерация {self.state.iteration + 1})",
            title="[cyan]Step 3 / Design Generator[/cyan]",
            border_style="cyan",
        ))

        # Правки критика берём из файла — не из self.state
        revision_notes = _build_revision_notes()

        agent = generator_agent()
        task = generation_task(
            agent,
            self.state.prompt,
            self.state.output_format.value,
            revision_notes=revision_notes,
        )
        # Никаких данных в description — агент сам читает brief.json
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        self.state.output_path = _find_latest_output(self.state.output_format)
        console.print(f"[green]Generated[/green] → {self.state.output_path}")
        return "critique"

    # ── 4. Critique ───────────────────────────────────────────────────────────

    @listen("critique")
    def critique(self) -> str:
        console.print(Panel(
            f"Читаю brief.json + {self.state.output_path} → оцениваю",
            title="[cyan]Step 4 / Critic Agent[/cyan]",
            border_style="cyan",
        ))

        agent = critic_agent()
        # Передаём только путь к финальному файлу — агент сам читает бриф
        task = critique_task(agent, self.state.prompt, self.state.output_path)
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Читаем результат из файла
        critique_result = _read_workspace_model("critique.json", CritiqueResult)
        if critique_result:
            self.state.critique = critique_result
            color = "green" if critique_result.overall_score >= MIN_SCORE else "yellow"
            console.print(
                f"[{color}]Score: {critique_result.overall_score:.1f}/10[/{color}]"
                f" — {critique_result.verdict}"
            )
        else:
            console.print("[yellow]Critique file not parsed — принимаем как есть[/yellow]")
            self.state.accepted = True
            return "done"

        return "check_quality"

    # ── 5. Router ─────────────────────────────────────────────────────────────

    @router(critique)
    def check_quality(self) -> str:
        c = self.state.critique
        if not c:
            return "done"

        if c.overall_score >= MIN_SCORE or not c.should_iterate:
            self.state.accepted = True
            console.print("[green]Качество принято.[/green]")
            return "done"

        console.print(
            f"[yellow]Нужна доработка (score {c.overall_score:.1f} < {MIN_SCORE})[/yellow]"
        )
        for imp in c.improvements[:3]:
            console.print(f"  • {imp}")

        self.state.iteration += 1
        return "generate"

    # ── 6. Done ───────────────────────────────────────────────────────────────

    @listen("done")
    def done(self):
        score_line = ""
        if self.state.critique:
            c = self.state.critique
            score_line = (
                f"\nОценка: {c.overall_score:.1f}/10 — {c.verdict}"
                f"\nСильные стороны: {', '.join(c.strengths[:2])}"
            )

        console.print(Panel(
            f"[bold green]Готово![/bold green]\n"
            f"Файл: {self.state.output_path}\n"
            f"Итераций: {self.state.iteration + 1}"
            f"{score_line}\n\n"
            f"[dim]Workspace: output/.workspace/ (references.md, brief.json, critique.json)[/dim]",
            title="Результат",
            border_style="green",
        ))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_workspace_model(filename: str, model_cls):
    """Читает JSON из workspace и валидирует через Pydantic."""
    path = workspace_path(filename)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        start = clean.rfind("{")
        end = clean.rfind("}") + 1
        if start == -1 or end <= 0:
            return None
        return model_cls(**json.loads(clean[start:end]))
    except Exception:
        return None


def _build_revision_notes() -> str:
    """Читает critique.json из workspace если есть — для revision hints."""
    critique = _read_workspace_model("critique.json", CritiqueResult)
    if not critique:
        return ""
    return (
        f"Предыдущая оценка: {critique.overall_score:.1f}/10\n"
        f"Улучши: {'; '.join(critique.improvements)}\n"
        f"{critique.revised_brief_notes}"
    )


def _find_latest_output(fmt: OutputFormat) -> str:
    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    ext = ".svg" if fmt == OutputFormat.SVG else ".html"
    files = sorted(out_dir.glob(f"*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else str(out_dir)