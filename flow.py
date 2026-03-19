from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path

from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, router, start
from rich.console import Console
from rich.panel import Panel
logger = logging.getLogger(__name__)

from agents.builders import (
    critic_agent,
    critique_task,
    generation_task,
    generator_agent,
    research_agent,
    research_task,
    style_analyst_agent,
    style_task,
)
from config.settings import settings
from models.state import CritiqueResult, DesignBrief, DesignerState, OutputFormat
from tools.workspace import workspace_path

console = Console()


class DesignerFlow(Flow[DesignerState]):
    """
    Pipeline:
      research → analyze_style → generate → critique
                                    ↑____________|  (if should_iterate)

    Data between agents is passed through files in output/.workspace/,
    not through self.state or task.description — keeps context clean.
    """

    # ── 1. Research ───────────────────────────────────────────────────────────

    @start()
    def research(self) -> str:
        from tools.orchestrator_tools import _generation_counter
        _generation_counter.clear()

        console.print(Panel(
            f"[bold]Task:[/bold] {self.state.prompt}\n"
            f"[bold]Format:[/bold] {self.state.output_format.value}",
            title="[cyan]Step 1 / Research Agent[/cyan]",
            border_style="cyan",
        ))
        logger.info("Pipeline started | prompt=%r format=%s", self.state.prompt, self.state.output_format.value)

        agent = research_agent()
        task = research_task(agent, self.state.prompt)
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Result is not read into state — it lives in output/.workspace/references.md
        # Style Analyst will read it directly via FileReaderTool
        console.print("[green]Research done[/green] → output/.workspace/references.md")
        return "analyze"

    # ── 2. Style Analysis ─────────────────────────────────────────────────────

    @listen("analyze")
    def analyze_style(self) -> str:
        console.print(Panel(
            "Reading references.md → synthesizing brief → saving brief.json",
            title="[cyan]Step 2 / Style Analyst[/cyan]",
            border_style="cyan",
        ))

        agent = style_analyst_agent()
        task = style_task(agent, self.state.prompt)
        # No data in description — agent reads the file itself
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Parse brief from file into state (only needed for final log)
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
            f"Reading brief.json → generating design (iteration {self.state.iteration + 1})",
            title="[cyan]Step 3 / Design Generator[/cyan]",
            border_style="cyan",
        ))

        # Critic's revision notes come from file — not from self.state
        revision_notes = _build_revision_notes()

        agent = generator_agent()
        task = generation_task(
            agent,
            self.state.prompt,
            self.state.output_format.value,
            revision_notes=revision_notes,
        )
        # No data in description — agent reads brief.json itself
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        self.state.output_path = _find_latest_output(self.state.output_format)
        console.print(f"[green]Generated[/green] → {self.state.output_path}")
        return "critique"

    # ── 4. Critique ───────────────────────────────────────────────────────────

    @listen("critique")
    def critique(self) -> str:
        console.print(Panel(
            f"Reading brief.json + {self.state.output_path} → evaluating",
            title="[cyan]Step 4 / Critic Agent[/cyan]",
            border_style="cyan",
        ))

        agent = critic_agent()
        # Only the output file path is passed — agent reads the brief itself
        task = critique_task(agent, self.state.prompt, self.state.output_path)
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

        # Read result from file
        critique_result = _read_workspace_model("critique.json", CritiqueResult)
        if critique_result:
            self.state.critique = critique_result
            color = "green" if critique_result.overall_score >= settings.min_quality_score else "yellow"
            console.print(
                f"[{color}]Score: {critique_result.overall_score:.1f}/10[/{color}]"
                f" — {critique_result.verdict}"
            )
        else:
            console.print("[yellow]Critique file not parsed — accepting as-is[/yellow]")
            self.state.accepted = True
            return "done"

        return "check_quality"

    # ── 5. Router ─────────────────────────────────────────────────────────────

    @router(critique)
    def check_quality(self) -> str:
        c = self.state.critique
        if not c:
            self.state.termination_reason = "no_critique"
            return "done"

        if c.overall_score >= settings.min_quality_score or not c.should_iterate:
            self.state.accepted = True
            self.state.termination_reason = "quality_passed"
            console.print("[green]Quality accepted.[/green]")
            return "done"

        console.print(
            f"[yellow]Revision needed (score {c.overall_score:.1f} < {settings.min_quality_score})[/yellow]"
        )
        for imp in c.improvements[:3]:
            console.print(f"  • {imp}")

        self.state.iteration += 1
        if self.state.iteration >= settings.max_generation_iter:
            self.state.termination_reason = "max_iterations"
            console.print(
                f"[yellow]Max iterations ({settings.max_generation_iter}) reached "
                f"— finishing with best effort result (score {c.overall_score:.1f}/10).[/yellow]"
            )
            logger.warning(
                "Pipeline stopped at iteration limit | score=%.1f min=%.1f",
                c.overall_score, settings.min_quality_score,
            )
            return "done"
        return "generate"

    # ── 6. Done ───────────────────────────────────────────────────────────────

    @listen("done")
    def done(self):
        from tools.workspace import promote_to_output
        if self.state.output_path and self.state.termination_reason in ("quality_passed", "max_iterations"):
            filename = Path(self.state.output_path).name
            promoted = promote_to_output(filename)
            if not promoted.startswith("[promote]"):
                self.state.output_path = promoted

        score_line = ""
        if self.state.critique:
            c = self.state.critique
            score_line = (
                f"\nScore: {c.overall_score:.1f}/10 — {c.verdict}"
                f"\nStrengths: {', '.join(c.strengths[:2])}"
            )

        reason = self.state.termination_reason
        if reason == "max_iterations":
            title = "Result [best effort]"
            border = "yellow"
            status = "[yellow]Stopped: iteration limit reached — result may not meet quality target.[/yellow]\n"
        else:
            title = "Result"
            border = "green"
            status = ""

        console.print(Panel(
            f"{status}"
            f"[bold]File:[/bold] {self.state.output_path}\n"
            f"[bold]Iterations:[/bold] {self.state.iteration + 1}"
            f"{score_line}\n\n"
            f"[dim]Workspace: output/.workspace/ (references.md, brief.json, critique.json)[/dim]",
            title=title,
            border_style=border,
        ))
        logger.info(
            "Pipeline finished | reason=%s output=%s score=%s",
            reason or "no_critique",
            self.state.output_path,
            self.state.critique.overall_score if self.state.critique else "n/a",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_workspace_model(filename: str, model_cls):
    """Reads JSON from workspace and validates via Pydantic."""
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
    """Reads critique.json from workspace if present — for revision hints."""
    critique = _read_workspace_model("critique.json", CritiqueResult)
    if not critique:
        return ""
    return (
        f"Previous score: {critique.overall_score:.1f}/10\n"
        f"Improve: {'; '.join(critique.improvements)}\n"
        f"{critique.revised_brief_notes}"
    )


def _find_latest_output(fmt: OutputFormat) -> str:
    from tools.workspace import staging_dir
    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    ext = ".svg" if fmt == OutputFormat.SVG else ".html"
    for search_dir in [out_dir, staging_dir()]:
        files = sorted(search_dir.glob(f"*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return str(files[0])
    return str(out_dir)