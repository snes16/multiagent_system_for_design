from __future__ import annotations
import os
from pathlib import Path
from crewai import Agent, Task, LLM
from config.settings import settings


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    return (Path(__file__).parent.parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")

from tools.design_tools import (
    FirecrawlScrapeTool,
    FirecrawlCrawlTool,
    WebSearchTool,
    FileWriterTool,
)
from tools.workspace import FileReaderTool, WorkspaceWriterTool

REFS_FILE     = "output/.workspace/references.md"
BRIEF_FILE    = "output/.workspace/brief.json"
CRITIQUE_FILE = "output/.workspace/critique.json"


def _llm(model: str, temperature: float) -> LLM:
    return LLM(
        model=f"anthropic/{model}",
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_tokens=8192,
    )


def research_agent() -> Agent:
    return Agent(
        role="Design Research Specialist",
        goal=(
            "Collect 4-6 visual references and save them to the workspace. "
            "Crawl artlebedev.ru/everything/, search Awwwards and Dribbble. "
            f"Save the result to {REFS_FILE} via workspace_writer."
        ),
        backstory=(
            "An experienced art director who quickly finds the best design solutions. "
            "Knows the Artlebedev catalog well — /everything/ contains all projects. "
            "Always saves results to a file — never keeps data only in memory."
        ),
        tools=[FirecrawlCrawlTool(), FirecrawlScrapeTool(), WebSearchTool(), WorkspaceWriterTool()],
        llm=_llm(settings.research_model, settings.research_temperature),
        max_iter=settings.max_research_iter,
        verbose=True,
        allow_delegation=False,
    )


def style_analyst_agent() -> Agent:
    return Agent(
        role="Senior Art Director & Style Analyst",
        goal=(
            f"Read {REFS_FILE}, synthesize a DesignBrief. "
            f"Save the JSON brief to {BRIEF_FILE} via workspace_writer."
        ),
        backstory=(
            "An art director with experience at Pentagram and Artlebedev. "
            "Always reads data from files rather than task descriptions — "
            "this allows working with large volumes of references without losing quality."
        ),
        tools=[FileReaderTool(), WebSearchTool(), WorkspaceWriterTool()],
        llm=_llm(settings.analyst_model, settings.analyst_temperature),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def generator_agent() -> Agent:
    return Agent(
        role="Senior Frontend Designer & Developer",
        goal=(
            f"Read {BRIEF_FILE}, create a production-ready design. "
            "Save the final file via file_writer to output/."
        ),
        backstory=(
            "A senior frontend developer who creates visually outstanding interfaces. "
            "Always works with up-to-date data read from files."
        ),
        tools=[FileReaderTool(), FileWriterTool()],
        llm=_llm(settings.generator_model, settings.generator_temperature),
        max_iter=settings.max_generation_iter,
        verbose=True,
        allow_delegation=False,
    )


def critic_agent() -> Agent:
    return Agent(
        role="Design Critic & Quality Reviewer",
        goal=(
            f"Read {BRIEF_FILE}, evaluate the design against 5 criteria. "
            f"Save the CritiqueResult JSON to {CRITIQUE_FILE} via workspace_writer."
        ),
        backstory=(
            "A tough but fair critic. "
            "Checks the result against the brief by reading both from files."
        ),
        tools=[FileReaderTool(), WorkspaceWriterTool()],
        llm=_llm(settings.critic_model, settings.critic_temperature),
        max_iter=settings.max_critic_iter,
        verbose=True,
        allow_delegation=False,
    )


# ── Tasks — file paths only, no inline data in description ────────────────────

def research_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=_load_prompt("research_task").replace("{prompt}", prompt),
        expected_output=f"Confirmation of save: {REFS_FILE}",
        agent=agent,
    )


def style_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=_load_prompt("style_task").format(prompt=prompt, refs_file=REFS_FILE),
        expected_output=f"Confirmation of save: {BRIEF_FILE}",
        agent=agent,
    )


_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "html": (
        "Create a complete single-page HTML landing.\n"
        "- <!DOCTYPE html>, Google Fonts via @import\n"
        "- At least 5 sections: hero, about, services, showcase, contacts\n"
        "- CSS variables for colors and fonts from the brief\n"
        "- Hover animations, responsiveness (media queries)\n"
        "- Real content, not Lorem ipsum; decorations via CSS/inline SVG\n"
        "Save: file_writer(filename=\"landing.html\", content=<full HTML>)"
    ),
    "svg": (
        "Create an SVG layout 1440x900px.\n"
        "- viewBox=\"0 0 1440 900\", fonts via <defs><style>@import\n"
        "- Colors from the brief, structure: header, hero, content, footer\n"
        "Save: file_writer(filename=\"layout.svg\", content=<full SVG>)"
    ),
    "moodboard": (
        "Create an HTML moodboard.\n"
        "- CSS Grid layout, color chips with HEX, typography samples\n"
        "- Reference cards, visual keys block\n"
        "Save: file_writer(filename=\"moodboard.html\", content=<full HTML>)"
    ),
    "brandbook": (
        "Create an HTML brandbook.\n"
        "Sections: mission, logo (SVG inline), color system,\n"
        "typography, visual language, usage examples, do/don't.\n"
        "Save: file_writer(filename=\"brandbook.html\", content=<full HTML>)"
    ),
}


def generation_task(agent: Agent, prompt: str, output_format: str,
                    revision_notes: str = "") -> Task:
    instructions = _FORMAT_INSTRUCTIONS.get(output_format, _FORMAT_INSTRUCTIONS["html"])
    revision_block = f"\n[CRITIC REVISIONS]\n{revision_notes}\n" if revision_notes else ""

    return Task(
        description=_load_prompt("generation_task").format(
            prompt=prompt,
            output_format=output_format,
            revision_block=revision_block,
            brief_file=BRIEF_FILE,
            instructions=instructions,
        ),
        expected_output="Confirmation of final file saved to output/.",
        agent=agent,
    )


def critique_task(agent: Agent, prompt: str, output_path: str) -> Task:
    return Task(
        description=_load_prompt("critique_task").format(
            prompt=prompt,
            output_path=output_path,
            brief_file=BRIEF_FILE,
        ),
        expected_output=f"Confirmation of save: {CRITIQUE_FILE}",
        agent=agent,
    )