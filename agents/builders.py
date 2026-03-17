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
        llm=_llm("claude-sonnet-4-5", temperature=0.3),
        verbose=True,
        allow_delegation=False,
        max_iter=6,
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
        llm=_llm("claude-opus-4-5", temperature=0.8),
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
        llm=_llm("claude-opus-4-5", temperature=0.9),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
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
        llm=_llm("claude-sonnet-4-5", temperature=0.2),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


# ── Tasks — file paths only, no inline data in description ────────────────────

def research_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=f"""
Research design references for the task: **{prompt}**

1. Crawl Artlebedev:
   firecrawl_crawl(url="https://www.artlebedev.ru/everything/", limit=8)
   For 2-3 projects — firecrawl_scrape with extract_prompt:
   "Extract: project name, visual style, colors used, typography, key design patterns"

2. Search for additional references:
   web_search: "{prompt} website design awwwards 2024"
   web_search: "{prompt} brand identity dribbble behance"

3. Compile a report on the 4-6 best references:
   SOURCE / PROJECT / URL / KEY_PATTERNS / COLOR_NOTES / TYPOGRAPHY_NOTES
   + SYNTHESIS block (3 sentences on common trends)

4. Save the FULL result:
   workspace_writer(filename="references.md", content=<full report>)
""",
        expected_output=f"Confirmation of save: {REFS_FILE}",
        agent=agent,
    )


def style_task(agent: Agent, prompt: str) -> Task:
    return Task(
        description=f"""
Create a DesignBrief for the task: **{prompt}**

1. Read the references:
   file_reader(filepath="{REFS_FILE}")

2. Synthesize the brief:
   - Google Fonts with character (not Inter, not Roboto):
     Headings: Cormorant Garamond / Syne / DM Serif Display / Fraunces / Playfair Display
     Body: DM Sans / Plus Jakarta Sans / Outfit / Manrope / Epilogue
   - 5 specific HEX colors
   - style_direction: at least 3 sentences

3. Save JSON (no markdown wrapper):
   workspace_writer(filename="brief.json", content=<JSON>)

JSON format:
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
        description=f"""
Create a design for the task: **{prompt}**
Format: **{output_format}**
{revision_block}
1. Read the brief:
   file_reader(filepath="{BRIEF_FILE}")

2. {instructions}

Strictly follow the colors and fonts from the brief.
""",
        expected_output="Confirmation of final file saved to output/.",
        agent=agent,
    )


def critique_task(agent: Agent, prompt: str, output_path: str) -> Task:
    return Task(
        description=f"""
Evaluate the design for the task: **{prompt}**
File to evaluate: {output_path}

1. Read the brief for cross-checking:
   file_reader(filepath="{BRIEF_FILE}")

2. Score the design (each criterion 0.0–10.0):
   - visual_hierarchy_score, typography_score, color_harmony_score
   - layout_score, brief_alignment_score
   overall_score = average; should_iterate=true if < 7.0

3. Save the result:
   workspace_writer(filename="critique.json", content=<JSON without markdown>)

JSON format:
{{
  "overall_score": 0.0,
  "visual_hierarchy_score": 0.0, "typography_score": 0.0,
  "color_harmony_score": 0.0, "layout_score": 0.0, "brief_alignment_score": 0.0,
  "strengths": ["...", "..."], "improvements": ["...", "..."],
  "verdict": "...", "should_iterate": false, "revised_brief_notes": "..."
}}
""",
        expected_output=f"Confirmation of save: {CRITIQUE_FILE}",
        agent=agent,
    )