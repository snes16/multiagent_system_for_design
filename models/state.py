from __future__ import annotations
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from crewai.flow.flow import FlowState


class OutputFormat(str, Enum):
    HTML = "html"
    SVG = "svg"
    MOODBOARD = "moodboard"
    BRANDBOOK = "brandbook"


class DesignReference(BaseModel):
    source: str
    project_name: str
    url: str = ""
    key_patterns: str
    color_notes: str = ""
    typography_notes: str = ""


class DesignBrief(BaseModel):
    color_primary: str
    color_secondary: str
    color_accent: str
    color_background: str
    color_text: str
    font_heading: str
    font_body: str
    style_direction: str
    mood: str
    layout_pattern: str
    visual_metaphor: str
    key_elements: list[str] = Field(default_factory=list)
    target_audience: str = ""
    brand_personality: str = ""


class CritiqueResult(BaseModel):
    overall_score: float = Field(ge=0, le=10)
    visual_hierarchy_score: float = Field(ge=0, le=10)
    typography_score: float = Field(ge=0, le=10)
    color_harmony_score: float = Field(ge=0, le=10)
    layout_score: float = Field(ge=0, le=10)
    brief_alignment_score: float = Field(ge=0, le=10)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    verdict: str = ""
    should_iterate: bool = False
    revised_brief_notes: str = ""


class DesignerState(FlowState):
    """Typed state — automatically passed between all Flow steps."""

    # Input
    prompt: str = ""
    output_format: OutputFormat = OutputFormat.HTML

    # Pipeline data (populated by agents as the pipeline runs)
    references_raw: str = ""
    brief: Optional[DesignBrief] = None
    brief_raw: str = ""
    output_path: str = ""
    critique: Optional[CritiqueResult] = None
    critique_raw: str = ""

    # Iterations
    iteration: int = 0
    accepted: bool = False
    termination_reason: str = ""  # "quality_passed" | "max_iterations" | "no_critique"
    generation_count: int = 0       # hard counter incremented in code, not by LLM
    best_output_path: str = ""      # path with the best score seen so far
    best_score: float = 0.0         # best overall_score seen across all iterations