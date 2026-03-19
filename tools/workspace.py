from __future__ import annotations
import logging
import os
from pathlib import Path
logger = logging.getLogger(__name__)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def workspace_dir() -> Path:
    """Directory for temporary pipeline files."""
    out = Path(os.getenv("OUTPUT_DIR", "./output"))
    ws = out / ".workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def workspace_path(filename: str) -> Path:
    return workspace_dir() / filename


def staging_dir() -> Path:
    """Temporary output before quality check passes."""
    out = Path(os.getenv("OUTPUT_DIR", "./output"))
    stg = out / ".staging"
    stg.mkdir(parents=True, exist_ok=True)
    return stg


def staging_path(filename: str) -> Path:
    return staging_dir() / filename


def promote_to_output(filename: str) -> str:
    """Move file from staging to output/ after quality check passes."""
    import shutil
    src = staging_dir() / filename
    dst = Path(os.getenv("OUTPUT_DIR", "./output")) / filename
    if not src.exists():
        return f"[promote] staging file not found: {src}"
    shutil.move(str(src), str(dst))
    logger.info("Promoted %s → %s", src, dst)
    return str(dst)


# ── FileReaderTool ────────────────────────────────────────────────────────────

class ReadInput(BaseModel):
    filepath: str = Field(description="Path to the file to read")


class FileReaderTool(BaseTool):
    name: str = "file_reader"
    description: str = (
        "Reads file contents by path. "
        "Use to load only the needed data: "
        "references.md, brief.json, critique.json from the output/.workspace/ directory"
    )
    args_schema: type[BaseModel] = ReadInput

    def _run(self, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists():
            return f"[file_reader] File not found: {filepath}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("file_reader failed path=%s: %s", filepath, e)
            return f"[file_reader] Read error {filepath}: {e}"


# ── WorkspaceWriterTool ───────────────────────────────────────────────────────

class WorkspaceWriteInput(BaseModel):
    filename: str = Field(description="Filename in workspace, e.g. references.md or brief.json")
    content: str = Field(description="File contents")


class WorkspaceWriterTool(BaseTool):
    name: str = "workspace_writer"
    description: str = (
        "Saves an intermediate file to output/.workspace/. "
        "Use for: references.md (Research), brief.json (Style Analyst), "
        "critique.json (Critic). Data persists on disk and won't pollute the context."
    )
    args_schema: type[BaseModel] = WorkspaceWriteInput

    def _run(self, filename: str, content: str) -> str:
        path = workspace_path(filename)
        path.write_text(content, encoding="utf-8")
        return f"workspace_saved:{path}"