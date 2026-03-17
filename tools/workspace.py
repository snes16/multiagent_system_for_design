from __future__ import annotations
import os
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def workspace_dir() -> Path:
    """Папка для временных файлов пайплайна."""
    out = Path(os.getenv("OUTPUT_DIR", "./output"))
    ws = out / ".workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def workspace_path(filename: str) -> Path:
    return workspace_dir() / filename


# ── FileReaderTool ────────────────────────────────────────────────────────────

class ReadInput(BaseModel):
    filepath: str = Field(description="Путь к файлу для чтения")


class FileReaderTool(BaseTool):
    name: str = "file_reader"
    description: str = (
        "Читает содержимое файла по пути. "
        "Используй чтобы загрузить только нужные данные: "
        "references.md, brief.json, critique.json из папки output/.workspace/"
    )
    args_schema: type[BaseModel] = ReadInput

    def _run(self, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists():
            return f"[file_reader] Файл не найден: {filepath}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"[file_reader] Ошибка чтения {filepath}: {e}"


# ── WorkspaceWriterTool ───────────────────────────────────────────────────────

class WorkspaceWriteInput(BaseModel):
    filename: str = Field(description="Имя файла в workspace, напр. references.md или brief.json")
    content: str = Field(description="Содержимое файла")


class WorkspaceWriterTool(BaseTool):
    name: str = "workspace_writer"
    description: str = (
        "Сохраняет промежуточный файл в output/.workspace/. "
        "Используй для: references.md (Research), brief.json (Style Analyst), "
        "critique.json (Critic). Данные останутся на диске и не будут засорять контекст."
    )
    args_schema: type[BaseModel] = WorkspaceWriteInput

    def _run(self, filename: str, content: str) -> str:
        path = workspace_path(filename)
        path.write_text(content, encoding="utf-8")
        return f"workspace_saved:{path}"