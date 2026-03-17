"""Unit tests for CrewAI tools — no API calls, no external services."""
from __future__ import annotations
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── WorkspaceWriterTool ───────────────────────────────────────────────────────

def test_workspace_writer_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from tools.workspace import WorkspaceWriterTool
    tool = WorkspaceWriterTool()
    result = tool._run(filename="test.md", content="hello world")
    assert "workspace_saved" in result
    written = (tmp_path / ".workspace" / "test.md").read_text()
    assert written == "hello world"


def test_workspace_writer_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from tools.workspace import WorkspaceWriterTool
    tool = WorkspaceWriterTool()
    tool._run(filename="nested.json", content="{}")
    assert (tmp_path / ".workspace" / "nested.json").exists()


# ── FileReaderTool ────────────────────────────────────────────────────────────

def test_file_reader_reads_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    target = tmp_path / "sample.txt"
    target.write_text("sample content")
    from tools.workspace import FileReaderTool
    tool = FileReaderTool()
    result = tool._run(filepath=str(target))
    assert result == "sample content"


def test_file_reader_missing_file(tmp_path):
    from tools.workspace import FileReaderTool
    tool = FileReaderTool()
    result = tool._run(filepath=str(tmp_path / "does_not_exist.txt"))
    assert "not found" in result.lower() or "File not found" in result


# ── FileWriterTool ────────────────────────────────────────────────────────────

def test_file_writer_saves_to_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from tools.design_tools import FileWriterTool
    tool = FileWriterTool()
    result = tool._run(filename="landing.html", content="<html></html>")
    assert "saved" in result
    assert (tmp_path / "landing.html").read_text() == "<html></html>"


# ── ReadWorkspaceFileTool ─────────────────────────────────────────────────────

def test_read_workspace_file_truncates_large_content(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    ws = tmp_path / ".workspace"
    ws.mkdir()
    big = "x" * 5000
    (ws / "big.md").write_text(big)
    from tools.orchestrator_tools import ReadWorkspaceFileTool
    tool = ReadWorkspaceFileTool()
    result = tool._run(filename="big.md")
    assert len(result) < 5000
    assert "truncated" in result.lower() or "..." in result


def test_read_workspace_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    from tools.orchestrator_tools import ReadWorkspaceFileTool
    tool = ReadWorkspaceFileTool()
    result = tool._run(filename="missing.json")
    assert "not found" in result.lower()


# ── Settings ──────────────────────────────────────────────────────────────────

def test_settings_defaults():
    from config.settings import Settings
    s = Settings(anthropic_api_key="test", firecrawl_api_key="test")
    assert s.research_model == "claude-sonnet-4-5"
    assert s.min_quality_score == 7.0
    assert s.max_orchestrator_iter == 12


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("RESEARCH_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("MIN_QUALITY_SCORE", "8.5")
    from config.settings import Settings
    s = Settings(anthropic_api_key="test", firecrawl_api_key="test")
    assert s.research_model == "claude-haiku-4-5"
    assert s.min_quality_score == 8.5


# ── _run_crew_with_retry ──────────────────────────────────────────────────────

def test_retry_succeeds_on_second_attempt():
    from tools.orchestrator_tools import _run_crew_with_retry
    call_count = 0
    mock_crew = MagicMock()

    def flaky_kickoff():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("temporary error")
        return "success"

    mock_crew.kickoff = flaky_kickoff
    with patch("time.sleep"):
        result = _run_crew_with_retry(mock_crew, max_retries=3, wait=1)
    assert result == "success"
    assert call_count == 2


def test_retry_raises_after_all_attempts():
    from tools.orchestrator_tools import _run_crew_with_retry
    mock_crew = MagicMock()
    mock_crew.kickoff.side_effect = ConnectionError("always fails")
    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            _run_crew_with_retry(mock_crew, max_retries=3, wait=1)
