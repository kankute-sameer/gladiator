"""Tests for glad.agent.script: question-set loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from glad.conversation import session as script


def _write_set(tmp_path: Path, data: dict) -> None:
    (tmp_path / "test_set.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _load_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "test_set"):
    monkeypatch.setattr(script, "_QUESTION_SETS_DIR", tmp_path)
    return script.load_question_set(name)


def test_valid_set_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(
        tmp_path,
        {
            "id": "test_set",
            "version": 1,
            "questions": [
                {"id": "budget", "text": "What's your budget?"},
                {"id": "timeline", "text": "What's your timeline?"},
            ],
        },
    )

    question_set = _load_from(tmp_path, monkeypatch)

    assert question_set.id == "test_set"
    assert question_set.version == 1
    assert [q.id for q in question_set.questions] == ["budget", "timeline"]
    assert question_set.get("budget").text == "What's your budget?"
    assert question_set.get("budget").notes is None
    assert question_set.namespaced_id("budget") == "test_set.budget"


def test_optional_notes_are_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(
        tmp_path,
        {
            "id": "test_set",
            "version": 1,
            "questions": [
                {
                    "id": "income",
                    "text": "What's your income?",
                    "notes": "Gross annual, USD.",
                },
            ],
        },
    )

    question_set = _load_from(tmp_path, monkeypatch)
    assert question_set.get("income").notes == "Gross annual, USD."


def test_duplicate_ids_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(
        tmp_path,
        {
            "id": "test_set",
            "version": 1,
            "questions": [
                {"id": "budget", "text": "What's your budget?"},
                {"id": "budget", "text": "Budget again?"},
            ],
        },
    )

    with pytest.raises(ValueError, match="duplicate"):
        _load_from(tmp_path, monkeypatch)


def test_invalid_id_characters_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(
        tmp_path,
        {
            "id": "test_set",
            "version": 1,
            "questions": [
                {"id": "Budget-1", "text": "What's your budget?"},
            ],
        },
    )

    with pytest.raises(ValueError, match="invalid question id"):
        _load_from(tmp_path, monkeypatch)


def test_empty_question_list_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(tmp_path, {"id": "test_set", "version": 1, "questions": []})

    with pytest.raises(ValueError, match="no questions"):
        _load_from(tmp_path, monkeypatch)


def test_missing_id_field_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_set(tmp_path, {"version": 1, "questions": [{"id": "budget", "text": "x"}]})

    with pytest.raises(ValueError, match="'id'"):
        _load_from(tmp_path, monkeypatch)
