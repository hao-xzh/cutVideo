"""Small persistent UI preferences shared by both workspaces."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings


def _settings() -> QSettings:
    return QSettings("Hao", "cutVideo")


def dialog_start(key: str, fallback: str = "") -> str:
    stored = str(_settings().value(f"dialogs/{key}", "") or "")
    if stored and Path(stored).is_dir():
        return stored
    candidate = Path(fallback).expanduser() if fallback else None
    if candidate is not None:
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir():
            return str(candidate)
    return ""


def remember_dialog_path(key: str, selected_path: str) -> None:
    path = Path(selected_path).expanduser()
    directory = path if path.is_dir() else path.parent
    if directory.is_dir():
        _settings().setValue(f"dialogs/{key}", str(directory.resolve()))


def preferred_theme() -> str:
    value = str(_settings().value("appearance/theme", "dark") or "dark").casefold()
    return "light" if value == "light" else "dark"


def remember_theme(theme: str) -> None:
    normalized = "light" if str(theme).casefold() == "light" else "dark"
    _settings().setValue("appearance/theme", normalized)


__all__ = ["dialog_start", "preferred_theme", "remember_dialog_path", "remember_theme"]
