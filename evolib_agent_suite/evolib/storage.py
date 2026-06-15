from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol


class LibraryStorage(Protocol):
    """Storage adapter protocol for serialized EvoLib library payloads."""

    path: Path
    backend: str

    def load(self) -> Dict[str, Any]:
        """Load a serialized library payload, or an empty payload when absent."""

    def save(self, payload: Dict[str, Any]) -> None:
        """Persist a serialized library payload."""


class JsonLibraryStorage:
    """Backward-compatible JSON storage for the existing library.json format."""

    backend = "json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class SQLiteLibraryStorage:
    """Experimental SQLite storage for EvoLib library payloads.

    This adapter intentionally persists the same serialized payload contract as
    JsonLibraryStorage. That keeps the first SQLite backend backward-compatible at
    the EvoLib boundary while leaving room to normalize selected event streams in
    later experiments.
    """

    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT payload FROM library_state WHERE id = 1"
            ).fetchone()
        if row is None:
            return {}
        return json.loads(row[0])

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with sqlite3.connect(self.path) as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO library_state (id, payload, updated_at)
                VALUES (1, ?, strftime('%s', 'now'))
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (serialized,),
            )
            conn.commit()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS library_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """)


def build_library_storage(
    path: str | Path,
    storage_config: Mapping[str, Any] | None = None,
) -> LibraryStorage:
    """Build the configured storage backend, defaulting to JSON."""

    cfg = storage_config or {}
    backend = str(cfg.get("backend", "json")).lower()
    if backend == "json":
        return JsonLibraryStorage(path)
    if backend == "sqlite":
        return SQLiteLibraryStorage(path)
    raise ValueError(f"Unsupported library.storage.backend: {backend}")
