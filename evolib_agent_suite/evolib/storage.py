from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class LibraryStorage(Protocol):
    """Storage adapter interface for EvolvingLibrary payloads."""

    def load(self) -> Optional[Dict[str, Any]]:
        """Return a serialized library payload, or None when no library exists."""

    def save(self, payload: Dict[str, Any]) -> None:
        """Persist a serialized library payload."""


class JsonLibraryStorage:
    """Backward-compatible storage for the existing library.json format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SQLiteLibraryStorage:
    """Experimental SQLite backend that stores the canonical library payload.

    This intentionally keeps a single JSON payload in SQLite so callers can
    experiment with a SQLite-backed file without changing EvolvingLibrary's
    schema or losing compatibility with the JSON representation.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with self._connect() as conn:
            self._ensure_schema(conn)
            row = conn.execute("SELECT payload FROM library_state WHERE id = 1").fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO library_state (id, payload, updated_at)
                VALUES (1, ?, strftime('%s','now'))
                ON CONFLICT(id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (serialized,),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS library_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
