from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol


class LibraryStorage(Protocol):
    path: Path
    backend: str

    def load(self) -> Dict[str, Any]: ...

    def save(self, payload: Dict[str, Any]) -> None: ...


class JsonLibraryStorage:
    backend = "json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SQLiteLibraryStorage:
    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path) as conn:
            self._ensure_schema(conn)
            row = conn.execute("SELECT payload FROM library_state WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else {}

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with sqlite3.connect(self.path) as conn:
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO library_state (id, payload, updated_at)
                VALUES (1, ?, strftime('%s', 'now'))
                ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (serialized,),
            )
            conn.commit()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS library_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )


def build_library_storage(path: str | Path, config: Mapping[str, Any] | None = None) -> LibraryStorage:
    backend = str((config or {}).get("backend", "json")).lower()
    if backend == "json":
        return JsonLibraryStorage(path)
    if backend == "sqlite":
        return SQLiteLibraryStorage(path)
    raise ValueError(f"Unsupported library.storage.backend: {backend}")
