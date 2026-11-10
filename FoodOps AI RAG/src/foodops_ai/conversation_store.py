"""Persistent local conversation history backed by SQLite."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ConversationTurn:
    """One user question and assistant answer in a conversation."""

    question: str
    answer: str
    sources: list[dict[str, str | int]]
    created_at: datetime


class ConversationStore:
    """Store and retrieve conversation turns in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_id
                ON conversation_turns (conversation_id, id)
                """
            )

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, str | int]],
    ) -> ConversationTurn:
        """Persist a turn and return it with its UTC timestamp."""

        created_at = datetime.now(UTC)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns (
                    conversation_id, question, answer, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    question,
                    answer,
                    json.dumps(sources, ensure_ascii=False),
                    created_at.isoformat(),
                ),
            )
        return ConversationTurn(
            question=question,
            answer=answer,
            sources=sources,
            created_at=created_at,
        )

    def get_history(self, conversation_id: str) -> list[ConversationTurn]:
        """Return all turns for one conversation in chronological order."""

        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT question, answer, sources_json, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        return [self._to_turn(row) for row in rows]

    def get_recent_turns(
        self, conversation_id: str, limit: int = 5
    ) -> list[ConversationTurn]:
        """Return the latest turns in chronological order for prompt context."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT question, answer, sources_json, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [self._to_turn(row) for row in reversed(rows)]

    def _to_turn(self, row: tuple[str, str, str, str]) -> ConversationTurn:
        question, answer, sources_json, created_at = row
        return ConversationTurn(
            question=question,
            answer=answer,
            sources=json.loads(sources_json),
            created_at=datetime.fromisoformat(created_at),
        )
