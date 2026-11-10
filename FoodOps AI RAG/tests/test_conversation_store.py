from pathlib import Path

from foodops_ai.conversation_store import ConversationStore


def test_conversation_history_persists_and_returns_turns_in_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "history.sqlite3"
    store = ConversationStore(database_path)
    store.add_turn("session-1", "Question one", "Answer one", [])
    store.add_turn("session-1", "Question two", "Answer two", [])

    reopened_store = ConversationStore(database_path)
    history = reopened_store.get_history("session-1")

    assert [turn.question for turn in history] == ["Question one", "Question two"]
    assert [turn.answer for turn in history] == ["Answer one", "Answer two"]
    assert history[0].created_at.tzinfo is not None


def test_recent_history_returns_latest_turns_in_chronological_order(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "history.sqlite3")
    for number in range(1, 7):
        store.add_turn("session-1", f"Question {number}", f"Answer {number}", [])

    recent = store.get_recent_turns("session-1", limit=3)

    assert [turn.question for turn in recent] == [
        "Question 4",
        "Question 5",
        "Question 6",
    ]


def test_recent_history_requires_positive_limit(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "history.sqlite3")

    try:
        store.get_recent_turns("session-1", limit=0)
    except ValueError as error:
        assert "greater than zero" in str(error)
    else:
        raise AssertionError("Expected ValueError for a zero history limit.")
