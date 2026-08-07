"""
Thread-safe conversation memory and isolated session storage.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class SessionCapacityError(RuntimeError):
    """
    Raised when every session slot is actively in use.
    """


class ConversationMemory:
    """
    Store recent conversation history.

    Keeps only the latest N messages and protects all
    reads and writes with a reentrant lock.
    """

    def __init__(
        self,
        max_messages: int = 10,
    ) -> None:
        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.messages: deque[dict[str, str]] = deque(
            maxlen=max_messages
        )

        self._lock = threading.RLock()

    # ---------------------------------------------------------
    # Add Messages
    # ---------------------------------------------------------

    def add_user_message(
        self,
        message: str,
    ) -> None:
        with self._lock:
            self.messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

    def add_ai_message(
        self,
        message: str,
    ) -> None:
        with self._lock:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    def add_exchange(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add one complete conversation exchange atomically.
        """

        with self._lock:
            self.messages.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )

            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                }
            )

    def latest_exchange_matches(
        self,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        """
        Return True when the latest complete exchange matches.

        Comparison ignores case and repeated whitespace so
        equivalent cached questions do not consume memory twice.
        """

        def normalize(value: str) -> str:
            return " ".join(
                value.casefold().split()
            )

        with self._lock:
            if len(self.messages) < 2:
                return False

            user_entry = self.messages[-2]
            assistant_entry = self.messages[-1]

            if (
                user_entry["role"] != "user"
                or assistant_entry["role"] != "assistant"
            ):
                return False

            return (
                normalize(user_entry["content"])
                == normalize(user_message)
                and normalize(assistant_entry["content"])
                == normalize(assistant_message)
            )

    # ---------------------------------------------------------
    # Read Memory
    # ---------------------------------------------------------

    def get_messages(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                message.copy()
                for message in self.messages
            ]

    def formatted_history(self) -> str:
        """
        Return conversation history as text.
        """

        with self._lock:
            messages = list(self.messages)

        lines: list[str] = []

        for message in messages:
            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(lines)

    # ---------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self.messages.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self.messages)


@dataclass
class _SessionEntry:
    """
    Internal session state.
    """

    memory: ConversationMemory
    lock: threading.Lock = field(
        default_factory=threading.Lock
    )
    last_access: float = field(
        default_factory=time.monotonic
    )
    active_users: int = 0


class SessionMemoryStore:
    """
    Bounded, expiring store of isolated conversation memories.

    Requests using the same session are serialized. Requests
    using different sessions may run concurrently.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_sessions: int = 1000,
        max_messages: int = 10,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        if max_sessions <= 0:
            raise ValueError(
                "max_sessions must be greater than zero."
            )

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_messages = max_messages

        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _remove_expired_locked(
        self,
        now: float,
    ) -> None:
        expired_session_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if (
                entry.active_users == 0
                and now - entry.last_access
                > self.ttl_seconds
            )
        ]

        for session_id in expired_session_ids:
            del self._sessions[session_id]

    def _evict_oldest_inactive_locked(self) -> None:
        inactive_sessions = [
            (session_id, entry)
            for session_id, entry in self._sessions.items()
            if entry.active_users == 0
        ]

        if not inactive_sessions:
            raise SessionCapacityError(
                "Every conversation session is currently active."
            )

        oldest_session_id, _ = min(
            inactive_sessions,
            key=lambda item: item[1].last_access,
        )

        del self._sessions[oldest_session_id]

    def _reserve(
        self,
        session_id: str,
    ) -> _SessionEntry:
        normalized_session_id = session_id.strip()

        if not normalized_session_id:
            raise ValueError(
                "session_id must not be empty."
            )

        now = time.monotonic()

        with self._lock:
            self._remove_expired_locked(now)

            entry = self._sessions.get(
                normalized_session_id
            )

            if entry is None:
                if (
                    len(self._sessions)
                    >= self.max_sessions
                ):
                    self._evict_oldest_inactive_locked()

                entry = _SessionEntry(
                    memory=ConversationMemory(
                        max_messages=self.max_messages
                    )
                )

                self._sessions[
                    normalized_session_id
                ] = entry

            entry.active_users += 1
            entry.last_access = now

            return entry

    def _release(
        self,
        entry: _SessionEntry,
    ) -> None:
        with self._lock:
            entry.active_users -= 1
            entry.last_access = time.monotonic()

    # ---------------------------------------------------------
    # Session Access
    # ---------------------------------------------------------

    @contextmanager
    def session(
        self,
        session_id: str,
    ) -> Iterator[ConversationMemory]:
        """
        Yield isolated memory while locking that session.
        """

        entry = self._reserve(session_id)

        try:
            with entry.lock:
                yield entry.memory
        finally:
            self._release(entry)

    # ---------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Remove every inactive session.
        """

        with self._lock:
            inactive_session_ids = [
                session_id
                for session_id, entry
                in self._sessions.items()
                if entry.active_users == 0
            ]

            for session_id in inactive_session_ids:
                del self._sessions[session_id]

    def size(self) -> int:
        """
        Return the number of live sessions.
        """

        now = time.monotonic()

        with self._lock:
            self._remove_expired_locked(now)
            return len(self._sessions)
