"""
Conversation memory for the Production RAG system.
"""

from __future__ import annotations

from collections import deque


class ConversationMemory:
    """
    Stores recent conversation history.

    Keeps only the latest N messages.
    """

    def __init__(
        self,
        max_messages: int = 10,
    ) -> None:

        self.messages: deque[dict[str, str]] = deque(
            maxlen=max_messages
        )

    # ---------------------------------------------------------
    # Add Messages
    # ---------------------------------------------------------

    def add_user_message(
        self,
        message: str,
    ) -> None:

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

        self.messages.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

    # ---------------------------------------------------------
    # Read Memory
    # ---------------------------------------------------------

    def get_messages(self) -> list[dict[str, str]]:

        return list(self.messages)

    def formatted_history(self) -> str:
        """
        Return conversation history as text.
        """

        lines: list[str] = []

        for message in self.messages:

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

        self.messages.clear()

    def __len__(self) -> int:

        return len(self.messages)
