from typing import Dict, List

from app.schemas import ConversationRequest, ConversationMessage

BOT_SCREEN_NAME = "StoryBot"


def _role(screen_name: str) -> str:
    """Return 'assistant' if the speaker is StoryBot else 'user'."""
    return "assistant" if screen_name == BOT_SCREEN_NAME else "user"


def from_dataset(entry: Dict) -> ConversationRequest:
    """Convert a single dataset entry."""
    user_id = str(entry["ref_user_id"])
    conv_id = str(entry["ref_conversation_id"])

    messages: List[ConversationMessage] = []
    for m in entry["messages_list"]:
        messages.append(
            ConversationMessage(
                role=_role(m["screen_name"]),
                text=m["message"],
                timestamp=m["transaction_datetime_utc"],
            )
        )

    return ConversationRequest(user_id=user_id, conv_id=conv_id, messages=messages)
