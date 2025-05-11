from datetime import datetime
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]  # TODO: make this dynamic
    text: str
    timestamp: datetime


class ConversationRequest(BaseModel):
    user_id: str
    conv_id: str
    messages: List[ConversationMessage]


class BeliefSpan(BaseModel):
    facet: str
    belief_id: str
    confidence: float
    evidence: str
    timestamp: datetime


class BeliefResponse(BaseModel):
    user_id: str
    conv_id: str
    pillars: Dict[str, float]
    beliefs: List[BeliefSpan]
    model_version: str
    latency_ms: Optional[int] = None
