import time
from fastapi import FastAPI, Depends, HTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from app.schemas import (
    ConversationRequest,
    BeliefResponse,
    BeliefSpan,
)
from app.config import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
)


# ————————————————————————————————————————————
# TODO: dependency injection placeholders -
#          put the real model server / Triton gRPC client here
# ————————————————————————————————————————————
def get_belief_classifier():
    class _DummyClassifier:
        def predict(self, _messages):
            return [
                {
                    "facet": "Self-Efficacy",
                    "belief_id": "SEF_01",
                    "confidence": 0.42,
                    "evidence": "I think I can manage most problems by myself.",
                }
            ]

    return _DummyClassifier()


@app.post("/beliefs", response_model=BeliefResponse)
def beliefs(
    req: ConversationRequest,
    classifier=Depends(get_belief_classifier),
):
    start = time.perf_counter()

    try:
        predictions = classifier.predict(req.messages)

        spans = [
            BeliefSpan(
                facet=p["facet"],
                belief_id=p["belief_id"],
                confidence=p["confidence"],
                evidence=p["evidence"],
                timestamp=req.messages[-1].timestamp,
            )
            for p in predictions
        ]

        pillars = {"Coping": max(b.confidence for b in spans)}

        latency = int((time.perf_counter() - start) * 1000)
        if latency > settings.MAX_LATENCY_MS:
            app.logger.warning("Latency breach: %sms", latency)

        return BeliefResponse(
            user_id=req.user_id,
            conv_id=req.conv_id,
            pillars=pillars,
            beliefs=spans,
            model_version=settings.VERSION,
            latency_ms=latency,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Belief extraction failed: {exc}",
        ) from exc
