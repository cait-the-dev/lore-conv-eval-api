import pytest

from app.services import belief_mapper as bm


@pytest.mark.parametrize(
    "logits,expected_facets",
    [
        ({"Self-Efficacy": 0.4, "Growth Mindset": 0.2}, ["Self-Efficacy"]),
        (
            {"Social Worth": 0.5, "Self-Compassion": 0.38},
            [
                "Social Worth",
                "Self-Compassion",
            ],
        ),
        ({"Coping Ability": 0.1}, []),
    ],
)
def test_map_logits(logits, expected_facets):
    beliefs = bm.map_logits(logits, messages=[])
    facets = [b["facet"] for b in beliefs]
    assert facets == expected_facets


def test_rollup():
    logits = {"Self-Efficacy": 0.5, "Coping Ability": 0.8}
    beliefs = bm.map_logits(logits, messages=[])
    pillars = bm.rollup(beliefs)
    # both facets map to same pillar "Coping"
    assert "Coping" in pillars
    assert 0.5 < pillars["Coping"] <= 0.8
