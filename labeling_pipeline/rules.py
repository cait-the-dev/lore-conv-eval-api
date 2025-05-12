import re

FACETS = {
    # Agency & Self-belief
    "Self-Efficacy": re.compile(
        r"\bI\s+(?:can|managed to|am able to|handle|cope)\b", re.I
    ),
    "Autonomy": re.compile(
        r"\bI\s+(?:decided|choose|chose|want(?:ed)? to|would like to)\b", re.I
    ),
    "Growth Mindset": re.compile(
        r"\b(?:learn(?:ing)?|get better|improv(?:e|ing)|practice more)\b", re.I
    ),
    "Locus of Control": re.compile(
        r"\b(?:up to me|depends on me|nothing I can do|beyond my control)\b", re.I
    ),
    # Emotional Resilience
    "Optimism": re.compile(
        r"\b(?:hopeful|optimistic|things will work out|look on the bright side)\b", re.I
    ),
    "Emotional Regulation": re.compile(
        r"\b(?:deep breaths?|calm(?:ed)? myself|ground(?:ing)? exercises?)\b", re.I
    ),
    "Self-Compassion": re.compile(
        r"\b(?:kind to myself|forgave myself|give myself grace)\b", re.I
    ),
    # Social & Belonging
    "Social Connectedness": re.compile(
        r"\b(?:family|friends|grand(?:son|daughter)|talk(?:ed)? to|felt lonely|reached out)\b",
        re.I,
    ),
    "Contribution": re.compile(
        r"\b(?:volunteer|help others|give back|support(?:ing)? someone)\b", re.I
    ),
    # Purpose & Meaning
    "Values Alignment": re.compile(
        r"\b(?:important to me|I value|matters to me)\b", re.I
    ),
    "Purpose": re.compile(r"\b(?:my goal|mission|purpose|aim(?:ing)? to)\b", re.I),
    # Safety & Health
    "Physical Safety": re.compile(
        r"\b(?:safe|safety|fall|seizure|worried about (?:my )?health)\b", re.I
    ),
    "Health Agency": re.compile(
        r"\b(?:take my meds|follow(?:ing)? the doctor|appointment|check[- ]?up)\b", re.I
    ),
    # Help Seeking
    "Support Utilization": re.compile(
        r"\b(?:therapist|counselor|ask(?:ed)? for help|support group)\b", re.I
    ),
}
