"""
Emotion-word intensity lexicon, scored ONCE per show.

The screenplays tag nearly every dialogue line with a delivery note
("Cold, Commanding", "Desperate, Trembling"). 99% of lines in the real pack carry
one, so this is the strongest cheap signal for building a tension curve.

One LLM call rates each distinct word 0-3 ("would this delivery make a scrolling
stranger stop?"). The result is saved to lexicon.json so a human can audit and
edit it. If the model is unavailable, a small hand lexicon is used and unknown
words default to a neutral score, so the pipeline never depends on the API here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from llm import LLMConfig, LLMError, call_llm

NEUTRAL = 1.0

# Fallback. 3 = stops a scroll, 2 = charged, 1 = neutral/unknown, 0 = calming.
FALLBACK: Dict[str, float] = {}
for _w in (
    "screaming shouting furious venomous threatening enraged sobbing hysterical savage "
    "murderous seething livid agonized anguished horrified shrieking roaring wrathful "
    "menacing vicious explosive"
).split():
    FALLBACK[_w] = 3.0
for _w in (
    "desperate defiant cold cruel mocking pleading trembling urgent accusing scornful "
    "breathless choked weeping tearful fierce steely smug demanding sharp gloating "
    "outraged shocked panicked stunned bitter contemptuous commanding ominous grave "
    "devastated crying alarmed"
).split():
    FALLBACK[_w] = 2.0
for _w in (
    "calm gentle soft reassuring tender warm relieved serene peaceful sincere quiet "
    "affectionate loving content cheerful lighthearted amused playful"
).split():
    FALLBACK[_w] = 0.0

SYSTEM = """You rate how attention-grabbing an acting direction is for a vertical microdrama clip shown to a stranger who is scrolling past.

Rate each delivery word from 0 to 3:
0 = calming or low-energy (calm, gentle, tender)
1 = neutral or ordinary (firm, curious, thoughtful)
2 = charged emotion (desperate, cold, mocking, trembling)
3 = scroll-stopping intensity (screaming, venomous, threatening, sobbing)

Rate the word on its own. Reply with ONLY this JSON, no prose, no code fences:
{"scores": {"<word>": <0-3>, ...}}
Include every word given, spelled exactly as given."""

SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "object"}},
    "required": ["scores"],
}

_WORD = re.compile(r"[a-z][a-z'\-]*")


def emotion_words(emotion: Optional[str]) -> List[str]:
    if not emotion:
        return []
    return _WORD.findall(emotion.lower())


def collect_words(episodes: List[Dict[str, Any]]) -> Counter:
    words: Counter = Counter()
    for ep in episodes:
        for b in ep["beats"]:
            for w in emotion_words(b.get("emotion")):
                words[w] += 1
    return words


def build_lexicon(
    episodes: List[Dict[str, Any]],
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    use_llm: bool = True,
    log=print,
) -> Dict[str, Any]:
    """
    Returns {"scores": {word: 0..3}, "source": "llm"|"fallback", "n_words": int,
             "n_llm_scored": int}.
    Words the model skips or mis-scores fall back to the hand lexicon, then NEUTRAL.
    """
    words = collect_words(episodes)
    scores: Dict[str, float] = {w: FALLBACK.get(w, NEUTRAL) for w in words}
    source = "fallback"
    n_llm = 0

    if use_llm and words:
        listing = ", ".join(sorted(words))
        try:
            result = call_llm(
                SYSTEM, f"RATE_WORDS: {listing}", SCHEMA,
                cfg=cfg, model=model, tag="lexicon",
            )
            raw = result.get("scores") if isinstance(result, dict) else None
            if isinstance(raw, dict):
                for w, v in raw.items():
                    key = str(w).strip().lower()
                    try:
                        val = float(v)
                    except (TypeError, ValueError):
                        continue
                    if key in scores and 0.0 <= val <= 3.0:
                        scores[key] = val
                        n_llm += 1
                if n_llm:
                    source = "llm"
            log(f"  lexicon: {n_llm}/{len(words)} words scored by model")
        except LLMError as exc:
            log(f"  lexicon: model unavailable ({str(exc)[:100]}); using hand lexicon")

    return {"scores": dict(sorted(scores.items())), "source": source,
            "n_words": len(words), "n_llm_scored": n_llm}


def emotion_score(emotion: Optional[str], lex: Dict[str, float]) -> float:
    """Max score over the words of an emotion string. Untagged lines are neutral."""
    ws = emotion_words(emotion)
    if not ws:
        return NEUTRAL
    return max(lex.get(w, FALLBACK.get(w, NEUTRAL)) for w in ws)
