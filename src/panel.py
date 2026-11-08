"""
Simulated cold-viewer panel -- replaces the 1-5 "judge".

A promo's job is to hold the attention of a stranger, so instead of asking "how
good is this clip?" we ask three personas to watch it beat by beat and tell us
WHEN they would swipe away and whether they would tap through to episode 2.

  hook_survival = share of viewers still watching at 3s
  watch_through = mean fraction of the clip watched
  tap_next      = mean probability of tapping through to the next episode
  retention     = 0.35*hook_survival + 0.25*watch_through + 0.40*tap_next

Honest limit: this is an LLM role-playing viewers. It is a better-shaped judgement
than a 1-5 score, not measured audience behaviour. The blind human sample in
evaluate.py is the independent check, and the panel's own disagreement is reported
per clip as an uncertainty flag.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from llm import LLMConfig, LLMError, call_llm
from rank import clip_beat_list, clip_lines
from select_promos import _clean_quote, _norm

PANEL_BATCH = int(os.getenv("PROMO_PANEL_BATCH", "10"))
HOOK_S = 3.0
W_HOOK, W_WATCH, W_TAP = 0.35, 0.25, 0.40
MIN_VIEWERS = 2
N_PERSONAS = 3

SYSTEM = """You run a SIMULATED VIEWER PANEL for short vertical microdrama promos. Each clip is posted on Instagram, TikTok or YouTube Shorts to strangers who have NEVER seen the show. You see only the clip: screenplay lines with the seconds into the clip at which they occur.

Simulate these three different viewers watching each clip in a scrolling feed:
1. A short-drama fan (US, 25-45) who has never seen THIS show. Enjoys melodrama and cliffhangers, but is not loyal to a show they do not know: they swipe if the clip is confusing, repetitive, or the characters mean nothing to them yet. They only watch to the end if it works for a newcomer.
2. A casual scroller with very little patience. Swipes the moment nothing interesting happens.
3. A skeptic who dislikes melodrama and usually swipes. Shouting and tears alone never hold them. They CAN be won over, within the first few seconds, by a clear relationship, concrete stakes and a specific question the clip leaves open; if they get that, they watch on and may tap through.

For each viewer decide:
- swipe_at_s: the second (offset into the clip) at which they would swipe away, or null if they watch to the end. Be realistic: most people swipe within the first 3 seconds unless something truly stops them.
- tap_next: the probability from 0 to 1 that, after the clip ends, they tap through to watch the next episode. High only if the ending leaves them needing the answer. If the viewer swipes away before the end, tap_next is 0 (always include the field).
- reason: one short line.
- evidence: the moment that drove the decision. For a viewer who swipes, the line playing when they swipe; for a viewer who watches to the end, the line that most shaped their tap_next. Give at_s (the offset shown next to that line) and quote (words copied VERBATIM from that line's text, at least 3 words, not the speaker tag and not paraphrased). A reason with no real line behind it will be rejected.

Judge only what is in the clip. Reply with ONLY this JSON, no prose, no code fences:
{"views": [{"id": "<candidate id>", "viewers": [{"persona": 1, "swipe_at_s": <number or null>, "tap_next": <0-1>, "reason": "<short>", "evidence": {"at_s": <number>, "quote": "<verbatim words from a clip line>"}}, {"persona": 2, ...}, {"persona": 3, ...}]}]}
Return exactly one entry per CANDIDATE, using the id given, with all three personas."""

SCHEMA = {
    "type": "object",
    "properties": {
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "viewers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "persona": {"type": "integer"},
                                "swipe_at_s": {"type": "number"},
                                "tap_next": {"type": "number"},
                                "reason": {"type": "string"},
                                "evidence": {
                                    "type": "object",
                                    "properties": {"at_s": {"type": "number"}, "quote": {"type": "string"}},
                                },
                            },
                            "required": ["persona", "tap_next"],
                        },
                    },
                },
                "required": ["id", "viewers"],
            },
        }
    },
    "required": ["views"],
}


def _clean_viewer(v: Any, duration: float) -> Optional[Dict[str, Any]]:
    """Validate one viewer. None = unusable. swipe None = watched to the end."""
    if not isinstance(v, dict):
        return None

    sw = v.get("swipe_at_s")
    if sw is None or (isinstance(sw, str) and sw.strip().lower() in ("", "null", "none", "end")):
        swipe: Optional[float] = None
    else:
        try:
            swipe = float(sw)
        except (TypeError, ValueError):
            return None
        if swipe < 0:
            return None
        if swipe >= duration:
            swipe = None                       # past the end == watched it all

    # A viewer who swipes away early never reaches the end, so there is no tap
    # decision: models routinely OMIT tap_next for them, and that means 0. Rejecting
    # those viewers (as the first version did) silently removed exactly the clips the
    # panel liked least -- survivorship bias. A viewer who watched to the end but gave
    # no tap_next is genuinely incomplete and is discarded.
    raw_tap = v.get("tap_next")
    if raw_tap is None:
        if swipe is None:
            return None
        tap = 0.0
    else:
        try:
            tap = float(raw_tap)
        except (TypeError, ValueError):
            return None
        tap = max(0.0, min(1.0, tap))
    ev = v.get("evidence")
    if isinstance(ev, str):
        ev = {"quote": ev}
    evidence = None
    if isinstance(ev, dict) and str(ev.get("quote", "")).strip():
        try:
            at = float(ev["at_s"]) if ev.get("at_s") is not None else None
        except (TypeError, ValueError):
            at = None
        evidence = {"at_s": at, "quote": str(ev["quote"]).strip()[:200]}
    return {
        "persona": v.get("persona"),
        "swipe_at_s": None if swipe is None else round(swipe, 1),
        "tap_next": round(tap, 3),
        "reason": str(v.get("reason", ""))[:160],
        "evidence": evidence,
    }


def clip_beats(ep: Dict[str, Any], cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The beats the viewers were shown (same selection as rank.clip_lines), with clip-relative times."""
    s = cand["start_s"]
    return [
        {"start_off": b["start_s"] - s, "end_off": b["end_s"] - s, "text": b["text"]}
        for b, _carried in clip_beat_list(ep, cand)
    ]


def verify_evidence(viewer: Dict[str, Any], beats: List[Dict[str, Any]]) -> str:
    """
    Deterministic check that a viewer's cited evidence is real. Never touches a score.

      none                 the viewer cited nothing
      unsupported          the quote is not in any line of the clip (invented or paraphrased)
      quote_ok_timing_off  the quote is real, but that line is not playing near the swipe time
      supported            the quote is real and, for a viewer who swiped, the line is on screen
                           within a second of the swipe. (A viewer who watched to the end has no
                           swipe time, so for them only the quote can be verified.)
    """
    ev = viewer.get("evidence")
    if not ev:
        return "none"
    q = _clean_quote(ev["quote"])
    if len(q) < 3:
        return "unsupported"
    hits = [b for b in beats if q in _norm(b["text"])]
    if not hits:
        return "unsupported"
    swipe = viewer.get("swipe_at_s")
    if swipe is None:
        return "supported"
    if any(b["start_off"] - 1.0 <= swipe <= b["end_off"] + 1.0 for b in hits):
        return "supported"
    return "quote_ok_timing_off"


def score_panel(
    duration: float, viewers: List[Any], beats: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Pure maths. Returns None if fewer than MIN_VIEWERS usable viewers.
    If `beats` is given, each viewer's cited evidence is verified against them. That check
    only FLAGS: retention is computed identically with or without it.
    """
    clean = [c for c in (_clean_viewer(v, duration) for v in viewers) if c is not None]
    if len(clean) < MIN_VIEWERS or duration <= 0:
        return None
    ev_counts = {"supported": 0, "quote_ok_timing_off": 0, "unsupported": 0, "none": 0}
    for c in clean:
        c["evidence_status"] = verify_evidence(c, beats) if beats is not None else "unchecked"
        if c["evidence_status"] in ev_counts:
            ev_counts[c["evidence_status"]] += 1
    cited = ev_counts["supported"] + ev_counts["quote_ok_timing_off"] + ev_counts["unsupported"]
    support = round(ev_counts["supported"] / cited, 3) if cited else None
    # Flag for a human only when at least two viewers cited evidence and fewer than half of
    # it holds up. A flag, not a rejection: the clip may still be good.
    needs_review = bool(cited >= 2 and support is not None and support < 0.5)
    fracs = [1.0 if c["swipe_at_s"] is None else c["swipe_at_s"] / duration for c in clean]
    survives = [c["swipe_at_s"] is None or c["swipe_at_s"] >= HOOK_S for c in clean]
    hook_survival = sum(survives) / len(clean)
    watch_through = sum(fracs) / len(clean)
    tap_next = sum(c["tap_next"] for c in clean) / len(clean)
    spread = max(fracs) - min(fracs)
    # No "viewers disagree" flag: the personas differ BY DESIGN, so a spread threshold
    # fired on every clip in the first live run and carried no information. The raw
    # spread is kept; per-persona discrimination is reported in evaluate.py instead.
    return {
        "viewers": clean,
        "n_viewers": len(clean),
        "hook_survival": round(hook_survival, 3),
        "watch_through": round(watch_through, 3),
        "tap_next": round(tap_next, 3),
        "retention": round(W_HOOK * hook_survival + W_WATCH * watch_through + W_TAP * tap_next, 4),
        "swipe_spread": round(spread, 3),
        "evidence": ev_counts,
        "evidence_support": support,
        "needs_review": needs_review,
    }


def _prompt(batch: List[Dict[str, Any]], by_ep: Dict[int, Dict[str, Any]]) -> str:
    blocks = []
    for c in batch:
        lines = clip_lines(by_ep[c["episode"]], c)
        blocks.append(f"CANDIDATE {c['id']} (length {c['duration_s']:.0f}s)\n" + "\n".join(lines))
    return f"Run the panel on these {len(batch)} clip(s).\n\n" + "\n\n".join(blocks)


def run_panel(
    cands: List[Dict[str, Any]],
    by_ep: Dict[int, Dict[str, Any]],
    cfg: LLMConfig,
    model: Optional[str],
    log=print,
) -> None:
    """Attach `panel` (dict or None) to every candidate in place."""
    pending = list(cands)
    # Retry ladder: full batch, then thirds, then stragglers one at a time (a response with an
    # unclosed brace early on loses the whole batch, so single-clip calls are the last resort).
    for attempt, size in enumerate((PANEL_BATCH, max(1, PANEL_BATCH // 3), 1)):
        if not pending:
            break
        batches = [pending[i:i + size] for i in range(0, len(pending), size)]
        for bi, batch in enumerate(batches, 1):
            tag = f"panel.a{attempt}.b{bi}"
            try:
                res = call_llm(SYSTEM, _prompt(batch, by_ep), SCHEMA, cfg=cfg, model=model, tag=tag,
                               salvage_key="views")
            except LLMError as exc:
                log(f"  {tag} FAILED: {str(exc)[:110]}")
                continue
            views = res.get("views") if isinstance(res, dict) else res
            if not isinstance(views, list):
                continue
            by_id = {c["id"]: c for c in batch}
            for v in views:
                if not isinstance(v, dict) or v.get("id") not in by_id:
                    continue
                c = by_id[v["id"]]
                scored = score_panel(c["duration_s"], v.get("viewers") or [],
                                     clip_beats(by_ep[c["episode"]], c))
                if scored is not None:
                    c["panel"] = scored
            log(f"  {tag} ok")
        pending = [c for c in pending if c.get("panel") is None]
    for c in pending:
        c["panel"] = None
