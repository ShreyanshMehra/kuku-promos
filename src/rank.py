"""
Pass 2 -- score, de-duplicate and rank the validated candidates.

  LLM  : scores each clip 1-5 on hook / cliffhanger / standalone clarity, seeing
         ONLY the clip text (not Pass-1's reasoning, not the logline), so
         "standalone clarity" is judged the way a cold viewer would experience it.
  CODE : weights, the recruitment factor, de-duplication and the final ordering.
         The model never produces a rank.

Judge independence, stated honestly: the judge runs on a DIFFERENT model than
the selector (JUDGE_MODEL), but by default both are Gemini-family. That is
partial independence, not a clean-room judge. The blind human sample in
evaluate.py is the genuinely independent signal.

A promo's job is to recruit a stranger to episode 1, so ranking is not raw
drama. A twist from episode 47 spoils the arc and strands a cold viewer.
Final score = quality x recruitment_factor (see recruitment_factor()).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Set

from llm import LLMConfig, LLMError, call_llm
from parse import seconds_to_ts

JUDGE_BATCH = int(os.getenv("PROMO_JUDGE_BATCH", "10"))
EARLY_EP_MAX = int(os.getenv("PROMO_EARLY_EP_MAX", "15"))     # "early" = a viable entry point
EP_DECAY = float(os.getenv("PROMO_EP_DECAY", "15"))           # episode-position prior half-scale
OVERLAP_IOU = float(os.getenv("PROMO_DEDUP_IOU", "0.30"))     # same-episode time overlap
CONTENT_JACCARD = float(os.getenv("PROMO_DEDUP_JACCARD", "0.50"))

W_HOOK, W_CLIFF, W_STANDALONE = 0.40, 0.35, 0.25

JUDGE_SYSTEM = """You are a skeptical short-form-video editor judging promo clips for a vertical microdrama app. Each clip will be posted on its own to strangers who have NEVER seen the show. You see only the clip: screenplay lines with the number of seconds into the clip they occur.

Score each clip 1-5 on three things. Be harsh; most clips are not 5s.
- hook_strength: in the FIRST 3 SECONDS, is there something that makes a scrolling viewer stop (slap, reveal, threat, shocking line)? 1 = nothing happens, 5 = unmissable.
- cliffhanger_strength: does the clip END on an unresolved moment that makes the viewer need the next episode? 1 = resolved or fades out, 5 = cut mid-threat or mid-reveal.
- standalone_clarity: could someone who has never seen the show follow who these people are and what is at stake? 1 = needs lots of backstory, 5 = fully self-contained.

Also report:
- spoils_arc: true if the clip reveals a twist that a first-time viewer would be spoiled on, or only makes sense with earlier episodes.
- carries_premise: true if the clip alone communicates the show's core situation to a stranger.

Reply with ONLY this JSON, no prose, no code fences:
{"scores": [{"id": "<candidate id>", "hook_strength": <1-5>, "cliffhanger_strength": <1-5>, "standalone_clarity": <1-5>, "spoils_arc": <bool>, "carries_premise": <bool>, "note": "<one short line>"}]}
Return exactly one entry per CANDIDATE, using the id given."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "hook_strength": {"type": "integer"},
                    "cliffhanger_strength": {"type": "integer"},
                    "standalone_clarity": {"type": "integer"},
                    "spoils_arc": {"type": "boolean"},
                    "carries_premise": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["id", "hook_strength", "cliffhanger_strength", "standalone_clarity"],
            },
        }
    },
    "required": ["scores"],
}


# --------------------------------------------------------------------------
# clip rendering for the judge
# --------------------------------------------------------------------------

def clip_lines(ep: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """Clip text with offsets relative to the clip start, so the judge can see 'first 3s'."""
    out = []
    for b in ep["beats"]:
        if b["start_s"] < cand["start_s"] - 1e-6 or b["start_s"] >= cand["end_s"] - 1e-6:
            continue
        tag = b["speaker"] if b["is_dialogue"] else b["type"]
        if b["is_dialogue"] and b["emotion"]:
            tag = f"{tag}({b['emotion']})"
        out.append(f"+{b['start_s'] - cand['start_s']:04.1f}s {tag}: {b['text']}")
    return out


def _judge_prompt(batch: List[Dict[str, Any]], by_ep: Dict[int, Dict[str, Any]]) -> str:
    blocks = []
    for c in batch:
        lines = clip_lines(by_ep[c["episode"]], c)
        blocks.append(
            f"CANDIDATE {c['id']} (length {c['duration_s']:.0f}s)\n" + "\n".join(lines)
        )
    return f"Score these {len(batch)} clip(s).\n\n" + "\n\n".join(blocks)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _clamp_score(v: Any) -> Optional[int]:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def judge_candidates(
    cands: List[Dict[str, Any]],
    by_ep: Dict[int, Dict[str, Any]],
    cfg: LLMConfig,
    judge_model: Optional[str],
    log=print,
) -> None:
    """Attach `judge` to each candidate in place. Unscored ones are flagged, never invented."""
    pending = list(cands)
    for attempt, size in enumerate((JUDGE_BATCH, max(1, JUDGE_BATCH // 3))):
        if not pending:
            break
        batches = [pending[i : i + size] for i in range(0, len(pending), size)]
        for bi, batch in enumerate(batches, 1):
            tag = f"pass2.a{attempt}.b{bi}"
            try:
                result = call_llm(
                    JUDGE_SYSTEM, _judge_prompt(batch, by_ep), JUDGE_SCHEMA,
                    cfg=cfg, model=judge_model, tag=tag,
                )
            except LLMError as exc:
                log(f"  pass2 batch {bi}/{len(batches)} FAILED: {str(exc)[:120]}")
                continue
            scores = result.get("scores") if isinstance(result, dict) else result
            if not isinstance(scores, list):
                continue
            ids = {c["id"] for c in batch}
            for s in scores:
                if not isinstance(s, dict) or s.get("id") not in ids:
                    continue
                h = _clamp_score(s.get("hook_strength"))
                k = _clamp_score(s.get("cliffhanger_strength"))
                a = _clamp_score(s.get("standalone_clarity"))
                if None in (h, k, a):
                    continue
                target = next(c for c in batch if c["id"] == s["id"])
                target["judge"] = {
                    "hook_strength": h,
                    "cliffhanger_strength": k,
                    "standalone_clarity": a,
                    "spoils_arc": bool(s.get("spoils_arc", False)),
                    "carries_premise": bool(s.get("carries_premise", False)),
                    "note": str(s.get("note", ""))[:200],
                }
            log(f"  pass2 batch {bi}/{len(batches)} ok")
        pending = [c for c in pending if "judge" not in c]
    for c in pending:
        c["judge"] = None


def quality_score(j: Dict[str, Any]) -> float:
    """Weighted 1-5 sub-scores -> 0..1."""
    raw = (
        W_HOOK * j["hook_strength"]
        + W_CLIFF * j["cliffhanger_strength"]
        + W_STANDALONE * j["standalone_clarity"]
    )
    return round((raw - 1.0) / 4.0, 4)


def recruitment_factor(episode_no: int, spoils_arc: bool, carries_premise: bool) -> float:
    """
    How well this clip recruits a STRANGER, independent of how dramatic it is.

      episode-position prior : earlier episodes are better entry points
      spoiler penalty        : a clip that needs the arc (or spoils it) recruits worse
      premise bonus          : a clip that shows the show's situation recruits better

    A heuristic, not a measured effect: the weights are my assumptions about how
    short-drama funnels work (see doc). Only the ORDER of inputs is defended.
    """
    prior = math.exp(-(max(episode_no, 1) - 1) / EP_DECAY)
    factor = 0.55 + 0.45 * prior
    if spoils_arc:
        factor *= 0.70
    if carries_premise:
        factor *= 1.15
    return round(factor, 4)


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------

_STOP = set(
    "a an and are as at be but by for from had has have he her him his i if in is it its me my "
    "no not of on or our she so that the their them then there they this to was we were what when "
    "who will with you your do does did can could would should just now than too very".split()
)


def _content_words(text: str, names: Set[str]) -> Set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP and w not in names}


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    lo, hi = max(a["start_s"], b["start_s"]), min(a["end_s"], b["end_s"])
    inter = max(0.0, hi - lo)
    union = (a["end_s"] - a["start_s"]) + (b["end_s"] - b["start_s"]) - inter
    return inter / union if union > 0 else 0.0


def mark_duplicates(ranked: List[Dict[str, Any]], names: Set[str]) -> None:
    """
    Walk best-first; a candidate is a duplicate of an earlier, better one if
      (1) same episode and the windows overlap (IoU), or
      (2) different episode and their content words overlap (Jaccard).

    Character NAMES are excluded from (2): in a single show nearly every clip
    features the same two leads, so comparing characters would over-merge.
    """
    kept: List[Dict[str, Any]] = []
    for c in ranked:
        c["duplicate_of"] = None
        words = _content_words(
            " ".join([c.get("hook_quote", ""), c.get("cliffhanger_quote", ""), c.get("why", "")]),
            names,
        )
        c["_words"] = words
        for k in kept:
            if c["episode"] == k["episode"]:
                if _iou(c, k) >= OVERLAP_IOU:
                    c["duplicate_of"] = k["id"]
                    c["duplicate_reason"] = "overlaps a higher-ranked clip in the same episode"
                    break
            else:
                union = words | k["_words"]
                if union and len(words & k["_words"]) / len(union) >= CONTENT_JACCARD:
                    c["duplicate_of"] = k["id"]
                    c["duplicate_reason"] = "near-identical content to a higher-ranked clip"
                    break
        if c["duplicate_of"] is None:
            kept.append(c)
    for c in ranked:
        c.pop("_words", None)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def rank_candidates(
    candidates: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    cfg: Optional[LLMConfig] = None,
    judge_model: Optional[str] = None,
    log=print,
    scorer: str = "judge",
) -> Dict[str, Any]:
    """
    scorer: "panel" (simulated viewer panel -> retention), "judge" (legacy 1-5 scores),
            or "curve" (no model: the tension-curve score; used as the offline fallback).
    """
    cfg = cfg or LLMConfig()
    by_ep = {e["episode_no"]: e for e in episodes}
    names = {
        b["speaker"].lower()
        for e in episodes for b in e["beats"] if b["is_dialogue"] and b["speaker"]
    }
    # also drop individual name tokens (e.g. "lady", "alice")
    names |= {tok for n in list(names) for tok in n.split()}

    if scorer == "judge":
        judge_candidates(candidates, by_ep, cfg, judge_model, log=log)
    elif scorer == "panel":
        from panel import run_panel          # lazy: panel imports this module
        run_panel(candidates, by_ep, cfg, judge_model, log=log)
    elif scorer != "curve":
        raise ValueError(f"unknown scorer {scorer!r}")

    for c in candidates:
        usable = True
        if scorer == "judge":
            j = c.get("judge")
            usable = j is not None
            if usable:
                quality = quality_score(j)
                spoils, premise = j["spoils_arc"], j["carries_premise"]
                # Judge's own flags drive the factor; the selector's stay in the
                # output so selector/judge disagreement is visible.
                c["flag_disagreement"] = {
                    "spoils_arc": [c["spoils_arc"], j["spoils_arc"]],
                    "carries_premise": [c["carries_premise"], j["carries_premise"]],
                }
        elif scorer == "panel":
            p = c.get("panel")
            usable = p is not None
            if usable:
                quality = p["retention"]
                spoils, premise = c["spoils_arc"], c["carries_premise"]   # confirmer's flags
        else:
            quality = c["curve"]["score"]
            spoils, premise = c.get("spoils_arc", False), c.get("carries_premise", False)

        if not usable:
            c.update(quality=0.0, recruitment=0.0, final_score=0.0, judged=False)
            continue
        c["judged"] = True
        c["scorer"] = scorer
        c["quality"] = round(float(quality), 4)
        c["spoils_flag"] = bool(spoils)
        c["premise_flag"] = bool(premise)
        c["recruitment"] = recruitment_factor(c["episode"], c["spoils_flag"], c["premise_flag"])
        c["final_score"] = round(c["quality"] * c["recruitment"], 4)

    ordered = sorted(candidates, key=lambda c: (-c["final_score"], c["episode"], c["start_s"]))
    mark_duplicates(ordered, names)

    unique = [c for c in ordered if c["duplicate_of"] is None and c["judged"]]
    for i, c in enumerate(unique, 1):
        c["rank"] = i
        early = c["episode"] <= EARLY_EP_MAX
        c["feed"] = "post_first" if (early and not c["spoils_flag"]) else "deeper_cut"

    post_first = [c for c in unique if c["feed"] == "post_first"]
    deeper = sorted(
        (c for c in unique if c["feed"] == "deeper_cut"), key=lambda c: -c["quality"]
    )
    best_per_ep: Dict[int, Dict[str, Any]] = {}
    for c in unique:
        best_per_ep.setdefault(c["episode"], c)
    # One top pick per archetype: a diverse slate to A/B test, not five near-identical clips.
    slate: Dict[str, Dict[str, Any]] = {}
    for c in unique:
        slate.setdefault(c.get("archetype", "other"), c)

    return {
        "ranked": unique,
        "duplicates": [c for c in ordered if c["duplicate_of"] is not None],
        "unjudged": [c for c in ordered if not c["judged"]],
        "feeds": {
            "post_first": [c["id"] for c in post_first],
            "deeper_cuts": [c["id"] for c in deeper],
            "best_per_episode": [best_per_ep[k]["id"] for k in sorted(best_per_ep)],
            "slate_by_archetype": {a: c["id"] for a, c in slate.items()},
        },
        "scorer": scorer,
    }


TIER_MESSAGES = {
    "full": "All clips were model-confirmed and model-scored.",
    "partial": "Some clips fell back to curve-only selection or were not model-scored. "
               "Re-run with a working API key / fresh quota to improve quality.",
    "degraded": "Most model calls failed. Most clips are curve-only fallbacks; treat the ranking as unreliable.",
    "offline": "Zero-API run. Output is heuristic (tension curve only), not model-judged.",
    "mock": "Mock provider: model-stage output is a deterministic placeholder, not a quality result.",
    "empty": "No promos survived selection and validation.",
}


def confidence_tier(ranked: List[Dict[str, Any]], mock: bool = False, offline: bool = False) -> Dict[str, Any]:
    """
    How much should a reviewer trust this run? Computed from what actually happened to
    each ranked clip, not from the flags the run was started with.

      full     every clip model-confirmed (or model-selected) AND model-scored
      partial  at least half model-confirmed, but some curve-only or unscored
      degraded most clips are curve-only fallbacks
      offline  zero-API run
      mock     placeholder model output
    """
    total = len(ranked)
    src = {}
    for c in ranked:
        src[c.get("source", "llm-selected")] = src.get(c.get("source", "llm-selected"), 0) + 1
    model_sel = src.get("llm-confirmed", 0) + src.get("llm-selected", 0)
    scored = sum(1 for c in ranked if c.get("panel") or c.get("judge"))
    counts = {"clips": total, "model_selected": model_sel, "curve_only": src.get("curve-only", 0),
              "model_scored": scored}

    if mock:
        tier = "mock"
    elif total == 0:
        tier = "empty"
    elif offline or src.get("curve-only", 0) == total:
        tier = "offline" if offline else "degraded"
    elif model_sel == total and scored == total:
        tier = "full"
    elif model_sel >= total * 0.5:
        tier = "partial"
    else:
        tier = "degraded"
    return {"tier": tier, "message": TIER_MESSAGES[tier], "counts": counts}


def segments_str(c: Dict[str, Any]) -> str:
    return "; ".join(
        f"ep{s['episode']} {seconds_to_ts(s['start_s'])}-{seconds_to_ts(s['end_s'])}"
        for s in c["segments"]
    )
