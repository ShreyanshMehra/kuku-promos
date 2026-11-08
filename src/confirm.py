"""
Confirm pass -- the LLM sees only the short windows the tension curve found, never
whole episodes.

For each window the model decides keep/drop, may trim or extend it using the
context beats shown, and supplies the hook, the cliffhanger, a one-line reason and
an archetype. It still speaks only in beat INDICES, and every kept window goes
through the existing validate_candidate, so boundary snapping, duration repair,
the 3-second hook rule, cliffhanger placement and verbatim-quote grounding all
apply exactly as before.

If the model is unavailable, windows fall back to curve-only candidates, labelled
as such, so the pipeline always produces output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm import LLMConfig, LLMError, call_llm
from parse import seconds_to_ts
from select_promos import (
    HOOK_WINDOW_S, MAX_S, MIN_S, _norm, parse_ref, validate_candidate,
)

CONFIRM_BATCH = 12
PAD_BEATS = 3
ARCHETYPES = [
    "humiliation", "betrayal_reveal", "confrontation", "identity_reveal",
    "romance_tension", "threat", "rescue", "other",
]

SYSTEM = f"""You are a short-form video editor for a vertical microdrama app. An algorithm has found candidate promo windows by looking for emotional intensity. Your job is to CONFIRM or DROP each one.

A promo is ONE continuous clip of {MIN_S:.0f} to {MAX_S:.0f} seconds posted to strangers who have never seen the show. A good one:
1. HOOK: within its first {HOOK_WINDOW_S:.0f} seconds something makes a scrolling viewer stop (a slap, a reveal, a threat, a shocking line).
2. CLIFFHANGER: it ends at an UNRESOLVED moment, a question, threat or reveal, BEFORE the answer.
3. STANDS ALONE: a stranger can follow who is who and what is at stake from the clip alone.

For each window:
- keep = false if it is just loud with no event, is a resolved scene, or a stranger could not follow it. Give a short drop_reason.
- If keeping, you MAY move start_idx / end_idx to other beats from the context shown to make the hook land first or to end on the open moment. Start at the start of a spoken line and end at the end of one.
- Refer to beats ONLY by the exact id shown (e.g. "ep3:41"). Never write timestamps.
- hook_quote and cliffhanger_quote must be copied VERBATIM from the text of the beat you name. hook_idx and cliffhanger_idx must lie between start_idx and end_idx.
- archetype is one of: {", ".join(ARCHETYPES)}.
- spoils_arc: true if it reveals a twist a first-time viewer would be spoiled on, or only makes sense with earlier episodes.
- carries_premise: true if the clip alone shows the show's core situation.

Reply with ONLY this JSON, no prose, no code fences:
{{"windows": [{{"id": "<window id>", "keep": <bool>, "drop_reason": "<if dropped>", "start_idx": "epN:IDX", "end_idx": "epN:IDX", "hook_idx": "epN:IDX", "hook_quote": "<verbatim>", "cliffhanger_idx": "epN:IDX", "cliffhanger_quote": "<verbatim>", "why": "<one line>", "standalone_notes": "<what a stranger might miss, or 'none'>", "archetype": "<one of the list>", "spoils_arc": <bool>, "carries_premise": <bool>}}]}}
Return exactly one entry per WINDOW, using the id given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "windows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}, "keep": {"type": "boolean"},
                    "drop_reason": {"type": "string"},
                    "start_idx": {"type": "string"}, "end_idx": {"type": "string"},
                    "hook_idx": {"type": "string"}, "hook_quote": {"type": "string"},
                    "cliffhanger_idx": {"type": "string"}, "cliffhanger_quote": {"type": "string"},
                    "why": {"type": "string"}, "standalone_notes": {"type": "string"},
                    "archetype": {"type": "string"},
                    "spoils_arc": {"type": "boolean"}, "carries_premise": {"type": "boolean"},
                },
                "required": ["id", "keep"],
            },
        }
    },
    "required": ["windows"],
}


def _beat_line(ep_no: int, b: Dict[str, Any]) -> str:
    tag = b["speaker"] if b["is_dialogue"] else b["type"]
    if b["is_dialogue"] and b["emotion"]:
        tag = f"{tag}({b['emotion']})"
    return (f"[ep{ep_no}:{b['idx']}] {seconds_to_ts(b['start_s'])}-"
            f"{seconds_to_ts(b['end_s'])} {tag}: {b['text']}")


def _window_block(w: Dict[str, Any], ep: Dict[str, Any]) -> str:
    n = ep["episode_no"]
    lo = max(0, w["start_idx"] - PAD_BEATS)
    hi = min(ep["n_beats"] - 1, w["end_idx"] + PAD_BEATS)
    head = (f"WINDOW {w['id']}  (proposed: ep{n}:{w['start_idx']} .. ep{n}:{w['end_idx']}; "
            f"beats before/after are context you may use)")
    return head + "\n" + "\n".join(_beat_line(n, ep["beats"][i]) for i in range(lo, hi + 1))


def _prompt(batch: List[Dict[str, Any]], by_ep: Dict[int, Dict[str, Any]]) -> str:
    blocks = [_window_block(w, by_ep[w["episode"]]) for w in batch]
    return f"Confirm or drop these {len(batch)} window(s).\n\n" + "\n\n".join(blocks)


def _ref(ep_no: int, idx: int) -> str:
    return f"ep{ep_no}:{idx}"


def _attach(cand: Dict[str, Any], w: Dict[str, Any], source: str, archetype: str) -> Dict[str, Any]:
    cand["id"] = w["id"]
    cand["window_id"] = w["id"]
    cand["curve"] = w["curve"]
    cand["curve_slice"] = w["curve_slice"]
    cand["curve_window"] = [w["start_s"], w["end_s"]]
    cand["archetype"] = archetype if archetype in ARCHETYPES else "other"
    cand["source"] = source
    if cand.get("start_s") is not None:
        cand["trimmed_by_model"] = (
            abs(cand["start_s"] - w["start_s"]) > 0.05 or abs(cand["end_s"] - w["end_s"]) > 0.05
        )
    return cand


def curve_only_candidate(w: Dict[str, Any], ep: Dict[str, Any]) -> Dict[str, Any]:
    """No model: hook and cliffhanger are the curve's own beats, quoted from the source."""
    n = ep["episode_no"]
    beats = ep["beats"]
    raw = {
        "episode": n,
        "start_idx": _ref(n, w["start_idx"]), "end_idx": _ref(n, w["end_idx"]),
        "hook_idx": _ref(n, w["hook_idx"]), "cliffhanger_idx": _ref(n, w["cliffhanger_idx"]),
        "hook_quote": _norm(beats[w["hook_idx"]]["text"])[:90],
        "cliffhanger_quote": _norm(beats[w["cliffhanger_idx"]]["text"])[:90],
        "why": "curve-only: opens on an intensity spike and ends still tense (not model-confirmed)",
        "standalone_notes": "not assessed (no model)",
        "spoils_arc": False, "carries_premise": False,
    }
    cand = validate_candidate(raw, ep, 0)
    return _attach(cand, w, "curve-only", "other")


def confirm_windows(
    windows: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    batch_size: int = CONFIRM_BATCH,
    log=print,
) -> Dict[str, Any]:
    cfg = cfg or LLMConfig()
    by_ep = {e["episode_no"]: e for e in episodes}
    ok: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def ingest(batch: List[Dict[str, Any]], resp: List[Any]) -> None:
        by_id = {str(r.get("id")): r for r in resp if isinstance(r, dict)}
        for w in batch:
            ep = by_ep[w["episode"]]
            r = by_id.get(w["id"])
            if r is None:
                dropped.append({"id": w["id"], "episode": w["episode"], "reason": "no response from model",
                                "curve": w["curve"], "by": "pipeline"})
                continue
            if not r.get("keep", True):
                dropped.append({"id": w["id"], "episode": w["episode"],
                                "reason": _norm(str(r.get("drop_reason", ""))) or "no reason given",
                                "curve": w["curve"], "by": "model"})
                continue
            n = w["episode"]
            raw = {
                "episode": n,
                "start_idx": r.get("start_idx") or _ref(n, w["start_idx"]),
                "end_idx": r.get("end_idx") or _ref(n, w["end_idx"]),
                "hook_idx": r.get("hook_idx") or _ref(n, w["hook_idx"]),
                "cliffhanger_idx": r.get("cliffhanger_idx") or _ref(n, w["cliffhanger_idx"]),
                "hook_quote": r.get("hook_quote", ""),
                "cliffhanger_quote": r.get("cliffhanger_quote", ""),
                "why": r.get("why", ""), "standalone_notes": r.get("standalone_notes", ""),
                "spoils_arc": r.get("spoils_arc", False), "carries_premise": r.get("carries_premise", False),
            }
            cand = _attach(validate_candidate(raw, ep, 0), w, "llm-confirmed", str(r.get("archetype", "other")))
            (ok if cand["status"] == "ok" else rejected).append(cand)

    def run(batch: List[Dict[str, Any]], tag: str) -> bool:
        try:
            res = call_llm(SYSTEM, _prompt(batch, by_ep), SCHEMA, cfg=cfg, model=model, tag=tag)
        except LLMError as exc:
            log(f"  confirm {tag} FAILED: {str(exc)[:110]}")
            return False
        items = res.get("windows") if isinstance(res, dict) else res
        if not isinstance(items, list):
            return False
        ingest(batch, items)
        return True

    for bi in range(0, len(windows), batch_size):
        batch = windows[bi:bi + batch_size]
        label = f"confirm.b{bi // batch_size + 1}"
        if run(batch, label):
            log(f"  {label} ok ({len(batch)} windows)")
            continue
        for si in range(0, len(batch), 4):                    # retry smaller
            sub = batch[si:si + 4]
            if run(sub, f"{label}.r{si // 4}"):
                continue
            for w in sub:                                     # last resort: curve-only
                cand = curve_only_candidate(w, by_ep[w["episode"]])
                failures.append({"id": w["id"], "episode": w["episode"], "error": "model unavailable; used curve-only"})
                (ok if cand["status"] == "ok" else rejected).append(cand)

    return {"candidates": ok, "rejected": rejected, "dropped": dropped, "failures": failures}


def all_curve_only(windows: List[Dict[str, Any]], episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Zero-API path: every window becomes a curve-only candidate."""
    by_ep = {e["episode_no"]: e for e in episodes}
    ok, rejected = [], []
    for w in windows:
        cand = curve_only_candidate(w, by_ep[w["episode"]])
        (ok if cand["status"] == "ok" else rejected).append(cand)
    return {"candidates": ok, "rejected": rejected, "dropped": [], "failures": []}
