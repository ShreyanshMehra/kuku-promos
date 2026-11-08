"""
Pass 1 -- find candidate promos.

Division of labour (the central design decision):

  LLM   : judgement only. Picks WHICH moments make a promo and returns BEAT
          INDICES ("ep3:41"), never timestamps.
  CODE  : everything checkable. Resolves indices to times, snaps cut points so
          no dialogue line is chopped, repairs duration, re-checks the hook and
          cliffhanger after repair, and verifies every quoted line against the
          source. Hallucinated timestamps are structurally impossible because
          the model never writes one.

Named select_promos.py (not select.py) so it does not shadow the stdlib
`select` module.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from llm import LLMConfig, LLMError, call_llm
from parse import render_compact, seconds_to_ts

MIN_S = float(os.getenv("PROMO_MIN_S", "20"))
MAX_S = float(os.getenv("PROMO_MAX_S", "60"))
HOOK_WINDOW_S = float(os.getenv("PROMO_HOOK_WINDOW_S", "3.0"))
CLIFF_WINDOW_S = float(os.getenv("PROMO_CLIFF_WINDOW_S", "6.0"))
BATCH_EPISODES = int(os.getenv("PROMO_BATCH_EPISODES", "4"))
PER_EPISODE = int(os.getenv("PROMO_PER_EPISODE", "3"))
EPS = 1e-6

SYSTEM_PROMPT = f"""You are a short-form video editor for a vertical microdrama app. You read screenplays and choose promo clips that make a stranger scrolling Instagram, TikTok or YouTube Shorts stop, watch, and want the next episode.

A promo is ONE continuous clip of {MIN_S:.0f} to {MAX_S:.0f} seconds taken from a single episode. A good promo:
1. HOOK: within its first {HOOK_WINDOW_S:.0f} seconds, something makes a scrolling viewer stop: a slap, a reveal, a threat, a shocking line, a humiliation. An establishing shot or quiet setup is fine only if the hook lands inside those first {HOOK_WINDOW_S:.0f} seconds.
2. CLIFFHANGER: it cuts at an UNRESOLVED moment, at a question, threat, accusation or reveal, BEFORE the answer or consequence. Do not end on a resolved beat or a gentle fade-out.
3. STANDS ALONE: someone who has never seen the show must understand who is who and what is at stake from the clip alone. Never rely on events from other episodes. Prefer clips where the relationship and conflict are visible inside the window.

How to read the input:
- Each line is one beat: [epN:IDX] start-end TYPE: text. CAMERA, ACTION and SOUND FX are production notes; anything else is a character speaking. Beats overlap in time because camera and dialogue are parallel tracks.
- Refer to beats ONLY by the exact id shown, e.g. "ep3:41". Never invent ids and never write timestamps.
- Pick start_idx at the START of a spoken line and end_idx at the END of one, so no line is cut off mid-sentence.
- hook_idx and cliffhanger_idx must lie between start_idx and end_idx inclusive.
- hook_quote and cliffhanger_quote must be copied VERBATIM from the text of the beat you name. Do not paraphrase.
- Treat every episode independently. Episodes in one request are consecutive but a promo uses only one.

Return up to {PER_EPISODE} NON-OVERLAPPING candidates per episode, strongest first. Return fewer, or none, rather than padding with weak ones.

For each candidate also report:
- spoils_arc: true if it reveals a twist or turn that a first-time viewer would be spoiled on, or that only makes sense with earlier episodes.
- carries_premise: true if the clip alone communicates the show's core situation to a stranger.

Reply with ONLY this JSON, no prose, no code fences:
{{"promos": [{{"episode": <int>, "start_idx": "epN:IDX", "end_idx": "epN:IDX", "hook_idx": "epN:IDX", "hook_quote": "<verbatim>", "cliffhanger_idx": "epN:IDX", "cliffhanger_quote": "<verbatim>", "why": "<one line on why it works>", "standalone_notes": "<what a stranger might not understand, or 'none'>", "spoils_arc": <bool>, "carries_premise": <bool>}}]}}"""

PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "promos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "episode": {"type": "integer"},
                    "start_idx": {"type": "string"},
                    "end_idx": {"type": "string"},
                    "hook_idx": {"type": "string"},
                    "hook_quote": {"type": "string"},
                    "cliffhanger_idx": {"type": "string"},
                    "cliffhanger_quote": {"type": "string"},
                    "why": {"type": "string"},
                    "standalone_notes": {"type": "string"},
                    "spoils_arc": {"type": "boolean"},
                    "carries_premise": {"type": "boolean"},
                },
                "required": [
                    "episode", "start_idx", "end_idx", "hook_idx", "hook_quote",
                    "cliffhanger_idx", "cliffhanger_quote", "why",
                ],
            },
        }
    },
    "required": ["promos"],
}

_RE_REF = re.compile(r"^\s*ep\s*(\d+)\s*:\s*(\d+)\s*$", re.I)


# --------------------------------------------------------------------------
# reference + text helpers
# --------------------------------------------------------------------------

def parse_ref(ref: Any) -> Optional[Tuple[int, int]]:
    """'ep3:41' -> (3, 41). None if malformed."""
    if not isinstance(ref, str):
        return None
    m = _RE_REF.match(ref)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_QUOTE_CHARS = "\"'“”‘’"


def _clean_quote(q: str) -> str:
    """Strip wrapping quotes and a leading/trailing ellipsis the model may add. Nothing else."""
    q = _norm(q).strip(_QUOTE_CHARS).strip()
    q = re.sub(r"(\.\.\.|…)\s*$", "", q).strip()
    q = re.sub(r"^(\.\.\.|…)\s*", "", q).strip()
    return q


def quote_is_grounded(quote: str, beat_text: str) -> bool:
    """Exact substring, whitespace-normalised. Case-sensitive on purpose."""
    q = _clean_quote(quote)
    return len(q) >= 3 and q in _norm(beat_text)


# --------------------------------------------------------------------------
# boundary logic (pure code, no model)
# --------------------------------------------------------------------------

def snap_window(beats: List[dict], start_s: float, end_s: float) -> Tuple[float, float]:
    """
    Move the cut points so no DIALOGUE line is chopped.

    Camera and dialogue are parallel tracks, so cutting at a beat's end index can
    slice a line that is still being spoken. Fix:
      start: if a dialogue line straddles the start, pull the start back to its beginning
      end  : push the end out to the end of every dialogue line that began inside
    Iterates to a fixed point because moving one edge can pull in another line.
    """
    for _ in range(10):
        moved = False
        for b in beats:
            if not b["is_dialogue"]:
                continue
            if b["start_s"] < start_s - EPS and b["end_s"] > start_s + EPS:
                start_s = b["start_s"]
                moved = True
            if b["start_s"] < end_s - EPS and b["end_s"] > end_s + EPS:
                end_s = b["end_s"]
                moved = True
        if not moved:
            break
    return round(start_s, 2), round(end_s, 2)


def dialogue_straddles(beats: List[dict], start_s: float, end_s: float) -> List[int]:
    """Indices of dialogue beats that cross either cut. Empty list = clean."""
    bad = []
    for b in beats:
        if not b["is_dialogue"]:
            continue
        crosses_start = b["start_s"] < start_s - EPS and b["end_s"] > start_s + EPS
        crosses_end = b["start_s"] < end_s - EPS and b["end_s"] > end_s + EPS
        if crosses_start or crosses_end:
            bad.append(b["idx"])
    return bad


def beats_in_window(beats: List[dict], start_s: float, end_s: float) -> List[dict]:
    return [b for b in beats if b["start_s"] >= start_s - EPS and b["start_s"] < end_s - EPS]


def check_hook(beats: List[dict], hook_idx: int, start_s: float) -> bool:
    """The tagged hook beat must START within HOOK_WINDOW_S of the clip start."""
    b = beats[hook_idx]
    return (start_s - EPS) <= b["start_s"] <= (start_s + HOOK_WINDOW_S + EPS)


def check_cliffhanger(beats: List[dict], cliff_idx: int, start_s: float, end_s: float) -> bool:
    """The tagged cliffhanger beat must sit inside the clip's final CLIFF_WINDOW_S."""
    b = beats[cliff_idx]
    return (
        b["start_s"] >= start_s - EPS
        and b["start_s"] <= end_s + EPS
        and b["end_s"] >= end_s - CLIFF_WINDOW_S
    )


def repair_duration(
    beats: List[dict], hook_idx: int, start_s: float, end_s: float
) -> Tuple[float, float, List[str]]:
    """
    Bring duration into [MIN_S, MAX_S] by moving cut points to real beat boundaries.

    Too long : open on the hook instead of the setup before it.
    Too short: pull the start back, one beat boundary at a time.
    The cliffhanger end is never moved -- it is the point of the promo.
    The caller re-checks the hook afterwards because repair can lose it.
    """
    notes: List[str] = []
    dur = end_s - start_s

    if dur > MAX_S + EPS:
        hook_start = beats[hook_idx]["start_s"]
        if hook_start > start_s + EPS:
            start_s, end_s = snap_window(beats, hook_start, end_s)
            notes.append(f"too long ({dur:.1f}s): reopened on hook beat")

    dur = end_s - start_s
    if dur < MIN_S - EPS:
        original = dur
        starts = sorted({b["start_s"] for b in beats if b["start_s"] < start_s - EPS}, reverse=True)
        for cand in starts:
            start_s, end_s = snap_window(beats, cand, end_s)
            if end_s - start_s >= MIN_S - EPS:
                break
        if end_s - start_s >= MIN_S - EPS:
            notes.append(
                f"too short ({original:.1f}s): extended start backward to {seconds_to_ts(start_s)}"
            )

    return start_s, end_s, notes


# --------------------------------------------------------------------------
# validation of one model candidate
# --------------------------------------------------------------------------

def validate_candidate(raw: Dict[str, Any], ep: Dict[str, Any], seq: int) -> Dict[str, Any]:
    """
    Turn one model proposal into a verified candidate dict.

    Never raises. Every proposal is returned with status 'ok' or 'rejected' and a
    list of reasons, so rejected picks stay visible for the failure analysis
    instead of silently vanishing.
    """
    beats = ep["beats"]
    n = len(beats)
    ep_no = ep["episode_no"]
    cand: Dict[str, Any] = {
        "id": f"ep{ep_no}-c{seq}",
        "episode": ep_no,
        "status": "rejected",
        "reasons": [],
        "repairs": [],
        "why": _norm(str(raw.get("why", ""))),
        "standalone_notes": _norm(str(raw.get("standalone_notes", ""))),
        "spoils_arc": bool(raw.get("spoils_arc", False)),
        "carries_premise": bool(raw.get("carries_premise", False)),
        "hook_quote": _clean_quote(str(raw.get("hook_quote", ""))),
        "cliffhanger_quote": _clean_quote(str(raw.get("cliffhanger_quote", ""))),
        "model_refs": {
            k: raw.get(k) for k in ("start_idx", "end_idx", "hook_idx", "cliffhanger_idx")
        },
    }

    def reject(reason: str) -> Dict[str, Any]:
        cand["reasons"].append(reason)
        return cand

    # 1. resolve every reference and verify it belongs to THIS episode
    idx: Dict[str, int] = {}
    for key in ("start_idx", "end_idx", "hook_idx", "cliffhanger_idx"):
        ref = parse_ref(raw.get(key))
        if ref is None:
            return reject(f"{key} malformed: {raw.get(key)!r}")
        r_ep, r_idx = ref
        if r_ep != ep_no:
            return reject(f"{key} points at episode {r_ep}, expected {ep_no} (cross-episode index)")
        if not (0 <= r_idx < n):
            return reject(f"{key} index {r_idx} out of range 0..{n - 1}")
        idx[key] = r_idx

    if idx["end_idx"] < idx["start_idx"]:
        return reject("end_idx before start_idx")

    # 2. grounding: quoted lines must be verbatim from the beat they name
    hook_beat = beats[idx["hook_idx"]]
    cliff_beat = beats[idx["cliffhanger_idx"]]
    cand["hook_grounded"] = quote_is_grounded(cand["hook_quote"], hook_beat["text"])
    cand["cliffhanger_grounded"] = quote_is_grounded(cand["cliffhanger_quote"], cliff_beat["text"])
    if not cand["hook_grounded"]:
        cand["reasons"].append("hook_quote not found verbatim in hook beat (ungrounded)")
    if not cand["cliffhanger_grounded"]:
        cand["reasons"].append("cliffhanger_quote not found verbatim in cliffhanger beat (ungrounded)")

    # 3. code resolves times -- the model supplied indices only
    start_s = beats[idx["start_idx"]]["start_s"]
    end_s = beats[idx["end_idx"]]["end_s"]
    cand["model_window"] = [start_s, end_s]

    # 4. snap so no dialogue line is chopped at either edge
    s2, e2 = snap_window(beats, start_s, end_s)
    if (s2, e2) != (start_s, end_s):
        cand["repairs"].append(
            f"snapped to avoid chopping dialogue: {seconds_to_ts(start_s)}-{seconds_to_ts(end_s)} "
            f"-> {seconds_to_ts(s2)}-{seconds_to_ts(e2)}"
        )
    start_s, end_s = s2, e2

    # 5. duration repair
    start_s, end_s, notes = repair_duration(beats, idx["hook_idx"], start_s, end_s)
    cand["repairs"].extend(notes)

    # 5b. hook arrives too late: open on the hook instead. This is the editor's move, and
    # it is only taken when the clip stays >= MIN_S, so a short clip is still rejected
    # (below) rather than padded. The cliffhanger end is never touched. Without this,
    # clips whose model-chosen start was a few seconds early were thrown away outright,
    # which cost episode 1 -- the best recruitment episode -- every one of its candidates.
    hook_start = beats[idx["hook_idx"]]["start_s"]
    if hook_start > start_s + HOOK_WINDOW_S + EPS:
        new_s, new_e = snap_window(beats, hook_start, end_s)
        if new_e - new_s >= MIN_S - EPS:
            cand["repairs"].append(
                f"hook was {hook_start - start_s:.1f}s in: opened on the hook at {seconds_to_ts(new_s)}"
            )
            start_s, end_s = new_s, new_e

    end_s = min(end_s, ep["duration_s"])
    dur = round(end_s - start_s, 2)

    # 6. re-check hook + cliffhanger AFTER repair (repair can silently lose the hook)
    cand["hook_within_3s"] = check_hook(beats, idx["hook_idx"], start_s)
    cand["cliffhanger_at_end"] = check_cliffhanger(beats, idx["cliffhanger_idx"], start_s, end_s)
    cand["dialogue_straddles"] = dialogue_straddles(beats, start_s, end_s)

    if not (MIN_S - EPS <= dur <= MAX_S + EPS):
        cand["reasons"].append(f"duration {dur:.1f}s outside [{MIN_S:.0f},{MAX_S:.0f}] after repair")
    if not cand["hook_within_3s"]:
        cand["reasons"].append(
            f"hook beat starts {hook_beat['start_s'] - start_s:.1f}s into clip (limit {HOOK_WINDOW_S:.0f}s)"
        )
    if not cand["cliffhanger_at_end"]:
        cand["reasons"].append(f"cliffhanger beat not within last {CLIFF_WINDOW_S:.0f}s of clip")
    if cand["dialogue_straddles"]:
        cand["reasons"].append(f"dialogue still straddles a cut: beats {cand['dialogue_straddles']}")

    window = beats_in_window(beats, start_s, end_s)
    cand.update(
        {
            "start_s": start_s,
            "end_s": end_s,
            "start_ts": seconds_to_ts(start_s),
            "end_ts": seconds_to_ts(end_s),
            "duration_s": dur,
            "first_beat": window[0]["idx"] if window else None,
            "last_beat": window[-1]["idx"] if window else None,
            "hook_beat": idx["hook_idx"],
            "cliffhanger_beat": idx["cliffhanger_idx"],
            "segments": [{"episode": ep_no, "start_s": start_s, "end_s": end_s}],
        }
    )

    if not cand["reasons"]:
        cand["status"] = "ok"
    return cand


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _build_prompt(batch: List[Dict[str, Any]]) -> str:
    # Logline deliberately withheld: a promo must stand alone without it.
    parts = [render_compact(e, include_logline=False) for e in batch]
    ids = ", ".join(str(e["episode_no"]) for e in batch)
    return f"Find promo candidates in these {len(batch)} episode(s): {ids}.\n\n" + "\n\n".join(parts)


def _run_batch(batch: List[Dict[str, Any]], cfg: LLMConfig, model: Optional[str]) -> List[dict]:
    tag = "pass1.eps" + "-".join(str(e["episode_no"]) for e in batch)
    result = call_llm(SYSTEM_PROMPT, _build_prompt(batch), PASS1_SCHEMA, cfg=cfg, model=model, tag=tag,
                      salvage_key="promos")
    if isinstance(result, list):
        result = {"promos": result}
    promos = result.get("promos") if isinstance(result, dict) else None
    if not isinstance(promos, list):
        raise LLMError(f"pass1 response missing 'promos' list: {str(result)[:200]}")
    return promos


def select_candidates(
    episodes: List[Dict[str, Any]],
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    batch_size: int = BATCH_EPISODES,
    log=print,
) -> Dict[str, Any]:
    """
    Run Pass 1 over every episode. Returns
      {"candidates": [...ok...], "rejected": [...], "failures": [...]}
    A failed batch is retried one episode at a time, so one bad response never
    costs more than a batch, and never aborts the run.
    """
    cfg = cfg or LLMConfig()
    by_ep = {e["episode_no"]: e for e in episodes}
    ok: List[dict] = []
    rejected: List[dict] = []
    failures: List[dict] = []
    seq: Dict[int, int] = {}

    def ingest(proposals: List[dict], allowed: set) -> None:
        for p in proposals:
            if not isinstance(p, dict):
                continue
            ep_no = p.get("episode")
            if ep_no not in allowed:
                # fall back to the episode named inside the start ref
                ref = parse_ref(p.get("start_idx"))
                ep_no = ref[0] if ref and ref[0] in allowed else None
            if ep_no is None:
                rejected.append({"status": "rejected", "reasons": ["episode unresolved"], "model_refs": p})
                continue
            seq[ep_no] = seq.get(ep_no, 0) + 1
            cand = validate_candidate(p, by_ep[ep_no], seq[ep_no])
            (ok if cand["status"] == "ok" else rejected).append(cand)

    batches = [episodes[i : i + batch_size] for i in range(0, len(episodes), batch_size)]
    for bi, batch in enumerate(batches, 1):
        label = f"eps {batch[0]['episode_no']}-{batch[-1]['episode_no']}"
        allowed = {e["episode_no"] for e in batch}
        try:
            ingest(_run_batch(batch, cfg, model), allowed)
            log(f"  pass1 batch {bi}/{len(batches)} ({label}) ok")
        except LLMError as exc:
            log(f"  pass1 batch {bi}/{len(batches)} ({label}) FAILED: {str(exc)[:120]}; retrying per-episode")
            for e in batch:
                try:
                    ingest(_run_batch([e], cfg, model), {e["episode_no"]})
                except LLMError as exc2:
                    failures.append({"episode": e["episode_no"], "error": str(exc2)[:300]})
                    log(f"    episode {e['episode_no']} FAILED: {str(exc2)[:120]}")

    return {"candidates": ok, "rejected": rejected, "failures": failures}


if __name__ == "__main__":
    import argparse
    from parse import load_episodes

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--episodes", default=None, help="e.g. 1,2,3")
    args = ap.parse_args()
    eps = load_episodes(args.input)
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",")}
        eps = [e for e in eps if e["episode_no"] in want]
    res = select_candidates(eps)
    print(json.dumps({k: len(v) for k, v in res.items()}))
    for c in res["candidates"][:5]:
        print(c["id"], c["start_ts"], c["end_ts"], c["duration_s"], c["why"][:70])
