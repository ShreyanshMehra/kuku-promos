"""
Deterministic offline stand-in for a real model.

Purpose: the prototype must always run. If a key is missing, a quota is
exhausted or the venue has no network during the interview, `LLM_PROVIDER=mock`
still exercises every line of the pipeline end to end.

It is NOT a quality baseline and makes no judgement. It reads the beat lines
out of the prompt and returns schema-valid, index-valid, grounded-quote output
so the validator, repair, ranking, report and evaluation stages all have real
data to work on. Outputs are labelled so they can never be mistaken for model
picks -- `why` says so explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

_RE_LINE = re.compile(
    r"^\[ep(?P<ep>\d+):(?P<idx>\d+)\]\s+(?P<start>\d+:\d+\.\d+)-(?P<end>\d+:\d+\.\d+)\s+(?P<tag>[^:]+):\s*(?P<text>.*)$"
)


def _seconds(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def _stable_unit(key: str) -> float:
    """Deterministic pseudo-random in [0,1) from a string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _parse_beats(user: str) -> Dict[int, List[dict]]:
    episodes: Dict[int, List[dict]] = {}
    for line in user.splitlines():
        m = _RE_LINE.match(line.strip())
        if not m:
            continue
        ep = int(m.group("ep"))
        episodes.setdefault(ep, []).append(
            {
                "idx": int(m.group("idx")),
                "start_s": _seconds(m.group("start")),
                "end_s": _seconds(m.group("end")),
                "tag": m.group("tag").strip(),
                "text": m.group("text").strip(),
            }
        )
    return episodes


def _first_sentence(text: str, limit: int = 90) -> str:
    """A quote that is guaranteed to be an exact substring of the beat text."""
    snippet = text.strip()
    cut = snippet.find(". ")
    if 15 < cut < limit:
        snippet = snippet[:cut]
    return snippet[:limit].strip()


def _pick_window(beats: List[dict], target_s: float = 38.0) -> Optional[Tuple[int, int]]:
    """
    Walk back from the episode end to cover roughly `target_s`.

    This mirrors the naive 'last N seconds' editor heuristic on purpose: the
    mock should look like an unremarkable default, not like a good model.
    """
    if not beats:
        return None
    end_beat = beats[-1]
    end_s = end_beat["end_s"]
    start_i = 0
    for i in range(len(beats) - 1, -1, -1):
        if end_s - beats[i]["start_s"] >= target_s:
            start_i = i
            break
    return start_i, len(beats) - 1


def _pass1(episodes: Dict[int, List[dict]]) -> dict:
    promos = []
    for ep in sorted(episodes):
        beats = sorted(episodes[ep], key=lambda b: b["idx"])
        window = _pick_window(beats)
        if not window:
            continue
        start_i, end_i = window
        hook = beats[start_i]
        cliff = beats[end_i]
        # Prefer a dialogue beat near the start for the hook quote when present.
        for b in beats[start_i : min(start_i + 4, end_i + 1)]:
            if b["tag"] not in {"CAMERA", "ACTION", "SOUND FX"}:
                hook = b
                break
        promos.append(
            {
                "episode": ep,
                "start_idx": f"ep{ep}:{beats[start_i]['idx']}",
                "end_idx": f"ep{ep}:{beats[end_i]['idx']}",
                "hook_idx": f"ep{ep}:{hook['idx']}",
                "hook_quote": _first_sentence(hook["text"]),
                "cliffhanger_idx": f"ep{ep}:{cliff['idx']}",
                "cliffhanger_quote": _first_sentence(cliff["text"]),
                "why": "MOCK PROVIDER -- not a model judgement. Fixed last-~38s window.",
                "standalone_notes": "Not assessed by the mock provider.",
                "spoils_arc": ep > 10,
                "carries_premise": ep <= 10,
            }
        )
    return {"promos": promos}


def _pass2(user: str) -> dict:
    """Score candidates listed in the prompt as 'CANDIDATE <id>'."""
    ids = re.findall(r"^CANDIDATE\s+(\S+)", user, re.M)
    scores = []
    for cid in ids:
        scores.append(
            {
                "id": cid,
                "hook_strength": 1 + int(_stable_unit(cid + "h") * 5),
                "cliffhanger_strength": 1 + int(_stable_unit(cid + "c") * 5),
                "standalone_clarity": 1 + int(_stable_unit(cid + "s") * 5),
                "spoils_arc": _stable_unit(cid + "x") > 0.7,
                "carries_premise": _stable_unit(cid + "p") > 0.5,
                "note": "MOCK PROVIDER -- deterministic placeholder, not a judgement.",
            }
        )
    return {"scores": scores}


_RE_WINDOW = re.compile(
    r"^WINDOW\s+(?P<id>\S+)\s+\(proposed:\s+ep(?P<ep>\d+):(?P<a>\d+)\s+\.\.\s+ep\d+:(?P<b>\d+);"
)
_PRODUCTION = {"CAMERA", "ACTION", "SOUND FX"}
_ARCHETYPES = ["humiliation", "betrayal_reveal", "confrontation", "identity_reveal",
               "romance_tension", "threat", "rescue", "other"]


def _lexicon(user: str) -> dict:
    from lexicon import FALLBACK, NEUTRAL

    words = [w.strip() for w in user.split("RATE_WORDS:", 1)[1].split(",") if w.strip()]
    return {"scores": {w: FALLBACK.get(w, NEUTRAL) for w in words}}


def _confirm(user: str) -> dict:
    """Keep ~4 of 5 windows; hook = first line spoken, cliffhanger = last, both quoted verbatim."""
    blocks: Dict[str, dict] = {}
    cur = None
    for line in user.splitlines():
        m = _RE_WINDOW.match(line.strip())
        if m:
            cur = m.group("id")
            blocks[cur] = {"ep": int(m.group("ep")), "a": int(m.group("a")),
                           "b": int(m.group("b")), "beats": []}
            continue
        bm = _RE_LINE.match(line.strip())
        if bm and cur:
            blocks[cur]["beats"].append(
                {"idx": int(bm.group("idx")), "tag": bm.group("tag").strip(), "text": bm.group("text").strip()}
            )
    out = []
    for wid, blk in blocks.items():
        if _stable_unit(wid + "keep") < 0.2:
            out.append({"id": wid, "keep": False, "drop_reason": "MOCK PROVIDER: arbitrary drop"})
            continue
        inside = [b for b in blk["beats"] if blk["a"] <= b["idx"] <= blk["b"]]
        dialog = [b for b in inside if b["tag"].split("(")[0].strip() not in _PRODUCTION]
        if not dialog:
            out.append({"id": wid, "keep": False, "drop_reason": "MOCK PROVIDER: no dialogue"})
            continue
        hook, cliff, ep = dialog[0], dialog[-1], blk["ep"]
        out.append({
            "id": wid, "keep": True,
            "start_idx": f"ep{ep}:{blk['a']}", "end_idx": f"ep{ep}:{blk['b']}",
            "hook_idx": f"ep{ep}:{hook['idx']}", "hook_quote": _first_sentence(hook["text"]),
            "cliffhanger_idx": f"ep{ep}:{cliff['idx']}", "cliffhanger_quote": _first_sentence(cliff["text"]),
            "why": "MOCK PROVIDER -- not a model judgement.",
            "standalone_notes": "Not assessed by the mock provider.",
            "archetype": _ARCHETYPES[int(_stable_unit(wid + "a") * len(_ARCHETYPES))],
            "spoils_arc": _stable_unit(wid + "x") > 0.7,
            "carries_premise": _stable_unit(wid + "p") > 0.5,
        })
    return {"windows": out}


_RE_CLIP_LINE = re.compile(r"^\+(?P<off>\d+\.\d)s\s+(?P<tag>[^:]+):\s*(?P<text>.*)$")


def _clip_lines(user: str, cid: str) -> list:
    """(offset, text) for each line of one CANDIDATE block in a panel prompt."""
    out, inside = [], False
    for line in user.splitlines():
        if line.startswith("CANDIDATE "):
            inside = line.split()[1] == cid
            continue
        if inside:
            m = _RE_CLIP_LINE.match(line.strip())
            if m:
                out.append((float(m.group("off")), m.group("text").strip()))
    return out


def _panel(user: str) -> dict:
    views = []
    for m in re.finditer(r"^CANDIDATE\s+(\S+)\s+\(length\s+(\d+)s\)", user, re.M):
        cid, length = m.group(1), int(m.group(2))
        lines = _clip_lines(user, cid)
        viewers = []
        for persona in (1, 2, 3):
            u = _stable_unit(f"{cid}s{persona}")
            swipe = None if u > 0.55 else round(1 + u * length, 1)
            when = swipe if swipe is not None else float(length)
            quote, at = "", when
            if lines:
                off, text = min(lines, key=lambda lt: abs(lt[0] - when))
                quote, at = _first_sentence(text, 60), off
            # a deliberately fabricated quote on some clips, so flagging is exercised end to end
            if persona == 3 and _stable_unit(cid + "fab") < 0.25:
                quote = "an invented line that nobody says"
            viewers.append({
                "persona": persona,
                "swipe_at_s": swipe,
                "tap_next": round(_stable_unit(f"{cid}t{persona}"), 2),
                "reason": "MOCK PROVIDER -- deterministic placeholder.",
                "evidence": {"at_s": at, "quote": quote},
            })
        views.append({"id": cid, "viewers": viewers})
    return {"views": views}


def mock_response(system: str, user: str, schema: Optional[dict]) -> str:
    """Return a JSON string matching whichever stage is calling."""
    if "SIMULATED VIEWER PANEL" in system:
        return json.dumps(_panel(user), ensure_ascii=False)
    if user.startswith("RATE_WORDS:"):
        return json.dumps(_lexicon(user), ensure_ascii=False)
    if re.search(r"^WINDOW\s+\S+", user, re.M):
        return json.dumps(_confirm(user), ensure_ascii=False)

    if re.search(r"^CANDIDATE\s+\S+", user, re.M):
        return json.dumps(_pass2(user), ensure_ascii=False)

    episodes = _parse_beats(user)
    if episodes:
        return json.dumps(_pass1(episodes), ensure_ascii=False)

    # Smoke test / unknown prompt shape.
    return json.dumps({"ok": True, "note": "mock"}, ensure_ascii=False)
