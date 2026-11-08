"""
Tension curve -> candidate promo windows. Pure code, zero API calls.

Idea: a promo is a stretch that opens on a spike and ends still tense. The script
already annotates intensity (emotion tags on ~99% of dialogue lines, plus actions
and sound cues), so we can search for such stretches directly instead of paying a
model to read every episode.

  1. every beat gets an intensity (dialogue: lexicon score of its delivery words,
     +bonus for ! and ?; action/camera/sound: a small keyword bonus)
  2. beats are laid onto a 0.5s grid (parallel tracks combine by max, not sum)
  3. every legal window is scored on  hook (first 3s) / end tension (last 6s,
     open-ending bonus, resolved-ending penalty) / sustain (no long dead valley)
  4. greedy non-max suppression keeps the best K non-overlapping windows per episode

Honest limit: the curve measures *annotated* intensity, not events. A line tagged
"screaming" over nothing happening is a false peak; the confirm pass can drop it
and the evaluation counts it.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from lexicon import emotion_score, emotion_words
from rank import _iou
from select_promos import MAX_S, MIN_S, snap_window

BIN = 0.5
MAX_INT = 4.0
HOOK_S = 3.0
END_S = 6.0
K_PER_EPISODE = int(os.getenv("PROMO_CURVE_K", "3"))
NMS_IOU = float(os.getenv("PROMO_CURVE_NMS_IOU", "0.30"))
W_HOOK, W_END, W_SUSTAIN = 0.40, 0.40, 0.20
EPS = 1e-6

ACTION_WORDS = {
    "shock", "shocked", "tears", "gasps", "gasp", "strikes", "grabs", "drags", "kneels",
    "blood", "sword", "falls", "shoves", "collapses", "stunned", "freezes", "slams",
    "stumbles", "recoils", "flinches", "clutches", "whirls", "draws",
}
SFX_WORDS = {
    "ominous", "cliffhanger", "sting", "thud", "crash", "silence", "drone", "boom",
    "swells", "gasp", "stinger", "dread",
}
THREAT_EMOTIONS = {
    "threatening", "venomous", "ominous", "menacing", "accusing", "warning", "chilling",
    "murderous", "sinister",
}
CALM_EMOTIONS = {"calm", "gentle", "tender", "soft", "reassuring", "warm", "relieved", "serene"}


def _words(text: str) -> List[str]:
    out, cur = [], []
    for ch in text.lower():
        if ch.isalpha():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def beat_intensity(b: Dict[str, Any], lex: Dict[str, float]) -> Tuple[float, float]:
    """Returns (dialogue_intensity, bonus). Exactly one is non-zero for a given beat."""
    if b["is_dialogue"]:
        val = emotion_score(b.get("emotion"), lex)
        text = b["text"]
        if "!" in text:
            val += 0.5
        if "?" in text:
            val += 0.3
        return min(val, MAX_INT), 0.0
    ws = set(_words(b["text"]))
    if b["type"] == "SOUND FX":
        hits = len(ws & SFX_WORDS)
        return 0.0, min(1.0, 0.5 * hits)
    hits = len(ws & ACTION_WORDS)
    return 0.0, min(1.0, 0.4 * hits)


def build_curve(ep: Dict[str, Any], lex: Dict[str, float]) -> List[float]:
    """Intensity on a 0.5s grid. Parallel tracks combine by max."""
    n = max(1, int(math.ceil(ep["duration_s"] / BIN)))
    dlg = [0.0] * n
    bonus = [0.0] * n
    for b in ep["beats"]:
        d, x = beat_intensity(b, lex)
        i0 = max(0, int(b["start_s"] / BIN))
        i1 = min(n, max(i0 + 1, int(math.ceil(b["end_s"] / BIN))))
        for i in range(i0, i1):
            if d > dlg[i]:
                dlg[i] = d
            if x > bonus[i]:
                bonus[i] = x
    return [min(MAX_INT, dlg[i] + bonus[i]) for i in range(n)]


def _prefix(curve: List[float]) -> List[float]:
    p = [0.0]
    for v in curve:
        p.append(p[-1] + v)
    return p


def _longest_dead_run_s(curve: List[float], i0: int, i1: int, thresh: float = 1.0) -> float:
    best = cur = 0
    for i in range(i0, i1):
        if curve[i] < thresh:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best * BIN


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_window(
    ep: Dict[str, Any], curve: List[float], pre: List[float],
    start_s: float, end_s: float, lex: Dict[str, float],
) -> Dict[str, float]:
    i0 = max(0, int(start_s / BIN))
    i1 = min(len(curve), max(i0 + 1, int(math.ceil(end_s / BIN))))
    n = i1 - i0
    hook_bins = max(1, int(HOOK_S / BIN))
    end_bins = max(1, int(END_S / BIN))

    hook = max(curve[i0:min(i1, i0 + hook_bins)]) / MAX_INT
    last = curve[max(i0, i1 - end_bins):i1]
    end_mean = (sum(last) / len(last)) / MAX_INT

    # last spoken line decides open vs resolved
    tail_dlg = [b for b in ep["beats"] if b["is_dialogue"] and b["start_s"] < end_s - EPS
                and b["start_s"] >= start_s - EPS]
    open_bonus = 0.0
    if tail_dlg:
        lb = tail_dlg[-1]
        ws = set(emotion_words(lb.get("emotion")))
        txt = lb["text"].rstrip("\"'” ")
        if txt.endswith("?"):
            open_bonus += 0.15
        if txt.endswith("...") or txt.endswith("…"):
            open_bonus += 0.05
        if ws & THREAT_EMOTIONS:
            open_bonus += 0.10
        if emotion_score(lb.get("emotion"), lex) <= 0.0 or (ws & CALM_EMOTIONS):
            open_bonus -= 0.20
    end_tension = _clamp01(end_mean + open_bonus)

    mean_all = (pre[i1] - pre[i0]) / max(1, n) / MAX_INT
    dead = _longest_dead_run_s(curve, i0, i1)
    sustain = _clamp01(mean_all - 0.30 * min(1.0, dead / 12.0))

    return {
        "hook": round(hook, 4),
        "end_tension": round(end_tension, 4),
        "sustain": round(sustain, 4),
        "score": round(W_HOOK * hook + W_END * end_tension + W_SUSTAIN * sustain, 4),
    }


def _slice_1s(curve: List[float], start_s: float, end_s: float) -> List[float]:
    """Downsample the window's curve to ~1 value per second for the report sparkline."""
    out = []
    t = start_s
    while t < end_s - EPS:
        i0 = int(t / BIN)
        i1 = min(len(curve), i0 + int(1 / BIN))
        seg = curve[i0:i1] or [0.0]
        out.append(round(sum(seg) / len(seg), 2))
        t += 1.0
    return out


def candidate_windows(
    ep: Dict[str, Any], lex: Dict[str, float], k: int = K_PER_EPISODE
) -> List[Dict[str, Any]]:
    """Top-k non-overlapping windows for one episode, best first."""
    beats = ep["beats"]
    curve = build_curve(ep, lex)
    pre = _prefix(curve)
    dlg = [b for b in beats if b["is_dialogue"]]
    if not dlg:
        return []

    seen = set()
    scored: List[Dict[str, Any]] = []
    for s_beat in dlg:
        for e_beat in dlg:
            if e_beat["end_s"] <= s_beat["start_s"]:
                continue
            raw_dur = e_beat["end_s"] - s_beat["start_s"]
            if raw_dur < MIN_S - 6 or raw_dur > MAX_S + 6:
                continue
            start_s, end_s = snap_window(beats, s_beat["start_s"], e_beat["end_s"])
            end_s = min(end_s, ep["duration_s"])
            dur = end_s - start_s
            if not (MIN_S - EPS <= dur <= MAX_S + EPS):
                continue
            key = (start_s, end_s)
            if key in seen:
                continue
            seen.add(key)
            sc = score_window(ep, curve, pre, start_s, end_s, lex)
            scored.append({"start_s": start_s, "end_s": end_s, "curve": sc})

    scored.sort(key=lambda w: (-w["curve"]["score"], w["start_s"]))
    kept: List[Dict[str, Any]] = []
    for w in scored:
        if all(_iou(w, x) < NMS_IOU for x in kept):
            kept.append(w)
        if len(kept) >= k:
            break

    out = []
    for n, w in enumerate(kept, 1):
        s, e = w["start_s"], w["end_s"]
        inside = [b for b in beats if b["start_s"] >= s - EPS and b["start_s"] < e - EPS]
        d_in = [b for b in inside if b["is_dialogue"]]
        hook_pool = [b for b in d_in if b["start_s"] <= s + HOOK_S + EPS] or d_in
        hook_b = max(hook_pool, key=lambda b: (beat_intensity(b, lex)[0], -b["start_s"]))
        out.append({
            "id": f"ep{ep['episode_no']}-w{n}",
            "episode": ep["episode_no"],
            "start_s": s,
            "end_s": e,
            "duration_s": round(e - s, 2),
            "start_idx": inside[0]["idx"],
            "end_idx": inside[-1]["idx"],
            "hook_idx": hook_b["idx"],
            "cliffhanger_idx": d_in[-1]["idx"],
            "curve": w["curve"],
            "curve_slice": _slice_1s(curve, s, e),
        })
    return out


def find_all(
    episodes: List[Dict[str, Any]], lex: Dict[str, float], k: int = K_PER_EPISODE
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep in episodes:
        out.extend(candidate_windows(ep, lex, k))
    return out
