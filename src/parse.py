"""
Screenplay -> structured beats.

Pure deterministic code. No LLM. Everything the pipeline later claims about
timestamps is grounded in what this module produces.

Input : a folder of .txt screenplays, or a .zip containing them.
Output: a list of Episode dicts (see parse_episode).

Format notes discovered by inspecting the real input pack (all 50 episodes):
  * Header keys EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES.
  * Scenes:  "SCENE 01" then Scene heading / Scene start / Scene end.
  * Beats:   "**TAG:**" then "Start: (MM:SS.ss) | End: (MM:SS.ss)" then text.
  * episode-006 uses UPPERCASE field labels where every other file uses
    title case  -> all label matching is case-insensitive.
  * 6 of 50 episodes list beats out of chronological order, because CAMERA and
    dialogue are parallel tracks -> beats are sorted by start time and indices
    are assigned AFTER sorting. Never trust file order.
  * Timestamps are episode-absolute and continuous across scenes (verified),
    always MM:SS.ss, never HH:.
  * Files are valid UTF-8.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

# The only non-dialogue tags in the corpus. Everything else is a character.
PRODUCTION_TAGS = {"CAMERA", "ACTION", "SOUND FX", "SFX", "MUSIC", "TEXT ON SCREEN", "VFX"}

# Parenthetical markers that are delivery flags, not emotion.
DELIVERY_FLAGS = {"V.O.", "VO", "O.S.", "OS", "O.C.", "CONT'D", "CONTD", "CONT’D"}

_RE_BEAT = re.compile(
    r"^\*\*(?P<tag>[^\n*]+?):\*\*[ \t]*\n"
    r"[ \t]*Start:[ \t]*\((?P<start>[\d:.]+)\)[ \t]*\|[ \t]*End:[ \t]*\((?P<end>[\d:.]+)\)[ \t]*\n"
    r"(?P<text>.*?)"
    r"(?=\n[ \t]*\n\*\*|\n[ \t]*\n[ \t]*SCENE[ \t]+\d|\n[ \t]*\n?-{2,}[ \t]*SCRIPT END|\Z)",
    re.M | re.S,
)

_RE_SCENE = re.compile(r"^[ \t]*SCENE[ \t]+(?P<num>\d+)[ \t]*$", re.M | re.I)
_RE_EP_NUM = re.compile(r"^EPISODE NUMBER:[ \t]*(?P<v>.+)$", re.M | re.I)
_RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:[ \t]*(?P<v>.+)$", re.M | re.I)
_RE_CONTINUITY = re.compile(r"^CONTINUITY NOTES:[ \t]*(?P<v>.+)$", re.M | re.I)
_RE_SCENE_HEADING = re.compile(r"^[ \t]*Scene heading:[ \t]*(?P<v>.+)$", re.M | re.I)
_RE_TS = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")

# Per-show alias map, e.g. {"LADY ALICE": "ALICE"}: the same person tagged several
# ways. EMPTY by default and loaded from a config file per show (--aliases), never
# hardcoded here, for two reasons:
#   * cast names, and which tags are the same person, are show content and this
#     code is meant to be shareable;
#   * a blind auto-merge would wrongly collapse distinct people ("LORD X" vs
#     "LADY X"), so merges are always explicit.
# Merging is cosmetic downstream: an unmerged alias never breaks anything.
ALIASES: Dict[str, str] = {}


def set_aliases(mapping: Dict[str, str]) -> None:
    """Replace the alias map in place (so existing references stay valid)."""
    ALIASES.clear()
    ALIASES.update({k.strip().upper(): v.strip().upper() for k, v in mapping.items()})


def load_aliases(path: str) -> int:
    """Load a JSON object {"LADY ALICE": "ALICE", ...}. Returns the number of aliases."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ParseError(f"alias file must be a JSON object: {path}")
    set_aliases(data)
    return len(ALIASES)


class ParseError(Exception):
    pass


def ts_to_seconds(value: str) -> float:
    """'01:23.45' -> 83.45. The corpus is always MM:SS.ss; HH:MM:SS is tolerated."""
    value = value.strip()
    parts = value.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise ParseError(f"unparseable timestamp: {value!r}")


def seconds_to_ts(seconds: float) -> str:
    """83.45 -> '01:23.45'. Used for human-facing output."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def _split_tag(raw_tag: str) -> Tuple[str, Optional[str], List[str]]:
    """
    '**ALICE (V.O.) (Tender, Safe)**' -> ('ALICE', 'Tender, Safe', ['V.O.'])

    Returns (speaker_or_type, emotion, flags). Aliases are NOT applied here;
    parse_episode applies them afterwards.
    """
    parens = re.findall(r"\(([^()]*)\)", raw_tag)
    name = re.sub(r"\([^()]*\)", " ", raw_tag)
    name = re.sub(r"\s+", " ", name).strip().upper()

    flags: List[str] = []
    emotions: List[str] = []
    for p in parens:
        token = p.strip()
        if token.upper().replace(" ", "") in {f.replace(" ", "") for f in DELIVERY_FLAGS}:
            flags.append(token.upper())
        else:
            emotions.append(token)

    emotion = ", ".join(emotions) if emotions else None
    return name, emotion, flags


def _scene_spans(body: str) -> List[Tuple[int, int, int]]:
    """Character spans of each SCENE block: (scene_number, start_offset, end_offset)."""
    marks = [(int(m.group("num")), m.start()) for m in _RE_SCENE.finditer(body)]
    if not marks:
        return [(1, 0, len(body))]
    spans = []
    for i, (num, off) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(body)
        spans.append((num, off, end))
    return spans


def parse_episode(text: str, source_name: str = "") -> Dict[str, Any]:
    """Parse one screenplay file into an episode dict with sorted, indexed beats."""
    m = _RE_EP_NUM.search(text)
    ep_label = m.group("v").strip() if m else ""
    ep_no = None
    if ep_label:
        digits = re.search(r"(\d+)", ep_label)
        if digits:
            ep_no = int(digits.group(1))
    if ep_no is None:
        digits = re.search(r"episode[-_ ]?(\d+)", source_name, re.I)
        ep_no = int(digits.group(1)) if digits else -1

    logline_m = _RE_LOGLINE.search(text)
    continuity_m = _RE_CONTINUITY.search(text)

    # Body = everything after SCRIPT START, so header text is never read as a beat.
    start_marker = re.search(r"-{2,}\s*SCRIPT START\s*-{2,}", text, re.I)
    body = text[start_marker.end():] if start_marker else text

    beats: List[Dict[str, Any]] = []
    for scene_no, lo, hi in _scene_spans(body):
        chunk = body[lo:hi]
        heading_m = _RE_SCENE_HEADING.search(chunk)
        heading = heading_m.group("v").strip() if heading_m else None
        for bm in _RE_BEAT.finditer(chunk):
            try:
                start_s = ts_to_seconds(bm.group("start"))
                end_s = ts_to_seconds(bm.group("end"))
            except ParseError:
                continue
            if end_s < start_s:
                start_s, end_s = end_s, start_s
            name, emotion, flags = _split_tag(bm.group("tag"))
            is_dialogue = name not in PRODUCTION_TAGS
            speaker = ALIASES.get(name, name) if is_dialogue else None
            body_text = re.sub(r"\s+", " ", bm.group("text")).strip()
            beats.append(
                {
                    "scene": scene_no,
                    "scene_heading": heading,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "type": "DIALOGUE" if is_dialogue else name,
                    "speaker": speaker,
                    "speaker_raw": name if is_dialogue else None,
                    "emotion": emotion,
                    "flags": flags,
                    "is_dialogue": is_dialogue,
                    "text": body_text,
                }
            )

    if not beats:
        raise ParseError(f"no beats parsed from {source_name!r}")

    # Sort by start time, then end time. 6/50 episodes are out of order on disk.
    beats.sort(key=lambda b: (b["start_s"], b["end_s"]))
    for i, b in enumerate(beats):
        b["idx"] = i

    duration_s = max(b["end_s"] for b in beats)

    return {
        "episode_no": ep_no,
        "episode_label": ep_label,
        "source_file": source_name,
        "logline": logline_m.group("v").strip() if logline_m else "",
        "continuity": continuity_m.group("v").strip() if continuity_m else "",
        "duration_s": round(duration_s, 2),
        "n_beats": len(beats),
        "beats": beats,
    }


def _iter_sources(path: str) -> Iterable[Tuple[str, str]]:
    """Yield (name, text) for every screenplay in a folder or zip."""
    if os.path.isdir(path):
        names = sorted(
            n for n in os.listdir(path) if n.lower().endswith(".txt")
        )
        if not names:
            raise ParseError(f"no .txt screenplays found in folder: {path}")
        for n in names:
            with open(os.path.join(path, n), "r", encoding="utf-8", errors="replace") as fh:
                yield n, fh.read()
        return

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                n for n in zf.namelist()
                if n.lower().endswith(".txt") and not n.startswith("__MACOSX")
            )
            if not names:
                raise ParseError(f"no .txt screenplays found in zip: {path}")
            for n in names:
                yield os.path.basename(n), zf.read(n).decode("utf-8", errors="replace")
        return

    if os.path.isfile(path) and path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield os.path.basename(path), fh.read()
        return

    raise ParseError(f"input must be a folder, a .zip, or a .txt file: {path}")


def load_episodes(path: str) -> List[Dict[str, Any]]:
    """Parse every screenplay under `path`. Sorted by episode number."""
    episodes = []
    for name, text in _iter_sources(path):
        episodes.append(parse_episode(text, source_name=name))
    episodes.sort(key=lambda e: (e["episode_no"], e["source_file"]))
    return episodes


def write_beats_jsonl(episodes: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")


def render_compact(ep: Dict[str, Any], include_logline: bool = False) -> str:
    """
    Compact per-beat rendering sent to the LLM.

    Indices are namespaced `ep{N}:{idx}` so a batched call cannot confuse one
    episode's index 41 with another's. Roughly halves tokens versus raw source.
    CAMERA beats are kept -- visual reveals and slaps live there -- but terse.
    """
    lines = [f"### EPISODE {ep['episode_no']} (duration {seconds_to_ts(ep['duration_s'])})"]
    if include_logline:
        lines.append(f"LOGLINE: {ep['logline']}")
    for b in ep["beats"]:
        tag = b["speaker"] if b["is_dialogue"] else b["type"]
        if b["is_dialogue"] and b["emotion"]:
            tag = f"{tag}({b['emotion']})"
        if b["flags"]:
            tag = f"{tag}[{'/'.join(b['flags'])}]"
        lines.append(
            f"[ep{ep['episode_no']}:{b['idx']}] {seconds_to_ts(b['start_s'])}-"
            f"{seconds_to_ts(b['end_s'])} {tag}: {b['text']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse screenplays into beats.")
    ap.add_argument("input", help="folder, .zip, or .txt")
    ap.add_argument("--out", default=None, help="write beats.jsonl here")
    ap.add_argument("--compact", type=int, default=None, help="print compact render of episode N")
    args = ap.parse_args()

    eps = load_episodes(args.input)
    total_beats = sum(e["n_beats"] for e in eps)
    print(f"episodes: {len(eps)}  beats: {total_beats}")
    print(
        "duration_s: min %.1f  max %.1f  mean %.1f"
        % (
            min(e["duration_s"] for e in eps),
            max(e["duration_s"] for e in eps),
            sum(e["duration_s"] for e in eps) / len(eps),
        )
    )
    if args.compact is not None:
        target = next(e for e in eps if e["episode_no"] == args.compact)
        print(render_compact(target))
    if args.out:
        write_beats_jsonl(eps, args.out)
        print(f"wrote {args.out}")
