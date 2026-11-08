"""
Turns a finished run (promos.json) into the flat per-clip records the web app draws.

with_text=False drops every line of screenplay (hook line, cliffhanger line, reasons,
quotes, why) and keeps only numbers and timestamps. That is the version shipped in the
repo for the real 50-episode run, because the pack is Kuku's and is not published.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

PERSONAS = {1: "Newcomer fan", 2: "Impatient scroller", 3: "Skeptic"}


def build_clips(promos_json_path: str, with_text: bool) -> Dict[str, Any]:
    with open(promos_json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    clips: List[Dict[str, Any]] = []
    for r in d.get("ranked", []):
        p = r.get("panel") or {}
        viewers = []
        for v in p.get("viewers", []):
            item = {
                "persona": v.get("persona"),
                "swipe_at_s": v.get("swipe_at_s"),
                "tap_next": v.get("tap_next"),
                "evidence_status": v.get("evidence_status"),
            }
            if with_text:
                item["reason"] = v.get("reason")
                item["quote"] = (v.get("evidence") or {}).get("quote")
            viewers.append(item)
        c = {
            "rank": r.get("rank"),
            "feed": r.get("feed"),
            "episode": r.get("episode"),
            "start": r.get("start_ts"),
            "end": r.get("end_ts"),
            "duration_s": r.get("duration_s"),
            "archetype": r.get("archetype"),
            "source": r.get("source"),
            "final_score": r.get("final_score"),
            "retention": p.get("retention"),
            "hook_survival": p.get("hook_survival"),
            "watch_through": p.get("watch_through"),
            "tap_next": p.get("tap_next"),
            "evidence_support": p.get("evidence_support"),
            "needs_review": p.get("needs_review"),
            "recruitment": r.get("recruitment"),
            "curve": r.get("curve_slice") or [],
            "viewers": viewers,
        }
        if with_text:
            c["hook_line"] = r.get("hook_quote")
            c["cliffhanger_line"] = r.get("cliffhanger_quote")
            c["why"] = r.get("why")
        clips.append(c)
    meta = d.get("meta", {})
    keep = ("model", "judge_model", "episodes", "windows_found", "dropped_by_model", "tier",
            "tier_message", "run_at", "scorer", "provider")
    return {"clips": clips, "meta": {k: meta.get(k) for k in keep}, "with_text": with_text}
