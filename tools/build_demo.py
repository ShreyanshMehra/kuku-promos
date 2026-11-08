"""
Builds demo/real_run_clips.json from a finished run, WITHOUT any screenplay text.

    python tools/build_demo.py out/full

The evaluation numbers below are copied from out/full/metrics.md (the blind rating is a
Gemini chat model's, not a human's). Re-run and update them if the run changes.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from webdata import build_clips  # noqa: E402

EVAL = {
    "rater": "a Gemini chat model (an LLM, not a human)",
    "checks": [
        ["duration 20-60s", 68, 68], ["inside the episode", 68, 68], ["starts on a line boundary", 68, 68],
        ["ends on a line boundary", 68, 68], ["no dialogue chopped", 68, 68], ["hook within 3s", 68, 68],
        ["cliffhanger in last 6s", 68, 68], ["hook quote is verbatim", 68, 68],
        ["cliffhanger quote is verbatim", 68, 68], ["segments ordered, disjoint", 68, 68],
    ],
    "blind": {
        "groups": [
            {"group": "Tool, all 15", "hook": 1.27, "cliffhanger": 1.60, "standalone": 1.53, "total": 4.4},
            {"group": "Baseline: last 40s", "hook": 0.80, "cliffhanger": 1.13, "standalone": 1.47, "total": 3.4},
            {"group": "Tool top 5", "hook": 1.2, "cliffhanger": 2.0, "standalone": 2.0, "total": 5.2},
            {"group": "Tool random 5", "hook": 1.4, "cliffhanger": 1.2, "standalone": 1.2, "total": 3.8},
            {"group": "Tool bottom 5", "hook": 1.2, "cliffhanger": 1.6, "standalone": 1.4, "total": 4.2},
        ],
        "paired": {"tool_wins": 7, "ties": 5, "baseline_wins": 2},
        "failures": {
            "tool": {"none": 5, "weak_hook": 4, "cut_mid_reaction": 3, "needs_backstory": 2, "resolved_ending": 1},
            "baseline": {"none": 4, "weak_hook": 6, "resolved_ending": 3, "needs_backstory": 1, "cut_mid_reaction": 1},
        },
    },
    "panel": {
        "all_survive_3s": 0.85, "watch_to_end": {"Newcomer fan": 0.88, "Impatient scroller": 0.23, "Skeptic": 0.28},
        "evidence_supported": 0.94, "evidence_unsupported": 0.02, "spearman_curve_vs_panel": 0.058,
        "model_dropped_share": 0.50, "windows": 143,
    },
}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "out/full"
    data = build_clips(os.path.join(run, "promos.json"), with_text=False)
    data["eval"] = EVAL
    os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
    path = os.path.join(ROOT, "demo", "real_run_clips.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    text_keys = {"hook_line", "cliffhanger_line", "why", "reason", "quote"}
    leaked = [k for c in data["clips"] for k in list(c) + [k for v in c["viewers"] for k in v] if k in text_keys]
    assert not leaked, f"script text leaked into demo: {set(leaked)}"
    print(f"wrote {path}: {len(data['clips'])} clips, no script text")


if __name__ == "__main__":
    main()
