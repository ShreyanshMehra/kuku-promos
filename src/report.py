"""
Write the outputs an editor (and a reviewer) actually opens.

  promos.csv    one row per ranked promo -- what an editor works from
  promos.json   everything, including repairs, scores, curve and panel data
  rejected.json proposals the validator refused AND windows the model dropped, with
                reasons (the raw material for the failure analysis -- never silently
                discarded)
  report.html   plain page, each clip's script text inlined, with its tension curve
                and the simulated viewers' swipe points

CONFIDENTIALITY: report.html, promos.* and rejected.json contain screenplay text.
They belong in a link-only folder, never a public repo. out/ is gitignored.
"""

from __future__ import annotations

import csv
import html
import json
import os
from typing import Any, Dict, List

from rank import clip_lines, segments_str

CSV_COLUMNS = [
    "rank", "feed", "promo_id", "episode", "segments", "start", "end", "duration_s",
    "archetype", "hook_line", "hook_at_s", "cliffhanger_line", "why_it_works",
    "source", "scorer", "retention", "hook_survival", "watch_through", "tap_next",
    "swipe_spread", "evidence_support", "needs_review",
    "curve_score", "curve_hook", "curve_end", "curve_sustain",
    "judge_hook", "judge_cliffhanger", "judge_standalone",
    "quality", "recruitment", "final_score", "spoils_arc", "carries_premise",
    "stranger_note", "repairs",
]


def _row(c: Dict[str, Any], ep: Dict[str, Any]) -> Dict[str, Any]:
    p = c.get("panel") or {}
    j = c.get("judge") or {}
    k = c.get("curve") or {}
    hook_at = round(ep["beats"][c["hook_beat"]]["start_s"] - c["start_s"], 1)
    return {
        "rank": c["rank"], "feed": c["feed"], "promo_id": c["id"], "episode": c["episode"],
        "segments": segments_str(c), "start": c["start_ts"], "end": c["end_ts"],
        "duration_s": c["duration_s"], "archetype": c.get("archetype", ""),
        "hook_line": c["hook_quote"], "hook_at_s": hook_at,
        "cliffhanger_line": c["cliffhanger_quote"], "why_it_works": c["why"],
        "source": c.get("source", "llm-selected"), "scorer": c.get("scorer", ""),
        "retention": p.get("retention", ""), "hook_survival": p.get("hook_survival", ""),
        "watch_through": p.get("watch_through", ""), "tap_next": p.get("tap_next", ""),
        "swipe_spread": p.get("swipe_spread", ""),
        "evidence_support": "" if p.get("evidence_support") is None else p.get("evidence_support"),
        "needs_review": p.get("needs_review", ""),
        "curve_score": k.get("score", ""), "curve_hook": k.get("hook", ""),
        "curve_end": k.get("end_tension", ""), "curve_sustain": k.get("sustain", ""),
        "judge_hook": j.get("hook_strength", ""), "judge_cliffhanger": j.get("cliffhanger_strength", ""),
        "judge_standalone": j.get("standalone_clarity", ""),
        "quality": c["quality"], "recruitment": c["recruitment"], "final_score": c["final_score"],
        "spoils_arc": c["spoils_flag"], "carries_premise": c["premise_flag"],
        "stranger_note": c.get("standalone_notes", ""), "repairs": " | ".join(c["repairs"]),
    }


def write_csv(path: str, ranked: List[Dict[str, Any]], by_ep: Dict[int, Dict[str, Any]], top: int = 0) -> None:
    rows = ranked[:top] if top else ranked
    # utf-8-sig so Excel opens the file with correct characters
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for c in rows:
            w.writerow(_row(c, by_ep[c["episode"]]))


_PERSONA_LABEL = {1: "fan", 2: "scroller", 3: "skeptic"}


def _sparkline(c: Dict[str, Any]) -> str:
    """Inline SVG of the clip's tension curve; dashed lines mark where each simulated viewer swipes."""
    sl = c.get("curve_slice") or []
    if len(sl) < 2:
        return ""
    w, h, pad = 600, 56, 4
    n = len(sl)
    pts = " ".join(
        f"{pad + i / (n - 1) * (w - 2 * pad):.1f},{h - pad - min(v, 4.0) / 4.0 * (h - 2 * pad):.1f}"
        for i, v in enumerate(sl)
    )
    parts = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' role='img' "
        f"aria-label='tension curve over the clip, with viewer swipe points'>",
        f"<rect width='{w}' height='{h}' fill='#fafafa' stroke='#ddd'/>",
        f"<polyline points='{pts}' fill='none' stroke='#b3261e' stroke-width='2'/>",
    ]
    dur = max(c["duration_s"], 1.0)
    for v in (c.get("panel") or {}).get("viewers", []):
        sw = v.get("swipe_at_s")
        if sw is None:
            continue
        x = pad + min(sw / dur, 1.0) * (w - 2 * pad)
        label = _PERSONA_LABEL.get(v.get("persona"), "viewer")
        parts.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='0' y2='{h}' stroke='#1a56db' "
                     f"stroke-dasharray='3 3'/><text x='{x + 3:.1f}' y='11' font-size='10' fill='#1a56db'>{label}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _summary_line(c: Dict[str, Any]) -> str:
    p, j = c.get("panel"), c.get("judge")
    if p:
        bits = (f"hook survival {p['hook_survival']:.0%} &middot; watch-through {p['watch_through']:.0%} "
                f"&middot; tap next {p['tap_next']:.0%}"
                f"{' &middot; <b>NEEDS REVIEW: panel evidence weak</b>' if p.get('needs_review') else ''}")
    elif j:
        bits = (f"hook {j['hook_strength']}/5 &middot; cliffhanger {j['cliffhanger_strength']}/5 "
                f"&middot; standalone {j['standalone_clarity']}/5")
    else:
        bits = "curve-only (no model confirmation)"
    k = c.get("curve")
    curve = f" &middot; curve {k['score']:.2f}" if k else ""
    return f"{bits}{curve} &middot; final {c['final_score']:.3f}"


def _provenance_html(c: Dict[str, Any]) -> str:
    """Collapsible 'how this clip was chosen' trail, from c['provenance']."""
    steps = c.get("provenance") or []
    if not steps:
        return ""
    esc = html.escape
    from parse import seconds_to_ts

    def span(a, b):
        return "?" if a is None or b is None else f"{seconds_to_ts(a)}-{seconds_to_ts(b)}"

    items = []
    for s in steps:
        st = s["stage"]
        if st == "curve":
            items.append(f"<b>curve</b> proposed {span(s['start_s'], s['end_s'])} (curve score {s['curve_score']})")
        elif st in ("confirm", "select"):
            if s.get("verdict") == "skipped":
                items.append(f"<b>confirm</b> skipped: {esc(s['note'])}")
            else:
                v = f" verdict <b>{esc(s['verdict'])}</b>" if s.get("verdict") else ""
                items.append(f"<b>{st}</b> ({esc(str(s.get('model')))}){v}; model's window {span(s['model_start_s'], s['model_end_s'])}")
        elif st == "repair":
            items.append("<b>repair</b> " + esc("; ".join(s["actions"])) + f" &rarr; {span(s['start_s'], s['end_s'])}")
        elif st == "panel":
            ev = "" if s.get("evidence_support") is None else f", evidence support {s['evidence_support']:.0%}"
            rv = " <b>(needs review)</b>" if s.get("needs_review") else ""
            items.append(f"<b>panel</b> ({esc(str(s.get('model')))}) retention {s['retention']}, "
                         f"hook survival {s['hook_survival']}, tap next {s['tap_next']}{ev}{rv}")
        elif st == "judge":
            items.append(f"<b>judge</b> ({esc(str(s.get('model')))}) quality {s['quality']}")
        elif st == "rank":
            items.append(f"<b>rank</b> #{s['rank']}: quality {s['quality']} &times; recruitment {s['recruitment']} "
                         f"= {s['final_score']} ({esc(s['feed'])})")
    return ("<details><summary class='m'>how this clip was chosen</summary><ol class='m'>"
            + "".join(f"<li>{i}</li>" for i in items) + "</ol></details>")


def write_html(path: str, ranked: List[Dict[str, Any]], by_ep: Dict[int, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Promo candidates</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}"
        "h2{margin:2rem 0 .25rem}.m{color:#555;font-size:13px}.b{background:#f5f5f5;padding:.6rem .8rem;"
        "border-radius:6px;font:13px/1.45 ui-monospace,monospace;white-space:pre-wrap}"
        ".hook{background:#fff3cd}.cliff{background:#fde2e1}.tag{display:inline-block;padding:0 .5rem;"
        "border-radius:9px;background:#e8e8e8;font-size:12px;margin-right:.3rem}</style></head><body>",
        "<h1>Promo candidates</h1>",
        f"<p style='padding:.5rem .8rem;border-radius:6px;background:"
        f"{'#e6f4ea' if meta.get('tier') == 'full' else '#fff3cd'}'>"
        f"<b>Run confidence: {esc(str(meta.get('tier', 'unknown')).upper())}.</b> "
        f"{esc(str(meta.get('tier_message', '')))}</p>",
        f"<p class='m'>finder: {esc(str(meta.get('finder')))} &middot; scorer: {esc(str(meta.get('scorer')))} "
        f"&middot; model: {esc(str(meta.get('model')))} &middot; judge: {esc(str(meta.get('judge_model')))} "
        f"&middot; {len(ranked)} ranked promos</p>",
        "<p class='m'>Judged from screenplay text only: visual impact, performance and pacing are NOT assessed. "
        "The viewer panel is an LLM simulating viewers, not measured audience behaviour. "
        "Red line = tension curve; blue dashed = where a simulated viewer swipes. "
        "Contains screenplay text. Do not publish.</p>",
    ]
    for c in ranked:
        ep = by_ep[c["episode"]]
        parts.append(
            f"<h2>#{c['rank']} &middot; Episode {c['episode']} &middot; "
            f"{esc(c['start_ts'])}&ndash;{esc(c['end_ts'])} ({c['duration_s']:.0f}s)</h2>"
            f"<div class='m'><span class='tag'>{esc(c['feed'])}</span>"
            f"<span class='tag'>{esc(c.get('archetype', ''))}</span>"
            f"<span class='tag'>{esc(c.get('source', 'llm-selected'))}</span>"
            f"{_summary_line(c)}"
            f"{' &middot; spoils arc' if c['spoils_flag'] else ''}"
            f"{' &middot; carries premise' if c['premise_flag'] else ''}</div>"
            f"{_sparkline(c)}"
            f"<p><b>Hook:</b> &ldquo;{esc(c['hook_quote'])}&rdquo;<br>"
            f"<b>Cliffhanger:</b> &ldquo;{esc(c['cliffhanger_quote'])}&rdquo;<br>"
            f"<b>Why:</b> {esc(c['why'])}</p>"
        )
        if c.get("panel"):
            reasons = "; ".join(
                f"{_PERSONA_LABEL.get(v.get('persona'), 'viewer')} [{v.get('evidence_status', 'unchecked')}]: {v['reason']}"
                for v in c["panel"]["viewers"] if v.get("reason")
            )
            if reasons:
                parts.append(f"<p class='m'>panel: {esc(reasons)}</p>")
        if c["repairs"]:
            parts.append(f"<p class='m'>boundary repairs: {esc(' | '.join(c['repairs']))}</p>")
        parts.append(_provenance_html(c))
        lines = []
        for ln in clip_lines(ep, c):
            cls = ""
            if c["hook_quote"] and c["hook_quote"] in ln:
                cls = "hook"
            if c["cliffhanger_quote"] and c["cliffhanger_quote"] in ln:
                cls = "cliff"
            lines.append(f"<span class='{cls}'>{esc(ln)}</span>" if cls else esc(ln))
        parts.append("<div class='b'>" + "\n".join(lines) + "</div>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def write_all(out_dir: str, ranking: Dict[str, Any], selection: Dict[str, Any],
              episodes: List[Dict[str, Any]], meta: Dict[str, Any], top: int = 0) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    by_ep = {e["episode_no"]: e for e in episodes}
    paths = {
        "csv": os.path.join(out_dir, "promos.csv"),
        "json": os.path.join(out_dir, "promos.json"),
        "rejected": os.path.join(out_dir, "rejected.json"),
        "html": os.path.join(out_dir, "report.html"),
    }
    write_csv(paths["csv"], ranking["ranked"], by_ep, top=top)
    write_html(paths["html"], ranking["ranked"][: top or None], by_ep, meta)
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "meta": meta,
                "feeds": ranking["feeds"],
                "ranked": ranking["ranked"],
                "duplicates": ranking["duplicates"],
                "unjudged": ranking["unjudged"],
            },
            fh, ensure_ascii=False, indent=2,
        )
    with open(paths["rejected"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "rejected": selection["rejected"],
                "dropped_by_model": selection.get("dropped", []),
                "failures": selection["failures"],
            },
            fh, ensure_ascii=False, indent=2,
        )
    return paths
