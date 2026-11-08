"""
Honest quality checking. Three kinds of evidence, kept strictly separate
because they carry different weight:

  1. PROGRAMMATIC  (python src/evaluate.py check)
     Re-derives every check from the FINAL timestamps and the parsed beats.
     Independent of the selection-time validator's flags: a bug that let a bad
     clip through the validator shows up here. Verifiable fact, 100% coverage.

  2. HUMAN, BLIND  (python src/evaluate.py sample  ->  you rate  ->  score)
     Top 5 + random 5 + bottom 5 clips, each paired with a naive baseline clip
     from the same episode ("last 40 seconds"), shuffled with labels hidden.
     Rating only the winners would say nothing about where the tool picks badly,
     and rating a labelled sample would flatter the tool. Subjective, small N,
     stated as small N.

  3. NOT JUDGEABLE FROM TEXT
     Visual impact of the footage, performance, pacing, music, whether the
     AI-generated video at a timestamp is clean. Named, not hidden.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from parse import load_episodes, seconds_to_ts  # noqa: E402
from rank import _iou, clip_lines  # noqa: E402
from select_promos import (  # noqa: E402
    CLIFF_WINDOW_S, HOOK_WINDOW_S, MAX_S, MIN_S, parse_ref, quote_is_grounded, snap_window,
)

EPS = 1e-6
BASELINE_S = 40.0
SAME_WINDOW_IOU = 0.9   # tool window this close to the baseline window = "same clip"
FAILURE_TYPES = [
    "none",
    "needs_backstory",          # a stranger cannot follow who/why
    "hook_is_camera_only",      # opening is a shot, no readable event or line
    "weak_hook",                # something happens but nothing stops a scroll
    "cut_mid_reaction",         # ends on a reaction beat, not an open question
    "resolved_ending",          # the conflict is answered before the cut
    "spoils_later_twist",       # gives away an arc turn
    "near_duplicate",           # same beat as a better clip
    "premise_invisible",        # does not show what the show is about
    "curve_false_peak",         # tagged intense (e.g. "screaming") but nothing actually happens
    "other",
]


# --------------------------------------------------------------------------
# 1. programmatic checks
# --------------------------------------------------------------------------

def check_promo(c: Dict[str, Any], ep: Dict[str, Any]) -> Dict[str, bool]:
    """Every check re-derived from final times + beats. Returns {check_name: passed}."""
    beats = ep["beats"]
    segs = c["segments"]
    out: Dict[str, bool] = {}

    s, e = c["start_s"], c["end_s"]
    out["duration_in_range"] = MIN_S - EPS <= (e - s) <= MAX_S + EPS
    out["within_episode"] = s >= -EPS and e <= ep["duration_s"] + EPS
    out["start_on_beat_boundary"] = any(abs(b["start_s"] - s) < 0.011 for b in beats)
    out["end_on_beat_boundary"] = any(abs(b["end_s"] - e) < 0.011 for b in beats)

    straddle = [
        b["idx"] for b in beats if b["is_dialogue"] and (
            (b["start_s"] < s - EPS and b["end_s"] > s + EPS)
            or (b["start_s"] < e - EPS and b["end_s"] > e + EPS)
        )
    ]
    out["no_dialogue_chopped"] = not straddle

    hook = beats[c["hook_beat"]]
    out["hook_within_3s"] = s - EPS <= hook["start_s"] <= s + HOOK_WINDOW_S + EPS
    cliff = beats[c["cliffhanger_beat"]]
    out["cliffhanger_in_last_6s"] = (
        cliff["start_s"] >= s - EPS and cliff["start_s"] <= e + EPS and cliff["end_s"] >= e - CLIFF_WINDOW_S
    )
    out["hook_quote_verbatim"] = quote_is_grounded(c["hook_quote"], hook["text"])
    out["cliffhanger_quote_verbatim"] = quote_is_grounded(c["cliffhanger_quote"], cliff["text"])

    ordered = all(segs[i]["end_s"] <= segs[i + 1]["start_s"] + EPS for i in range(len(segs) - 1))
    out["segments_ordered_disjoint"] = ordered
    return out


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Rank correlation with average ranks for ties. None if undefined."""
    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def programmatic(out_dir: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_ep = {e["episode_no"]: e for e in episodes}
    with open(os.path.join(out_dir, "promos.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    rej_path = os.path.join(out_dir, "rejected.json")
    rejected = json.load(open(rej_path, encoding="utf-8")) if os.path.exists(rej_path) else {"rejected": [], "failures": []}

    ranked = data["ranked"]
    totals: Counter = Counter()
    failing: List[str] = []
    for c in ranked:
        res = check_promo(c, by_ep[c["episode"]])
        for k, v in res.items():
            totals[k] += int(v)
        if not all(res.values()):
            failing.append(c["id"] + ": " + ", ".join(k for k, v in res.items() if not v))

    n = len(ranked)
    n_ok_sel = n + len(data.get("duplicates", [])) + len(data.get("unjudged", []))
    reasons: Counter = Counter()
    for r in rejected["rejected"]:
        for reason in r.get("reasons", []):
            key = (
                "ungrounded quote" if "ungrounded" in reason else
                "hook not within 3s" if reason.startswith("hook beat starts") else
                "cliffhanger not at end" if "cliffhanger beat not" in reason else
                "duration out of range" if reason.startswith("duration") else
                "dialogue chopped" if "straddles" in reason else
                "bad / cross-episode index" if any(t in reason for t in ("malformed", "out of range", "cross-episode", "end_idx before")) else
                "other"
            )
            reasons[key] += 1

    repaired = sum(1 for c in ranked if c["repairs"])
    judges = [c["judge"] for c in ranked if c.get("judge")]
    score_dist = {
        name: dict(sorted(Counter(j[name] for j in judges).items()))
        for name in ("hook_strength", "cliffhanger_strength", "standalone_clarity")
    } if judges else {}
    disagree = {
        flag: sum(1 for c in ranked if c.get("flag_disagreement") and
                  c["flag_disagreement"][flag][0] != c["flag_disagreement"][flag][1])
        for flag in ("spoils_arc", "carries_premise")
    }

    # --- panel discrimination: does it separate clips, or rate everything alike? ---
    panels = [c["panel"] for c in ranked if c.get("panel")]
    panel_stats: Dict[str, Any] = {}
    if panels:
        rets = [p["retention"] for p in panels]
        mean_r = sum(rets) / len(rets)
        panel_stats = {
            "n": len(panels),
            "retention_mean": round(mean_r, 3),
            "retention_sd": round((sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5, 3),
            "mean_hook_survival": round(sum(p["hook_survival"] for p in panels) / len(panels), 3),
            "mean_watch_through": round(sum(p["watch_through"] for p in panels) / len(panels), 3),
            "mean_tap_next": round(sum(p["tap_next"] for p in panels) / len(panels), 3),
            "share_all_viewers_survive_3s": round(
                sum(1 for p in panels if p["hook_survival"] >= 0.999) / len(panels), 3),
        }
        # Per persona: a persona whose tap_next never moves (or who always watches to
        # the end) adds a constant, not signal. Found this way in the first live run.
        per: Dict[Any, List[Dict[str, Any]]] = {}
        for p in panels:
            for v in p["viewers"]:
                per.setdefault(v.get("persona"), []).append(v)
        panel_stats["per_persona"] = {}
        for persona, vs in sorted(per.items(), key=lambda kv: str(kv[0])):
            taps = [v["tap_next"] for v in vs]
            mt = sum(taps) / len(taps)
            panel_stats["per_persona"][str(persona)] = {
                "n": len(vs),
                "tap_mean": round(mt, 2),
                "tap_sd": round((sum((t - mt) ** 2 for t in taps) / len(taps)) ** 0.5, 2),
                "share_watch_to_end": round(sum(1 for v in vs if v["swipe_at_s"] is None) / len(vs), 2),
            }

    # --- does the model add anything beyond the free curve? ---
    pairs = [(c["curve"]["score"], c["quality"]) for c in ranked
             if c.get("curve") and c.get("scorer") in ("panel", "judge")]
    rho = spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 5 else None
    meta = data.get("meta", {})
    dropped = rejected.get("dropped_by_model", [])
    n_windows = meta.get("windows_found")
    kept_curve = [c["curve"]["score"] for c in ranked if c.get("curve")]
    dropped_curve = [d["curve"]["score"] for d in dropped if d.get("curve") and d.get("by") == "model"]
    finder = {
        "finder": meta.get("finder"),
        "scorer": meta.get("scorer"),
        "windows_found": n_windows,
        "dropped_by_model": len(dropped),
        "share_dropped": round(len(dropped) / n_windows, 3) if n_windows else None,
        "mean_curve_score_kept": round(sum(kept_curve) / len(kept_curve), 3) if kept_curve else None,
        "mean_curve_score_dropped": round(sum(dropped_curve) / len(dropped_curve), 3) if dropped_curve else None,
        "spearman_curve_vs_scorer": rho,
        "n_pairs": len(pairs),
        "sources": dict(Counter(c.get("source", "llm-selected") for c in ranked)),
        "trimmed_by_model": sum(1 for c in ranked if c.get("trimmed_by_model")),
        "lexicon_source": meta.get("lexicon_source"),
    }
    eps_covered = sorted({c["episode"] for c in ranked})
    return {
        "panel_stats": panel_stats,
        "finder_stats": finder,
        "n_unjudged": len(data.get("unjudged", [])),
        "n_duplicates": len(data.get("duplicates", [])),
        "n_ranked": n,
        "n_selected_ok": n_ok_sel,
        "n_rejected_by_validator": len(rejected["rejected"]),
        "n_batch_failures": len(rejected["failures"]),
        "final_check_pass_counts": dict(totals),
        "final_failures": failing,
        "validator_rejection_reasons": dict(reasons),
        "ranked_needing_repair": repaired,
        "judge_score_distribution": score_dist,
        "selector_vs_judge_flag_disagreements": disagree,
        "episodes_covered": len(eps_covered),
        "early_ep_share_top10": round(
            sum(1 for c in ranked[:10] if c["episode"] <= 15) / max(1, min(10, n)), 2
        ),
        "mean_duration_s": round(sum(c["duration_s"] for c in ranked) / max(1, n), 1),
    }


def render_programmatic(p: Dict[str, Any]) -> str:
    n = max(1, p["n_ranked"])
    lines = [
        "## 1. Programmatic checks (re-derived from final timestamps, 100% of ranked promos)",
        "",
        f"- ranked promos: **{p['n_ranked']}** across **{p['episodes_covered']}** episodes",
        f"- proposals rejected at selection time: **{p['n_rejected_by_validator']}** "
        f"(of {p['n_rejected_by_validator'] + p['n_selected_ok']} model proposals)",
        f"- batches that failed outright: **{p['n_batch_failures']}**",
        f"- candidates the scorer could not score (left out of the ranking): **{p['n_unjudged']}**",
        f"- near-duplicates folded into a better clip: **{p['n_duplicates']}**",
        f"- ranked promos that needed a boundary repair: **{p['ranked_needing_repair']}**",
        "",
        "| check | passed | rate |",
        "|---|---|---|",
    ]
    for k, v in p["final_check_pass_counts"].items():
        lines.append(f"| {k} | {v}/{p['n_ranked']} | {v / n:.0%} |")
    lines += ["", "**Why the validator rejected proposals** (the model's raw misses):", ""]
    if p["validator_rejection_reasons"]:
        for k, v in sorted(p["validator_rejection_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines += ["", f"**Final-output failures:** {len(p['final_failures'])}"]
    lines += [f"- {f}" for f in p["final_failures"][:20]]
    f = p["finder_stats"]
    lines += [
        "",
        "### Where the candidates came from, and what the model added",
        "",
        f"- finder `{f['finder']}`, scorer `{f['scorer']}`, lexicon `{f['lexicon_source']}`, "
        f"sources of ranked promos: `{json.dumps(f['sources'])}`",
    ]
    if f["windows_found"]:
        lines.append(
            f"- the curve proposed **{f['windows_found']}** windows; the model dropped "
            f"**{f['dropped_by_model']}** ({f['share_dropped']:.0%}) and trimmed "
            f"**{f['trimmed_by_model']}** of the ranked ones"
        )
        if f["mean_curve_score_dropped"] is not None and f["mean_curve_score_kept"] is not None:
            lines.append(
                f"- mean curve score: kept {f['mean_curve_score_kept']} vs dropped "
                f"{f['mean_curve_score_dropped']} (if close, the model is rejecting windows the curve cannot tell apart)"
            )
    if f["spearman_curve_vs_scorer"] is not None:
        lines.append(
            f"- **Spearman correlation, curve score vs final scorer: {f['spearman_curve_vs_scorer']}** "
            f"(n={f['n_pairs']}). High = the free curve alone already orders clips much like the model; "
            f"low = the model is adding independent signal. Neither is 'good'; it tells you what the API spend buys."
        )
    if p["panel_stats"]:
        s = p["panel_stats"]
        lines += [
            "",
            "### Viewer-panel discrimination (does it separate clips, or rate everything alike?)",
            "",
            f"- retention mean {s['retention_mean']}, sd {s['retention_sd']} over {s['n']} clips "
            f"(sd near 0 = the panel is not discriminating)",
            f"- mean hook survival {s['mean_hook_survival']}, watch-through {s['mean_watch_through']}, "
            f"tap-next {s['mean_tap_next']}",
            f"- share of clips where all viewers survive 3s: {s['share_all_viewers_survive_3s']:.0%} "
            f"(very high = the simulated viewers are too generous)",
            "",
            "| persona | tap-next mean | tap-next sd | watches to end |",
            "|---|---|---|---|",
        ]
        for pid, st in s["per_persona"].items():
            lines.append(f"| {pid} | {st['tap_mean']} | {st['tap_sd']} | {st['share_watch_to_end']:.0%} |")
        lines.append("")
        lines.append("A persona with sd near 0, or that watches to the end every time, adds a constant, "
                     "not signal. Treat the panel's ordering as driven by the personas that move.")
    if p["judge_score_distribution"]:
        lines += ["", f"**Judge score distribution** (legacy scorer): `{json.dumps(p['judge_score_distribution'])}`"]
    lines += [
        "",
        f"**Selector vs judge disagreement** on flags: `{json.dumps(p['selector_vs_judge_flag_disagreements'])}`",
        f"**Share of top-10 from early episodes (<=15):** {p['early_ep_share_top10']:.0%}",
        "",
    ]
    return "\n".join(lines)


NOT_JUDGEABLE = (
    "\n## 3. Not judgeable from screenplay text\n\n"
    "Visual impact of the rendered footage, actor/voice performance, pacing, music and sound, "
    "and whether the AI-generated video at a given timestamp is clean. A line that reads "
    "shocking can land flat on screen. Treat every ranking as a shortlist for an editor, "
    "not a verdict.\n"
)


# --------------------------------------------------------------------------
# 2. blind human sample
# --------------------------------------------------------------------------

def baseline_clip(ep: Dict[str, Any]) -> Dict[str, Any]:
    """The naive editor heuristic: the last ~40s, snapped so no line is chopped."""
    end = ep["duration_s"]
    start, end = snap_window(ep["beats"], max(0.0, end - BASELINE_S), end)
    return {"episode": ep["episode_no"], "start_s": start, "end_s": end,
            "start_ts": seconds_to_ts(start), "end_ts": seconds_to_ts(end),
            "duration_s": round(end - start, 2)}


def make_sample(out_dir: str, episodes: List[Dict[str, Any]], seed: int = 7) -> None:
    by_ep = {e["episode_no"]: e for e in episodes}
    with open(os.path.join(out_dir, "promos.json"), encoding="utf-8") as fh:
        ranked = json.load(fh)["ranked"]
    if len(ranked) < 15:
        print(f"only {len(ranked)} ranked promos; sample will be smaller than 15")

    rng = random.Random(seed)
    top = ranked[:5]
    bottom = ranked[-5:] if len(ranked) >= 10 else []
    middle = ranked[5:-5] if len(ranked) > 10 else []
    rand = rng.sample(middle, min(5, len(middle)))
    picked = [("top", c) for c in top] + [("random", c) for c in rand] + [("bottom", c) for c in bottom]

    items = []
    n_same = 0
    for bucket, c in picked:
        tool = {"source": "tool", "bucket": bucket, "promo_id": c["id"], "episode": c["episode"],
                "start_s": c["start_s"], "end_s": c["end_s"], "start_ts": c["start_ts"], "end_ts": c["end_ts"]}
        b = baseline_clip(by_ep[c["episode"]])
        if _iou(c, b) >= SAME_WINDOW_IOU:
            # The tool simply rediscovered the naive heuristic. Showing the same
            # text twice would waste effort and could be rated inconsistently, so
            # it appears once and counts for both sides (a tie by construction).
            tool["source"] = "both"
            items.append(tool)
            n_same += 1
            continue
        items.append(tool)
        items.append({"source": "baseline", "bucket": bucket, "promo_id": f"baseline-ep{c['episode']}",
                      "episode": c["episode"], **{k: b[k] for k in ("start_s", "end_s", "start_ts", "end_ts")}})

    rng.shuffle(items)
    key: Dict[str, Any] = {}
    md = [
        "# Blind rating sheet",
        "",
        "Rate each clip 0-2 on hook / cliffhanger / standalone, as if you were a stranger scrolling past.",
        "0 = fails, 1 = okay, 2 = strong. Fill `ratings.csv`. Do NOT open `blind_key.json` until done.",
        "Contains screenplay text -- do not publish.",
        "",
    ]
    for i, it in enumerate(items, 1):
        cid = f"clip{i:02d}"
        key[cid] = {k: it[k] for k in ("source", "bucket", "promo_id", "episode", "start_ts", "end_ts")}
        ep = by_ep[it["episode"]]
        md.append(f"## {cid}  ({it['end_s'] - it['start_s']:.0f}s)")
        md.append("")
        md.append("```")
        md.extend(clip_lines(ep, it))
        md.append("```")
        md.append("")

    ev = os.path.join(out_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "blind_sample.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    with open(os.path.join(ev, "blind_key.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)
    with open(os.path.join(ev, "ratings.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "hook_0_2", "cliffhanger_0_2", "standalone_0_2", "failure_type", "notes"])
        for cid in key:
            w.writerow([cid, "", "", "", "", ""])
    print(f"wrote {len(items)} clips to {ev} ({len(picked)} tool picks; "
          f"{n_same} coincide with the last-40s baseline and are shown once)")
    print("failure_type options:", ", ".join(FAILURE_TYPES))


def score_ratings(out_dir: str) -> str:
    ev = os.path.join(out_dir, "eval")
    key = json.load(open(os.path.join(ev, "blind_key.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(ev, "ratings.csv"), encoding="utf-8")))
    rated = []
    for r in rows:
        try:
            h, k, s = int(r["hook_0_2"]), int(r["cliffhanger_0_2"]), int(r["standalone_0_2"])
        except (ValueError, TypeError):
            continue
        rated.append({**key[r["clip_id"]], "h": h, "k": k, "s": s,
                      "total": h + k + s, "failure": (r.get("failure_type") or "none").strip() or "none"})
    if not rated:
        return "No ratings filled in yet."

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else float("nan")

    lines = ["## 2. Blind human rating (small N -- indicative, not statistical)", "",
             f"Rated clips: **{len(rated)}** (labels hidden at rating time, shuffled).", "",
             "| group | n | hook | cliffhanger | standalone | total /6 |", "|---|---|---|---|---|---|"]
    def is_tool(r):
        return r["source"] in ("tool", "both")

    def is_base(r):
        return r["source"] in ("baseline", "both")

    groups = {
        "tool (all)": [r for r in rated if is_tool(r)],
        "baseline: last 40s": [r for r in rated if is_base(r)],
    }
    for b in ("top", "random", "bottom"):
        groups[f"tool / {b}"] = [r for r in rated if is_tool(r) and r["bucket"] == b]
    for name, g in groups.items():
        if g:
            lines.append(f"| {name} | {len(g)} | {mean([r['h'] for r in g])} | {mean([r['k'] for r in g])} | "
                         f"{mean([r['s'] for r in g])} | {mean([r['total'] for r in g])} |")

    by_ep: Dict[int, Dict[str, int]] = {}
    for r in rated:
        slot = by_ep.setdefault(r["episode"], {})
        if is_tool(r):
            slot["tool"] = r["total"]
        if is_base(r):
            slot["baseline"] = r["total"]
    paired = [v for v in by_ep.values() if "tool" in v and "baseline" in v]
    same = sum(1 for r in rated if r["source"] == "both")
    if paired:
        win = sum(1 for v in paired if v["tool"] > v["baseline"])
        tie = sum(1 for v in paired if v["tool"] == v["baseline"])
        loss = sum(1 for v in paired if v["tool"] < v["baseline"])
        lines += ["", f"**Paired by episode, tool vs last-40s baseline:** tool wins {win}, ties {tie}, "
                      f"baseline wins {loss} (n={len(paired)}). "
                      f"In {same} of these the tool picked the baseline window itself (counted as a tie)."]

    lines += ["", "**Failure taxonomy** (what went wrong, by who picked it):", "",
              "| failure type | tool | baseline |", "|---|---|---|"]
    ft = {t: Counter() for t in ("tool", "baseline")}
    for r in rated:
        if is_tool(r):
            ft["tool"][r["failure"]] += 1
        if is_base(r):
            ft["baseline"][r["failure"]] += 1
    for t in FAILURE_TYPES:
        if ft["tool"][t] or ft["baseline"][t]:
            lines.append(f"| {t} | {ft['tool'][t]} | {ft['baseline'][t]} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def write_metrics(out_dir: str, episodes: List[Dict[str, Any]]) -> str:
    text = "# Quality report\n\n" + render_programmatic(programmatic(out_dir, episodes))
    human = os.path.join(out_dir, "eval", "ratings.csv")
    if os.path.exists(human) and os.path.exists(os.path.join(out_dir, "eval", "blind_key.json")):
        text += "\n" + score_ratings(out_dir)
    else:
        text += "\n## 2. Blind human rating\n\n_Not yet run. See `python src/evaluate.py sample`._\n"
    text += NOT_JUDGEABLE
    path = os.path.join(out_dir, "metrics.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate promo output honestly.")
    ap.add_argument("command", choices=["check", "sample", "score"])
    ap.add_argument("--input", required=True, help="screenplay folder/zip (same as the run)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    eps = load_episodes(a.input)
    if a.command == "check":
        print(f"wrote {write_metrics(a.out, eps)}")
    elif a.command == "sample":
        make_sample(a.out, eps, a.seed)
    else:
        print(score_ratings(a.out))
        print(f"wrote {write_metrics(a.out, eps)}")
